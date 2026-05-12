"""
Streaming simulator — replays MIMIC-IV data in time order.

ARCHITECTURAL STUB — proves real-time architecture without live hospital data.
=============================================================================
In production, this simulator is replaced by the FHIR adapter
(`src/integrations/fhir_adapter.py`) which pulls live vitals and labs from
a hospital EHR. The PatientBuffer interface is identical in both cases —
swapping simulatated data for live data requires only a configuration change.

A live FHIR feed cannot be used at prototype stage because it requires a
signed hospital partnership agreement and OAuth credentials. This simulator
was built to demonstrate that the streaming architecture is correct without
that prerequisite.

For demo and testing: feeds historical ICU events through the
PatientBuffer exactly as a live FHIR stream would.

Usage:
    sim = MIMICStreamSimulator(n_patients=20, speed_factor=3600)
    for event_batch in sim.stream():
        agent.process_batch(event_batch)
        time.sleep(1)   # 1 real second = 1 ICU hour

See EVALUATION_GUIDE.md for full project scope context.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from src.data.patient_buffer import (
    Observation, BufferRegistry,
    ALL_VITAL_ITEMS, ALL_LAB_ITEMS,
    TIER1_VITAL_ITEMS, TIER1_LAB_ITEMS, TIER2_LAB_ITEMS,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file and return as dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class MIMICStreamSimulator:
    """
    Replays MIMIC-IV chartevents and labevents in time order
    for a sample of ICU patients.

    Simulates what a FHIR subscription webhook would deliver
    in a live hospital deployment.
    """

    def __init__(
        self,
        n_patients: int = 20,
        speed_factor: int = 3600,   # 1 real second = speed_factor ICU seconds
        cfg: dict | None = None,
    ):
        """Initialise simulator with patient count and config."""
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.n_patients = n_patients
        self.speed_factor = speed_factor
        self._events: pd.DataFrame | None = None

    def _sample_patients(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Sample a balanced mix of sepsis and non-sepsis patients."""
        sepsis = cohort[cohort["sepsis_label"] == 1].sample(
            min(self.n_patients // 2, len(cohort[cohort["sepsis_label"] == 1])),
            random_state=42
        )
        non_sepsis = cohort[cohort["sepsis_label"] == 0].sample(
            self.n_patients - len(sepsis),
            random_state=42
        )
        return pd.concat([sepsis, non_sepsis])

    def load_events(self) -> pd.DataFrame:
        """Load and merge chart + lab events for sampled patients, sorted by time."""
        icu_path  = self.cfg["data"]["icu_path"]
        hosp_path = self.cfg["data"]["hosp_path"]
        cohort = pd.read_parquet(
            Path(self.cfg["data"]["processed_path"]) / "cohort.parquet"
        )
        sample = self._sample_patients(cohort)

        stay_ids  = sample["stay_id"].tolist()

        vital_ids = list(ALL_VITAL_ITEMS.keys())
        lab_ids   = list(ALL_LAB_ITEMS.keys())

        con = duckdb.connect()
        con.register("sample_stays", sample[["stay_id", "hadm_id", "sepsis_label"]])

        print(f"Loading events for {len(stay_ids)} patients...")

        # Chart events (vitals)
        vitals = con.execute(f"""
            SELECT
                ce.stay_id,
                ce.charttime AS event_time,
                ce.itemid,
                ce.valuenum AS value,
                'vital' AS event_type
            FROM read_csv_auto('{icu_path}/chartevents.csv.gz', ignore_errors=true) ce
            WHERE ce.stay_id IN ({','.join(map(str, stay_ids))})
              AND ce.itemid IN ({','.join(map(str, vital_ids))})
              AND ce.valuenum IS NOT NULL
              AND ce.valuenum > 0
            ORDER BY ce.charttime
        """).df()

        # Lab events
        labs = con.execute(f"""
            SELECT
                s.stay_id,
                le.charttime AS event_time,
                le.itemid,
                le.valuenum AS value,
                'lab' AS event_type
            FROM read_csv_auto('{hosp_path}/labevents.csv.gz') le
            JOIN sample_stays s ON le.hadm_id = s.hadm_id
            WHERE le.itemid IN ({','.join(map(str, lab_ids))})
              AND le.valuenum IS NOT NULL
              AND le.valuenum >= 0
            ORDER BY le.charttime
        """).df()

        con.close()

        # Map itemid → feature name
        item_name_map = {**ALL_VITAL_ITEMS, **ALL_LAB_ITEMS}
        vitals["item_name"] = vitals["itemid"].map(item_name_map)
        labs["item_name"]   = labs["itemid"].map(item_name_map)

        # Tier labels
        tier1_ids = set(TIER1_VITAL_ITEMS) | set(TIER1_LAB_ITEMS)
        vitals["tier"] = vitals["itemid"].apply(lambda x: 1 if x in tier1_ids else 2)
        labs["tier"]   = labs["itemid"].apply(lambda x: 1 if x in tier1_ids else 2)

        events = pd.concat([vitals, labs], ignore_index=True)
        events["event_time"] = pd.to_datetime(events["event_time"])
        events = events.sort_values("event_time").reset_index(drop=True)

        # Attach sepsis label
        events = events.merge(
            sample[["stay_id", "sepsis_label"]], on="stay_id", how="left"
        )

        print(f"Loaded {len(events):,} events | "
              f"{events['stay_id'].nunique()} patients | "
              f"Time range: {events['event_time'].min()} → {events['event_time'].max()}")

        self._events = events
        return events

    def stream(self, batch_minutes: int = 60):
        """
        Generator that yields batches of events grouped by time bucket.

        Each yield represents 'batch_minutes' of ICU time.
        In production this would be replaced by a FHIR webhook handler.

        Yields:
            tuple[datetime, list[Observation]] — timestamp and new observations
        """
        if self._events is None:
            self.load_events()

        events = self._events
        min_time = events["event_time"].min()
        max_time = events["event_time"].max()
        current = min_time

        total_buckets = int((max_time - min_time).total_seconds() / 60 / batch_minutes)
        print(f"Streaming {total_buckets} time buckets ({batch_minutes}min each)")

        while current < max_time:
            bucket_end = current + timedelta(minutes=batch_minutes)
            mask = (events["event_time"] >= current) & (events["event_time"] < bucket_end)
            bucket = events[mask]

            observations = []
            for _, row in bucket.iterrows():
                if pd.isna(row["item_name"]) or np.isnan(row["value"]):
                    continue
                # Route to correct stream type based on event source
                from src.data.patient_buffer import StreamType  # noqa: PLC0415
                stream_type = (
                    StreamType.STREAMING if row["event_type"] == "vital"
                    else StreamType.BATCH
                )
                obs = Observation(
                    timestamp=row["event_time"].to_pydatetime(),
                    item_name=row["item_name"],
                    value=float(row["value"]),
                    source=row["event_type"],
                    tier=int(row["tier"]),
                    stream_type=stream_type,
                    stay_id=str(int(row["stay_id"])),
                    sepsis_label=int(row["sepsis_label"]),
                )
                observations.append(obs)

            yield current, observations
            current = bucket_end


# Expose BufferRegistry so callers can import it from this module if needed
__all__ = ["MIMICStreamSimulator", "load_config", "BufferRegistry", "TIER2_LAB_ITEMS"]
