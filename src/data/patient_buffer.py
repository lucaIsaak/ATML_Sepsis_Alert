"""
PatientBuffer — per-patient rolling window for streaming inference.

Maintains a sliding time window of observations for a single ICU patient.
When new data arrives (vitals or labs), the buffer updates and recomputes
features on demand.

This is the bridge between batch (MIMIC training) and streaming (production):
  - Training: replay historical MIMIC events through the buffer
  - Production: push incoming FHIR Observations into the buffer

Two explicit ingestion streams:
  STREAMING — vitals (heart rate, MAP, SpO2, resp. rate, temperature)
              arrive continuously from bedside monitoring devices.
              Update frequency: every few seconds to minutes.

  BATCH     — labs (lactate, WBC, creatinine, etc.) arrive once when
              a blood draw is processed by the lab.
              Update frequency: once or twice per day.

Two tiers of features:
  Tier 1 — Always available (model works without these missing)
            Vitals (HR, MAP, SpO2, RR, Temp) + Daily BMP/CBC
  Tier 2 — Conditionally available (improve prediction when present)
            Lactate, Bilirubin, BUN — ordered on clinical suspicion
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# MIMIC-IV item IDs → feature names                                    #
# (same as features.py — single source of truth here)                 #
# ------------------------------------------------------------------ #

# Tier 1: measured every shift, always available
TIER1_VITAL_ITEMS = {
    220045: "heart_rate",
    220050: "sbp",
    220052: "map",
    220210: "resp_rate",
    223761: "temperature_f",
    220277: "spo2",
}

TIER1_LAB_ITEMS = {
    51301: "wbc",
    51222: "hemoglobin",
    51265: "platelets",
    50983: "sodium",
    50971: "potassium",
    50912: "creatinine",
    50931: "glucose",
    50882: "bicarbonate",
}

# Tier 2: ordered on clinical suspicion
TIER2_LAB_ITEMS = {
    50813: "lactate",
    50885: "bilirubin",
    51006: "bun",
    50889: "crp",           # C-reactive protein (if available)
}

ALL_VITAL_ITEMS = {**TIER1_VITAL_ITEMS}
ALL_LAB_ITEMS   = {**TIER1_LAB_ITEMS, **TIER2_LAB_ITEMS}

# Which item names belong to each stream
VITAL_NAMES = set(ALL_VITAL_ITEMS.values())
LAB_NAMES   = set(ALL_LAB_ITEMS.values())

# ------------------------------------------------------------------ #
# Clinical threshold definitions                                        #
# Based on Sepsis-3 criteria and standard ICU reference ranges         #
# ------------------------------------------------------------------ #

VITAL_THRESHOLDS: dict[str, list[dict]] = {
    "map": [
        {"threshold": 65.0,  "direction": "below", "severity": "critical",
         "message": "MAP {value:.0f} mmHg — below Sepsis-3 threshold (65 mmHg)"},
        {"threshold": 70.0,  "direction": "below", "severity": "warning",
         "message": "MAP {value:.0f} mmHg — approaching hypotension threshold"},
    ],
    "heart_rate": [
        {"threshold": 130.0, "direction": "above", "severity": "critical",
         "message": "HR {value:.0f} bpm — severe tachycardia"},
        {"threshold": 100.0, "direction": "above", "severity": "warning",
         "message": "HR {value:.0f} bpm — tachycardia"},
    ],
    "spo2": [
        {"threshold": 90.0,  "direction": "below", "severity": "critical",
         "message": "SpO2 {value:.0f}% — severe hypoxia"},
        {"threshold": 94.0,  "direction": "below", "severity": "warning",
         "message": "SpO2 {value:.0f}% — below normal range"},
    ],
    "resp_rate": [
        {"threshold": 30.0,  "direction": "above", "severity": "critical",
         "message": "RR {value:.0f}/min — severe tachypnea"},
        {"threshold": 22.0,  "direction": "above", "severity": "warning",
         "message": "RR {value:.0f}/min — tachypnea (Sepsis-3 criterion)"},
    ],
    "temperature_f": [
        {"threshold": 100.4, "direction": "above", "severity": "warning",
         "message": "Temp {value:.1f}°F — fever"},
        {"threshold": 96.8,  "direction": "below", "severity": "warning",
         "message": "Temp {value:.1f}°F — hypothermia"},
    ],
}


# ------------------------------------------------------------------ #
# Vital threshold alert                                                 #
# ------------------------------------------------------------------ #

@dataclass
class VitalThresholdAlert:
    """A single clinical threshold violation detected in raw vital data."""

    vital: str          # e.g. "map", "heart_rate"
    value: float        # current value
    threshold: float    # the threshold that was crossed
    direction: str      # "below" | "above"
    severity: str       # "warning" | "critical"
    message: str        # human-readable description


# ------------------------------------------------------------------ #
# Stream type                                                          #
# ------------------------------------------------------------------ #

class StreamType(Enum):
    """
    Ingestion stream for an observation.

    STREAMING — continuous feed from bedside monitoring devices.
                Vitals only. Update every seconds to minutes.

    BATCH     — periodic batch from the hospital lab system.
                Labs only. Update once or twice per day when a
                blood draw is processed.
    """
    STREAMING = "streaming"
    BATCH     = "batch"


# ------------------------------------------------------------------ #
# Observation                                                          #
# ------------------------------------------------------------------ #

@dataclass
class Observation:
    """A single timestamped measurement."""

    timestamp: datetime
    item_name: str          # e.g. "heart_rate", "lactate"
    value: float
    source: str = "icu"     # "icu" (chartevents) | "lab" (labevents) | "fhir"
    tier: int = 1           # 1 = always available, 2 = conditional
    stream_type: StreamType = StreamType.STREAMING  # STREAMING | BATCH
    # Set by the stream simulator for routing and evaluation
    stay_id: Optional[str] = None
    sepsis_label: Optional[int] = None


# ------------------------------------------------------------------ #
# PatientBuffer                                                        #
# ------------------------------------------------------------------ #

@dataclass
class PatientBuffer:
    """
    Rolling window of observations for one ICU patient.

    Usage:
        buf = PatientBuffer(stay_id="39553978", window_hours=24)
        buf.add_vital(Observation(timestamp=..., item_name="heart_rate", value=95))
        buf.add_lab(Observation(timestamp=..., item_name="lactate", value=2.1,
                                stream_type=StreamType.BATCH))
        features = buf.extract_features()
        # → dict ready for model.predict_patient()
    """

    stay_id: str
    window_hours: int = 24
    observations: deque = field(default_factory=deque)

    # ---------------------------------------------------------------- #
    # Ingestion — typed by stream                                        #
    # ---------------------------------------------------------------- #

    def add_vital(self, obs: Observation) -> None:
        """
        Add a streaming vital observation (heart rate, MAP, SpO2, etc.).

        Called on every update from the bedside monitor feed.
        Automatically sets stream_type to STREAMING.
        """
        obs.stream_type = StreamType.STREAMING
        self.add(obs)

    def add_lab(self, obs: Observation) -> None:
        """
        Add a batch lab result (lactate, WBC, creatinine, etc.).

        Called once when the hospital lab system reports a new result.
        Automatically sets stream_type to BATCH.
        """
        obs.stream_type = StreamType.BATCH
        self.add(obs)

    def add(self, obs: Observation) -> None:
        """Add a new observation and prune old ones outside the window."""
        self.observations.append(obs)
        self._prune()

    def add_batch(self, observations: list[Observation]) -> None:
        """Add multiple observations at once, then prune the window."""
        for obs in observations:
            self.observations.append(obs)
        self._prune()

    # ---------------------------------------------------------------- #
    # Internal                                                           #
    # ---------------------------------------------------------------- #

    def _prune(self) -> None:
        """Remove observations older than window_hours."""
        if not self.observations:
            return
        latest = max(o.timestamp for o in self.observations)
        cutoff = latest - timedelta(hours=self.window_hours)
        while self.observations and self.observations[0].timestamp < cutoff:
            self.observations.popleft()

    # ---------------------------------------------------------------- #
    # Feature extraction                                                 #
    # ---------------------------------------------------------------- #

    def extract_features(self) -> dict[str, float]:
        """
        Compute the feature dict from current window.

        Returns the same feature schema the model was trained on:
          {feature_name_stat: value} e.g. {"heart_rate_mean": 88.5, ...}

        Missing values are np.nan — the model handles these natively.
        """
        features: dict[str, float] = {}

        all_items = {**ALL_VITAL_ITEMS, **ALL_LAB_ITEMS}
        item_names = set(all_items.values())

        for item_name in item_names:
            vals_with_time = [
                (o.timestamp, o.value)
                for o in self.observations
                if o.item_name == item_name and not np.isnan(o.value)
            ]

            if not vals_with_time:
                for stat in ["mean", "min", "max", "last", "delta", "trend"]:
                    features[f"{item_name}_{stat}"] = np.nan
                continue

            vals_with_time.sort(key=lambda x: x[0])
            vals = [v for _, v in vals_with_time]
            arr = np.array(vals)

            features[f"{item_name}_mean"]  = float(np.mean(arr))
            features[f"{item_name}_min"]   = float(np.min(arr))
            features[f"{item_name}_max"]   = float(np.max(arr))
            features[f"{item_name}_last"]  = float(arr[-1])

            # Delta: last - first (absolute change over window)
            features[f"{item_name}_delta"] = float(arr[-1] - arr[0]) if len(arr) > 1 else 0.0

            # Trend: slope of linear fit (positive = worsening for most markers)
            if len(arr) >= 3:
                hours = np.array([(t - vals_with_time[0][0]).total_seconds() / 3600
                                  for t, _ in vals_with_time])
                try:
                    slope, _ = np.polyfit(hours, arr, 1)
                    features[f"{item_name}_trend"] = float(slope)
                except np.linalg.LinAlgError:
                    features[f"{item_name}_trend"] = 0.0
            else:
                features[f"{item_name}_trend"] = 0.0

        return features

    def extract_multi_window_features(
        self, windows: list[int] | None = None
    ) -> dict[str, float]:
        """
        Extract features for multiple time windows.

        Returns all features prefixed by window size.
        e.g. {"6h_heart_rate_mean": ..., "24h_heart_rate_mean": ...}

        Used by the multi-window stacking model.
        """
        if windows is None:
            windows = [6, 12, 24]

        all_features = {}
        original_window = self.window_hours

        for window in windows:
            self.window_hours = window
            self._prune()
            feats = self.extract_features()
            for k, v in feats.items():
                all_features[f"{window}h_{k}"] = v

        self.window_hours = original_window
        self._prune()
        return all_features

    # ---------------------------------------------------------------- #
    # Stream-aware queries                                               #
    # ---------------------------------------------------------------- #

    def get_streaming_vitals(self) -> list[Observation]:
        """Return all current streaming (vital) observations in the window."""
        return [o for o in self.observations if o.stream_type == StreamType.STREAMING]

    def get_batch_labs(self) -> list[Observation]:
        """Return all current batch (lab) observations in the window."""
        return [o for o in self.observations if o.stream_type == StreamType.BATCH]

    def last_lab_time(self) -> Optional[datetime]:
        """Return timestamp of the most recent lab result, or None."""
        labs = self.get_batch_labs()
        if not labs:
            return None
        return max(o.timestamp for o in labs)

    def hours_since_last_lab(self) -> Optional[float]:
        """Return hours since the last lab result, or None if no labs yet."""
        t = self.last_lab_time()
        if t is None:
            return None
        latest = max(o.timestamp for o in self.observations)
        return (latest - t).total_seconds() / 3600

    # ---------------------------------------------------------------- #
    # Raw vital trend tracking                                           #
    # ---------------------------------------------------------------- #

    def get_vital_trajectory(
        self, vital_name: str, hours: float = 4.0
    ) -> list[tuple[datetime, float]]:
        """
        Return the raw time-series for a vital over the last N hours.

        Returns a list of (timestamp, value) tuples sorted oldest→newest.
        Used by the narrative agent to reason about trajectory shape,
        not just the aggregated slope.

        Example:
            trajectory = buf.get_vital_trajectory("map", hours=6)
            # → [(t0, 78), (t1, 72), (t2, 65), (t3, 61)]
            # Shows MAP declining steadily over 6 hours
        """
        if not self.observations:
            return []
        latest = max(o.timestamp for o in self.observations)
        cutoff = latest - timedelta(hours=hours)
        points = [
            (o.timestamp, o.value)
            for o in self.observations
            if o.item_name == vital_name
            and o.timestamp >= cutoff
            and not np.isnan(o.value)
        ]
        return sorted(points, key=lambda x: x[0])

    def get_sustained_direction(
        self,
        vital_name: str,
        hours: float = 4.0,
        min_points: int = 3,
    ) -> Optional[str]:
        """
        Detect if a vital has been moving consistently in one direction.

        Returns:
            "rising"  — every consecutive pair is higher than the last
            "falling" — every consecutive pair is lower than the last
            None      — mixed / insufficient data

        This catches clinical patterns the ML model misses, e.g.:
          MAP 72 → 68 → 64 → 61 over 4h (steadily falling, not yet critical)
          while the overall trend slope looks mild.
        """
        trajectory = self.get_vital_trajectory(vital_name, hours)
        if len(trajectory) < min_points:
            return None

        values = [v for _, v in trajectory]
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        if all(d > 0 for d in diffs):
            return "rising"
        if all(d < 0 for d in diffs):
            return "falling"
        return None

    def check_vital_thresholds(self) -> list[VitalThresholdAlert]:
        """
        Check the most recent value for each vital against clinical thresholds.

        Returns a list of VitalThresholdAlert for every violated threshold,
        sorted by severity (critical first).

        Called by the narrative agent to enrich alerts with raw vital context
        even when the ML risk score is only MODERATE.

        Example output:
            [VitalThresholdAlert(vital='map', value=62.0, threshold=65.0,
                                 direction='below', severity='critical',
                                 message='MAP 62 mmHg — below Sepsis-3 threshold')]
        """
        alerts: list[VitalThresholdAlert] = []

        for vital_name, thresholds in VITAL_THRESHOLDS.items():
            trajectory = self.get_vital_trajectory(vital_name, hours=self.window_hours)
            if not trajectory:
                continue
            current_value = trajectory[-1][1]  # most recent reading

            for spec in thresholds:
                violated = (
                    (spec["direction"] == "below" and current_value < spec["threshold"])
                    or
                    (spec["direction"] == "above" and current_value > spec["threshold"])
                )
                if violated:
                    alerts.append(VitalThresholdAlert(
                        vital=vital_name,
                        value=current_value,
                        threshold=spec["threshold"],
                        direction=spec["direction"],
                        severity=spec["severity"],
                        message=spec["message"].format(value=current_value),
                    ))
                    break  # only report the most severe violation per vital

        # Critical alerts first
        alerts.sort(key=lambda a: 0 if a.severity == "critical" else 1)
        return alerts

    # ---------------------------------------------------------------- #
    # Properties                                                         #
    # ---------------------------------------------------------------- #

    @property
    def n_observations(self) -> int:
        """Return total number of observations in the buffer."""
        return len(self.observations)

    @property
    def has_tier1_vitals(self) -> bool:
        """True if we have at least some Tier 1 vital signs."""
        tier1_names = set(TIER1_VITAL_ITEMS.values())
        observed_names = {o.item_name for o in self.observations}
        return bool(tier1_names & observed_names)

    @property
    def hours_of_data(self) -> float:
        """Return time span covered by observations in the buffer (hours)."""
        if len(self.observations) < 2:
            return 0.0
        times = [o.timestamp for o in self.observations]
        return (max(times) - min(times)).total_seconds() / 3600


# ------------------------------------------------------------------ #
# BufferRegistry                                                        #
# ------------------------------------------------------------------ #

class BufferRegistry:
    """
    Registry of all active patient buffers.

    One PatientBuffer per active ICU stay.
    The agent holds a single BufferRegistry instance.
    """

    def __init__(self, window_hours: int = 24):
        """Initialise registry with a shared rolling-window length."""
        self.window_hours = window_hours
        self._buffers: dict[str, PatientBuffer] = {}

    def get_or_create(self, stay_id: str) -> PatientBuffer:
        """Return existing buffer for stay_id or create a new one."""
        if stay_id not in self._buffers:
            self._buffers[stay_id] = PatientBuffer(
                stay_id=stay_id,
                window_hours=self.window_hours,
            )
        return self._buffers[stay_id]

    # ---------------------------------------------------------------- #
    # Typed push methods — use these in production                       #
    # ---------------------------------------------------------------- #

    def push_vital(self, stay_id: str, obs: Observation) -> None:
        """
        Route a streaming vital observation into the correct patient buffer.

        Call this when a new heart rate, MAP, SpO2, etc. arrives from
        the bedside monitor feed.
        """
        buf = self.get_or_create(stay_id)
        buf.add_vital(obs)

    def push_lab(self, stay_id: str, obs: Observation) -> None:
        """
        Route a batch lab result into the correct patient buffer.

        Call this when the hospital lab system reports a new blood result.
        """
        buf = self.get_or_create(stay_id)
        buf.add_lab(obs)

    def push(self, stay_id: str, obs: Observation) -> None:
        """
        Generic push — routes by item name automatically.

        Prefer push_vital() / push_lab() when the source is known.
        Falls back to name-based routing for backward compatibility.
        """
        if obs.item_name in VITAL_NAMES:
            self.push_vital(stay_id, obs)
        elif obs.item_name in LAB_NAMES:
            self.push_lab(stay_id, obs)
        else:
            # Unknown item — add as-is
            buf = self.get_or_create(stay_id)
            buf.add(obs)

    # ---------------------------------------------------------------- #
    # Queries                                                            #
    # ---------------------------------------------------------------- #

    def get_buffer(self, stay_id: str) -> Optional[PatientBuffer]:
        """Return the buffer for stay_id, or None if not found."""
        return self._buffers.get(stay_id)

    def get_all_features(self) -> pd.DataFrame:
        """Extract features for all active patients as a DataFrame."""
        rows = []
        for stay_id, buf in self._buffers.items():
            if buf.has_tier1_vitals:
                feats = buf.extract_features()
                feats["stay_id"] = stay_id
                rows.append(feats)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def discharge(self, stay_id: str) -> None:
        """Remove a patient when they leave the ICU."""
        self._buffers.pop(stay_id, None)

    @property
    def active_patients(self) -> list[str]:
        """Return list of stay IDs with active buffers."""
        return list(self._buffers.keys())
