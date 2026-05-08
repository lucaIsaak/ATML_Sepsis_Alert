"""
PatientBuffer — per-patient rolling window for streaming inference.

Maintains a sliding time window of observations for a single ICU patient.
When new data arrives (vitals or labs), the buffer updates and recomputes
features on demand.

This is the bridge between batch (MIMIC training) and streaming (production):
  - Training: replay historical MIMIC events through the buffer
  - Production: push incoming FHIR Observations into the buffer

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


@dataclass
class Observation:
    """A single timestamped measurement."""

    timestamp: datetime
    item_name: str          # e.g. "heart_rate", "lactate"
    value: float
    source: str = "icu"     # "icu" (chartevents) | "lab" (labevents) | "fhir"
    tier: int = 1           # 1 = always available, 2 = conditional


@dataclass
class PatientBuffer:
    """
    Rolling window of observations for one ICU patient.

    Usage:
        buf = PatientBuffer(stay_id="39553978", window_hours=24)
        buf.add(Observation(timestamp=..., item_name="heart_rate", value=95))
        features = buf.extract_features()
        # → dict ready for model.predict_patient()
    """

    stay_id: str
    window_hours: int = 24
    observations: deque = field(default_factory=deque)

    # Alert state
    last_risk_score: Optional[float] = None
    last_alert_time: Optional[datetime] = None
    alert_count: int = 0
    acknowledged: bool = False

    def add(self, obs: Observation) -> None:
        """Add a new observation and prune old ones outside the window."""
        self.observations.append(obs)
        self._prune()

    def add_batch(self, observations: list[Observation]) -> None:
        """Add multiple observations at once, then prune the window."""
        for obs in observations:
            self.observations.append(obs)
        self._prune()

    def _prune(self) -> None:
        """Remove observations older than window_hours."""
        if not self.observations:
            return
        latest = max(o.timestamp for o in self.observations)
        cutoff = latest - timedelta(hours=self.window_hours)
        while self.observations and self.observations[0].timestamp < cutoff:
            self.observations.popleft()

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

    def push(self, stay_id: str, obs: Observation) -> None:
        """Route a single observation into the correct patient buffer."""
        buf = self.get_or_create(stay_id)
        buf.add(obs)

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
