"""
Data drift monitor for SepsisAlert.

Computes Population Stability Index (PSI) for each feature by comparing
the training distribution (features.parquet — 93k patients) against the
live distribution (current predictions in memory).

PSI interpretation (industry standard):
  PSI < 0.10  → stable       — no action needed
  PSI 0.10–0.20 → moderate   — worth monitoring
  PSI > 0.20  → significant  — model may be degraded, review recommended

PSI formula:
  PSI = Σ (live_% - train_%) × ln(live_% / train_%)

The result is logged to logs/drift_history.jsonl (at most once per hour)
so the frontend can draw a PSI trend sparkline.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_DRIFT_LOG = Path("logs/drift_history.jsonl")
_LOG_INTERVAL_HOURS = 1.0   # only append to history once per hour

# Clinical feature labels for display
_FEATURE_LABELS: dict[str, str] = {
    "age":               "Age",
    "heart_rate_mean":   "Heart rate (mean)",
    "heart_rate_last":   "Heart rate (last)",
    "heart_rate_trend":  "Heart rate (trend)",
    "map_mean":          "MAP (mean)",
    "map_min":           "MAP (min)",
    "map_last":          "MAP (last)",
    "map_trend":         "MAP (trend)",
    "resp_rate_mean":    "Resp rate (mean)",
    "resp_rate_last":    "Resp rate (last)",
    "spo2_mean":         "SpO2 (mean)",
    "spo2_min":          "SpO2 (min)",
    "spo2_last":         "SpO2 (last)",
    "temperature_f_last":"Temperature",
    "lactate_last":      "Lactate",
    "lactate_delta":     "Lactate (delta)",
    "creatinine_last":   "Creatinine",
    "creatinine_delta":  "Creatinine (delta)",
    "wbc_last":          "WBC",
    "bilirubin_last":    "Bilirubin",
    "platelets_last":    "Platelets",
    "bicarbonate_last":  "Bicarbonate",
    "glucose_last":      "Glucose",
    "sodium_last":       "Sodium",
    "potassium_last":    "Potassium",
    "bun_last":          "BUN",
    "hemoglobin_last":   "Hemoglobin",
}

# Priority order — clinical vitals first, then labs, then engineered features
_PRIORITY_FEATURES = [
    "map_last", "map_mean", "heart_rate_last", "spo2_min", "resp_rate_last",
    "lactate_last", "creatinine_last", "wbc_last", "temperature_f_last",
    "bilirubin_last", "platelets_last", "bicarbonate_last", "glucose_last",
    "hemoglobin_last", "bun_last",
]


# ------------------------------------------------------------------ #
# PSI core                                                             #
# ------------------------------------------------------------------ #

def compute_psi(
    train_values: np.ndarray,
    live_values: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute PSI between training and live distributions.

    Bins are defined by training data deciles so the expected distribution
    is uniform across bins by construction.

    Returns NaN if there are fewer than 5 live values (not enough data).
    """
    live_values = live_values[~np.isnan(live_values)]
    train_values = train_values[~np.isnan(train_values)]

    if len(live_values) < 5 or len(train_values) < 5:
        return float("nan")

    # Build bin edges from training deciles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.nanpercentile(train_values, percentiles)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    # Remove duplicate edges (can happen with low-cardinality features)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return float("nan")

    eps = 1e-6

    train_counts, _ = np.histogram(train_values, bins=bin_edges)
    live_counts,  _ = np.histogram(live_values,  bins=bin_edges)

    train_pct = train_counts / len(train_values) + eps
    live_pct  = live_counts  / len(live_values)  + eps

    psi = float(np.sum((live_pct - train_pct) * np.log(live_pct / train_pct)))
    return round(psi, 4)


def psi_status(psi: float) -> str:
    """Convert PSI float to status string."""
    if math.isnan(psi):
        return "unknown"
    if psi < 0.10:
        return "stable"
    if psi < 0.20:
        return "moderate"
    return "significant"


# ------------------------------------------------------------------ #
# Full drift report                                                    #
# ------------------------------------------------------------------ #

def compute_drift_report(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    feature_cols: list[str],
    risk_scores_live: np.ndarray,
    risk_scores_train: np.ndarray | None = None,
) -> dict:
    """
    Compute a full drift report comparing training vs live distributions.

    Parameters
    ----------
    train_df        : full training feature DataFrame (features.parquet)
    live_df         : live predictions DataFrame (current patients)
    feature_cols    : list of feature column names to evaluate
    risk_scores_live: array of current patient risk scores

    Returns
    -------
    dict with keys: overall_status, overall_psi, features,
                    risk_distribution, psi_history, evaluated_at
    """
    # ── Feature PSI ───────────────────────────────────────────────
    # Evaluate priority features first, then remainder — cap at 15 total
    ordered = [f for f in _PRIORITY_FEATURES if f in feature_cols]
    rest    = [f for f in feature_cols if f not in ordered]
    eval_features = (ordered + rest)[:15]

    feature_rows: list[dict] = []
    for feat in eval_features:
        if feat not in train_df.columns or feat not in live_df.columns:
            continue

        train_vals = train_df[feat].dropna().values.astype(float)
        live_vals  = live_df[feat].dropna().values.astype(float)

        psi = compute_psi(train_vals, live_vals)
        status = psi_status(psi)

        train_mean = float(np.nanmean(train_vals)) if len(train_vals) else None
        live_mean  = float(np.nanmean(live_vals))  if len(live_vals)  else None

        feature_rows.append({
            "feature":    feat,
            "label":      _FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
            "train_mean": round(train_mean, 2) if train_mean is not None else None,
            "live_mean":  round(live_mean,  2) if live_mean  is not None else None,
            "psi":        psi if not math.isnan(psi) else None,
            "status":     status,
        })

    # Sort worst PSI first
    feature_rows.sort(
        key=lambda r: r["psi"] if r["psi"] is not None else -1,
        reverse=True,
    )

    # ── Overall PSI ───────────────────────────────────────────────
    valid_psi = [r["psi"] for r in feature_rows if r["psi"] is not None]
    overall_psi = round(float(np.mean(valid_psi)), 4) if valid_psi else None
    overall_status = psi_status(overall_psi) if overall_psi is not None else "unknown"

    # ── Risk score distribution ───────────────────────────────────
    def _risk_label(score: float) -> str:
        if score >= 0.8: return "CRITICAL"
        if score >= 0.6: return "HIGH"
        if score >= 0.4: return "MODERATE"
        return "LOW"

    live_labels = [_risk_label(s) for s in risk_scores_live]
    total_live  = len(live_labels) or 1

    live_dist = {
        "CRITICAL": live_labels.count("CRITICAL"),
        "HIGH":     live_labels.count("HIGH"),
        "MODERATE": live_labels.count("MODERATE"),
        "LOW":      live_labels.count("LOW"),
    }
    live_dist_pct = {k: round(v / total_live, 3) for k, v in live_dist.items()}

    # Training expected distribution — computed dynamically if scores provided,
    # otherwise falls back to known MIMIC-IV statistics
    if risk_scores_train is not None and len(risk_scores_train) > 0:
        train_labels = [_risk_label(float(s)) for s in risk_scores_train if not np.isnan(s)]
        total_train = len(train_labels) or 1
        train_dist_pct = {
            k: round(train_labels.count(k) / total_train, 3)
            for k in ["CRITICAL", "HIGH", "MODERATE", "LOW"]
        }
    else:
        train_dist_pct = {"CRITICAL": 0.04, "HIGH": 0.07, "MODERATE": 0.12, "LOW": 0.77}

    # ── Log to history ─────────────────────────────────────────────
    evaluated_at = datetime.now(timezone.utc).isoformat()
    _maybe_log_history(overall_psi, overall_status, evaluated_at)

    return {
        "overall_status":    overall_status,
        "overall_psi":       overall_psi,
        "features":          feature_rows,
        "risk_distribution": {
            "live":      live_dist_pct,
            "expected":  train_dist_pct,
            "live_counts": live_dist,
            "total_live":  total_live,
        },
        "psi_history":   _load_history(),
        "evaluated_at":  evaluated_at,
        "live_patients": total_live,
        "note": None,  # Explanation shown statically in the UI
    }


# ------------------------------------------------------------------ #
# History log                                                          #
# ------------------------------------------------------------------ #

def _maybe_log_history(psi: float | None, status: str, ts: str) -> None:
    """Append to drift history at most once per hour."""
    _DRIFT_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Check time of last entry
    if _DRIFT_LOG.exists():
        last_line = ""
        with _DRIFT_LOG.open(encoding="utf-8") as f:
            for line in f:
                last_line = line.strip()
        if last_line:
            try:
                last = json.loads(last_line)
                last_ts = datetime.fromisoformat(last["ts"])
                now_ts  = datetime.fromisoformat(ts)
                hours_since = (now_ts - last_ts).total_seconds() / 3600
                if hours_since < _LOG_INTERVAL_HOURS:
                    return
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass

    with _DRIFT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts":     ts,
            "psi":    psi,
            "status": status,
        }) + "\n")


def _load_history(max_days: int = 7) -> list[dict]:
    """Load last max_days of drift history for the sparkline."""
    if not _DRIFT_LOG.exists():
        return []
    records: list[dict] = []
    with _DRIFT_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Keep last max_days × 24 entries (one per hour max)
    return records[-(max_days * 24):]
