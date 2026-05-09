"""
Feature engineering from MIMIC-IV chart events and lab events.

For each ICU stay, extracts time-series features in a lookback window
(default 24h) before a reference timestamp.

Vitals from chartevents (MIMIC-IV item IDs):
    220045 Heart Rate
    220050 Arterial Blood Pressure systolic
    220051 Arterial Blood Pressure diastolic
    220052 Arterial Blood Pressure mean
    220210 Respiratory Rate
    223761 Temperature Fahrenheit
    220277 SpO2

Labs from labevents (MIMIC-IV item IDs):
    50813 Lactate
    51301 White Blood Cells
    50912 Creatinine
    50885 Bilirubin, Total
    51265 Platelet Count
    50882 Bicarbonate
    50902 Chloride
    50931 Glucose
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml


VITAL_ITEMS = {
    220045: "heart_rate",
    220052: "map",           # mean arterial pressure
    220210: "resp_rate",
    223761: "temperature_f",
    220277: "spo2",
}

LAB_ITEMS = {
    50813: "lactate",
    51301: "wbc",
    50912: "creatinine",
    50885: "bilirubin",
    51265: "platelets",
    50882: "bicarbonate",
    50931: "glucose",
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file and return as dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compute_trend(sorted_times: pd.Series, values: pd.Series) -> float:
    """
    Compute the linear slope of a time series (units per hour).

    Returns 0.0 when fewer than 3 points are available or the fit fails.
    Positive slope = worsening for most clinical markers.
    """
    if len(values) < 3:
        return 0.0
    hours = (sorted_times - sorted_times.iloc[0]).dt.total_seconds() / 3600
    try:
        slope, _ = np.polyfit(hours.values, values.values, 1)
        return float(slope)
    except np.linalg.LinAlgError:
        return 0.0


def extract_features(cohort: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    For each stay in cohort, extract aggregated features from the lookback window.

    Returns one row per stay_id with columns:
        stay_id + all feature columns + sepsis_label
    """
    if cfg is None:
        cfg = load_config()

    icu_path = cfg["data"]["icu_path"]
    hosp_path = cfg["data"]["hosp_path"]
    window_h = cfg["cohort"]["lookback_window_hours"]

    con = duckdb.connect()
    con.register("cohort", cohort)

    vital_item_ids = list(VITAL_ITEMS.keys())
    lab_item_ids = list(LAB_ITEMS.keys())

    # --- Vitals ---
    vitals_query = f"""
    SELECT
        c.stay_id,
        ce.itemid,
        ce.valuenum,
        ce.charttime
    FROM read_csv_auto('{icu_path}/chartevents.csv.gz', ignore_errors=true) ce
    JOIN cohort c ON ce.stay_id = c.stay_id
    WHERE ce.itemid IN ({','.join(map(str, vital_item_ids))})
      AND ce.valuenum IS NOT NULL
      AND ce.valuenum > 0
      AND ce.charttime >= c.intime
      AND ce.charttime <= c.intime + INTERVAL ({window_h}) HOUR
    """

    print("Extracting vitals... (this may take a few minutes for large datasets)")
    vitals_raw = con.execute(vitals_query).df()

    # --- Labs ---
    labs_query = f"""
    SELECT
        le.hadm_id,
        c.stay_id,
        le.itemid,
        le.valuenum,
        le.charttime
    FROM read_csv_auto('{hosp_path}/labevents.csv.gz') le
    JOIN cohort c ON le.hadm_id = c.hadm_id
    WHERE le.itemid IN ({','.join(map(str, lab_item_ids))})
      AND le.valuenum IS NOT NULL
      AND le.valuenum >= 0
      AND le.charttime >= c.intime
      AND le.charttime <= c.intime + INTERVAL ({window_h}) HOUR
    """

    print("Extracting labs... (this may take a few minutes)")
    labs_raw = con.execute(labs_query).df()
    con.close()

    # Convert charttimes to datetime for trend computation
    vitals_raw["charttime"] = pd.to_datetime(vitals_raw["charttime"])
    labs_raw["charttime"]   = pd.to_datetime(labs_raw["charttime"])

    # --- Aggregate into feature matrix ---
    features = _aggregate_vitals(vitals_raw)
    lab_features = _aggregate_labs(labs_raw)

    result = cohort[["stay_id", "hadm_id", "age", "gender", "sepsis_label"]].copy()
    result = result.merge(features, on="stay_id", how="left")
    result = result.merge(lab_features, on="stay_id", how="left")

    # Encode gender
    result["gender_male"] = (result["gender"] == "M").astype(int)
    result = result.drop(columns=["gender"])

    print(f"Feature matrix: {result.shape[0]:,} rows x {result.shape[1]} cols")
    return result


def _aggregate_vitals(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw chartevents into per-stay summary statistics including trend."""
    records = []
    for stay_id, group in raw_df.groupby("stay_id"):
        row = {"stay_id": stay_id}
        for item_id, name in VITAL_ITEMS.items():
            subset = group[group["itemid"] == item_id].sort_values("charttime")
            vals = subset["valuenum"].dropna()
            if len(vals) > 0:
                row[f"{name}_mean"]  = vals.mean()
                row[f"{name}_min"]   = vals.min()
                row[f"{name}_max"]   = vals.max()
                row[f"{name}_last"]  = vals.iloc[-1]
                row[f"{name}_trend"] = _compute_trend(subset["charttime"], vals)
            else:
                for suffix in ["mean", "min", "max", "last", "trend"]:
                    row[f"{name}_{suffix}"] = np.nan
        records.append(row)
    return pd.DataFrame(records)


def _aggregate_labs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw labevents into per-stay summary statistics including trend."""
    records = []
    for stay_id, group in raw_df.groupby("stay_id"):
        row = {"stay_id": stay_id}
        for item_id, name in LAB_ITEMS.items():
            subset = group[group["itemid"] == item_id].sort_values("charttime")
            vals = subset["valuenum"].dropna()
            if len(vals) > 0:
                row[f"{name}_last"]  = vals.iloc[-1]
                row[f"{name}_mean"]  = vals.mean()
                row[f"{name}_delta"] = vals.iloc[-1] - vals.iloc[0] if len(vals) > 1 else 0.0
                row[f"{name}_trend"] = _compute_trend(subset["charttime"], vals)
            else:
                for suffix in ["last", "mean", "delta", "trend"]:
                    row[f"{name}_{suffix}"] = np.nan
        records.append(row)
    return pd.DataFrame(records)


def save_features(feature_df: pd.DataFrame, cfg: dict | None = None) -> Path:
    """Save feature matrix to parquet and return the output path."""
    if cfg is None:
        cfg = load_config()
    out = Path(cfg["data"]["processed_path"]) / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out, index=False)
    print(f"Saved features to {out}")
    return out


if __name__ == "__main__":
    config = load_config()
    cohort_df = pd.read_parquet(Path(config["data"]["processed_path"]) / "cohort.parquet")
    feature_matrix = extract_features(cohort_df, config)
    save_features(feature_matrix, config)
