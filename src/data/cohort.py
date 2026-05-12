"""
Cohort extraction from MIMIC-IV using DuckDB.

Defines the ICU patient cohort used for training and inference:
- All adult ICU stays with minimum length of stay
- Sepsis-3 labels from ICD-10 diagnosis codes
- Temporal alignment: label onset mapped back by prediction_horizon_hours
"""

from pathlib import Path

import duckdb
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file and return as dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a new in-memory DuckDB connection."""
    return duckdb.connect()


def extract_cohort(cfg: dict | None = None) -> pd.DataFrame:
    """
    Build the base ICU cohort.

    Returns a DataFrame with one row per ICU stay containing:
        stay_id, subject_id, hadm_id, intime, outtime, los,
        first_careunit, age, gender, sepsis_label, sepsis_onset_proxy
    """
    if cfg is None:
        cfg = load_config()

    icu_path = cfg["data"]["icu_path"]
    hosp_path = cfg["data"]["hosp_path"]
    min_los = cfg["cohort"]["min_icu_los_hours"]
    horizon = cfg["cohort"]["prediction_horizon_hours"]

    con = get_connection()

    query = f"""
    WITH icu AS (
        SELECT
            i.stay_id,
            i.subject_id,
            i.hadm_id,
            i.intime,
            i.outtime,
            i.los,
            i.first_careunit
        FROM read_csv_auto('{icu_path}/icustays.csv.gz') i
        WHERE i.los * 24 >= {min_los}
    ),
    patients AS (
        SELECT subject_id, gender, anchor_age AS age
        FROM read_csv_auto('{hosp_path}/patients.csv.gz')
        WHERE anchor_age >= 18
    ),
    sepsis_dx AS (
        SELECT DISTINCT hadm_id,
               MIN(seq_num) AS sepsis_seq
        FROM read_csv_auto('{hosp_path}/diagnoses_icd.csv.gz')
        WHERE icd_version = 10
          AND (icd_code LIKE 'A41%' OR icd_code LIKE 'R652%')
        GROUP BY hadm_id
    )
    SELECT
        icu.stay_id,
        icu.subject_id,
        icu.hadm_id,
        icu.intime,
        icu.outtime,
        icu.los,
        icu.first_careunit,
        p.age,
        p.gender,
        CASE WHEN s.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS sepsis_label,
        -- sepsis_onset_proxy: intime + horizon retained for future prospective
        -- label alignment. Not used in current feature extraction (features.py
        -- always queries the full 24h window from intime regardless of horizon).
        CASE
            WHEN s.hadm_id IS NOT NULL
            THEN icu.intime + INTERVAL ({horizon}) HOUR
            ELSE NULL
        END AS sepsis_onset_proxy
    FROM icu
    JOIN patients p ON icu.subject_id = p.subject_id
    LEFT JOIN sepsis_dx s ON icu.hadm_id = s.hadm_id
    """

    result = con.execute(query).df()
    con.close()

    print(
        f"Cohort: {len(result):,} ICU stays | "
        f"{result['sepsis_label'].sum():,} sepsis "
        f"({result['sepsis_label'].mean():.1%})"
    )
    return result


def save_cohort(cohort: pd.DataFrame, cfg: dict | None = None) -> Path:
    """Save cohort DataFrame to parquet and return the output path."""
    if cfg is None:
        cfg = load_config()
    out = Path(cfg["data"]["processed_path"]) / "cohort.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out, index=False)
    print(f"Saved cohort to {out}")
    return out


if __name__ == "__main__":
    config = load_config()
    cohort_df = extract_cohort(config)
    save_cohort(cohort_df, config)
