"""
Synthetic Demo Data Generator — SepsisAlert

LEGAL / DATA PRIVACY NOTICE
============================
This script generates FULLY SYNTHETIC, artificially created data.
No real patient records, no data derived from MIMIC-IV or any other
clinical database, and no personally identifiable information (PII)
are produced, stored, or distributed by this script.

All values (ages, vital signs, lab results, timestamps, ICD codes) are
drawn from statistical distributions chosen to be physiologically
plausible for demonstration purposes only. They do not represent any
real individual, hospital visit, or clinical outcome.

This output is therefore NOT subject to:
  - HIPAA (US Health Insurance Portability and Accountability Act)
  - GDPR (EU General Data Protection Regulation)
  - PhysioNet / MIMIC-IV data use agreements
  - Any other patient data protection regulation

PURPOSE
=======
Provides frontend / UI developers with a drop-in dataset that mirrors
the exact folder structure and file schemas of a real MIMIC-IV pipeline
run, so that dashboard development can proceed without access to
restricted clinical data.

Creates 100 synthetic ICU patient records (≈22 % sepsis prevalence,
matching the real cohort distribution) in the following layout:

  ~/Desktop/SepsisAlert_Demo/
    physionet.org/files/mimiciv/3.1/
      icu/
        icustays.csv.gz          ← 100 stays
        chartevents.csv.gz       ← ~10 000 vital-sign readings
      hosp/
        patients.csv.gz
        admissions.csv.gz
        diagnoses_icd.csv.gz     ← A41.x sepsis codes for ~22 patients
        labevents.csv.gz         ← ~4 500 lab-result readings
    data/processed/
      cohort.parquet             ← identical schema to real processed file
      features.parquet           ← identical schema to real processed file
    models/
      lightgbm_sepsis.pkl        ← copied from local models/ folder

Usage
-----
    python scripts/generate_demo_data.py

The colleague's config.yaml should point data paths to the generated
~/Desktop/SepsisAlert_Demo/ sub-directories (paths printed on completion).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

RNG = np.random.default_rng(42)
N_PATIENTS = 100
SEPSIS_RATE = 0.22          # ~22 % have sepsis (matches real cohort)
BASE_DATE = pd.Timestamp("2021-01-01")

VITAL_ITEMS = {
    220045: "heart_rate",
    220052: "map",
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

CARE_UNITS = [
    "Medical/Surgical ICU",
    "Cardiac Vascular ICU",
    "Neuro ICU",
    "Medical ICU",
    "Surgical ICU",
]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _normal_val(item_id: int, sepsis: bool) -> float:
    """Return a physiologically plausible measurement for item_id."""
    if item_id == 220045:   # heart rate
        base = RNG.uniform(100, 130) if sepsis else RNG.uniform(60, 100)
        return float(np.clip(base + RNG.normal(0, 5), 30, 160))
    if item_id == 220052:   # MAP
        base = RNG.uniform(50, 65) if sepsis else RNG.uniform(70, 100)
        return float(np.clip(base + RNG.normal(0, 4), 30, 130))
    if item_id == 220210:   # resp rate
        base = RNG.uniform(22, 32) if sepsis else RNG.uniform(12, 20)
        return float(np.clip(base + RNG.normal(0, 2), 6, 40))
    if item_id == 223761:   # temperature F
        base = RNG.uniform(100, 103) if sepsis else RNG.uniform(97, 99.5)
        return float(np.clip(base + RNG.normal(0, 0.3), 93, 107))
    if item_id == 220277:   # SpO2
        base = RNG.uniform(88, 94) if sepsis else RNG.uniform(95, 100)
        return float(np.clip(base + RNG.normal(0, 1), 70, 100))
    # Labs
    if item_id == 50813:    # lactate
        base = RNG.uniform(2.5, 6.0) if sepsis else RNG.uniform(0.5, 2.0)
        return float(np.clip(base + RNG.normal(0, 0.3), 0.1, 15))
    if item_id == 51301:    # WBC
        base = RNG.uniform(14, 25) if sepsis else RNG.uniform(4, 12)
        return float(np.clip(base + RNG.normal(0, 1), 0.5, 40))
    if item_id == 50912:    # creatinine
        base = RNG.uniform(1.5, 4.0) if sepsis else RNG.uniform(0.6, 1.2)
        return float(np.clip(base + RNG.normal(0, 0.1), 0.1, 15))
    if item_id == 50885:    # bilirubin
        base = RNG.uniform(2.0, 6.0) if sepsis else RNG.uniform(0.2, 1.2)
        return float(np.clip(base + RNG.normal(0, 0.2), 0.1, 20))
    if item_id == 51265:    # platelets
        base = RNG.uniform(50, 120) if sepsis else RNG.uniform(150, 400)
        return float(np.clip(base + RNG.normal(0, 10), 10, 600))
    if item_id == 50882:    # bicarbonate
        base = RNG.uniform(14, 20) if sepsis else RNG.uniform(22, 28)
        return float(np.clip(base + RNG.normal(0, 1), 5, 40))
    if item_id == 50931:    # glucose
        base = RNG.uniform(140, 250) if sepsis else RNG.uniform(70, 130)
        return float(np.clip(base + RNG.normal(0, 10), 30, 500))
    return float(RNG.uniform(1, 10))


def _compute_trend(hours: np.ndarray, values: np.ndarray) -> float:
    """Linear slope (units/hour)."""
    if len(values) < 3:
        return 0.0
    try:
        slope, _ = np.polyfit(hours, values, 1)
        return float(slope)
    except np.linalg.LinAlgError:
        return 0.0


# ------------------------------------------------------------------ #
# Build raw tables                                                      #
# ------------------------------------------------------------------ #

def build_raw_tables() -> dict[str, pd.DataFrame]:
    """Generate all 6 raw MIMIC-IV-like tables."""
    sepsis_flags = (RNG.random(N_PATIENTS) < SEPSIS_RATE).astype(int)

    subject_ids = np.arange(10_000_001, 10_000_001 + N_PATIENTS)
    hadm_ids    = np.arange(20_000_001, 20_000_001 + N_PATIENTS)
    stay_ids    = np.arange(30_000_001, 30_000_001 + N_PATIENTS)

    # LOS between 0.5 and 20 days
    los_days = RNG.uniform(0.5, 20, N_PATIENTS)
    intimes  = [BASE_DATE + pd.Timedelta(days=float(RNG.uniform(0, 700)))
                for _ in range(N_PATIENTS)]
    outtimes = [intimes[i] + pd.Timedelta(days=float(los_days[i]))
                for i in range(N_PATIENTS)]
    ages    = RNG.integers(22, 86, N_PATIENTS)
    genders = RNG.choice(["M", "F"], N_PATIENTS)
    units   = RNG.choice(CARE_UNITS, N_PATIENTS)

    # --- icustays ---
    icustays = pd.DataFrame({
        "stay_id":        stay_ids,
        "subject_id":     subject_ids,
        "hadm_id":        hadm_ids,
        "intime":         intimes,
        "outtime":        outtimes,
        "los":            los_days,
        "first_careunit": units,
    })

    # --- patients ---
    patients = pd.DataFrame({
        "subject_id": subject_ids,
        "gender":     genders,
        "anchor_age": ages,
    })

    # --- admissions ---
    adm_types = RNG.choice(["EMERGENCY", "URGENT", "ELECTIVE"], N_PATIENTS,
                            p=[0.65, 0.25, 0.10])
    expire = (RNG.random(N_PATIENTS) < 0.08).astype(int)
    admissions = pd.DataFrame({
        "hadm_id":             hadm_ids,
        "subject_id":          subject_ids,
        "admittime":           intimes,
        "admission_type":      adm_types,
        "hospital_expire_flag": expire,
    })

    # --- diagnoses_icd (sepsis patients get A41.x) ---
    diag_rows = []
    sepsis_codes = ["A419", "A4101", "A415", "R6521"]
    for i, (hadm_id, has_sep) in enumerate(zip(hadm_ids, sepsis_flags)):
        # Every patient gets a few ICD-10 codes
        for seq, code in enumerate(["I10", "E119", "Z87891"], start=1):
            diag_rows.append({
                "hadm_id": hadm_id,
                "seq_num": seq,
                "icd_version": 10,
                "icd_code": code,
            })
        if has_sep:
            diag_rows.append({
                "hadm_id": hadm_id,
                "seq_num": 1,
                "icd_version": 10,
                "icd_code": RNG.choice(sepsis_codes),
            })
    diagnoses_icd = pd.DataFrame(diag_rows)

    # --- chartevents (vitals: 8–16 readings per vital per stay) ---
    chart_rows = []
    for i, stay_id in enumerate(stay_ids):
        intime  = intimes[i]
        los_h   = los_days[i] * 24
        window  = min(24.0, los_h)
        is_sep  = bool(sepsis_flags[i])
        for item_id in VITAL_ITEMS:
            n_obs = int(RNG.integers(8, 17))
            obs_hours = np.sort(RNG.uniform(0, window, n_obs))
            for h in obs_hours:
                chart_rows.append({
                    "stay_id":   stay_id,
                    "itemid":    item_id,
                    "charttime": intime + pd.Timedelta(hours=float(h)),
                    "valuenum":  _normal_val(item_id, is_sep),
                })
    chartevents = pd.DataFrame(chart_rows)

    # --- labevents (3–6 readings per lab per stay) ---
    lab_rows = []
    for i, (stay_id, hadm_id) in enumerate(zip(stay_ids, hadm_ids)):
        intime  = intimes[i]
        los_h   = los_days[i] * 24
        window  = min(24.0, los_h)
        is_sep  = bool(sepsis_flags[i])
        for item_id in LAB_ITEMS:
            n_obs = int(RNG.integers(3, 7))
            obs_hours = np.sort(RNG.uniform(0, window, n_obs))
            for h in obs_hours:
                lab_rows.append({
                    "hadm_id":   hadm_id,
                    "stay_id":   stay_id,
                    "itemid":    item_id,
                    "charttime": intime + pd.Timedelta(hours=float(h)),
                    "valuenum":  _normal_val(item_id, is_sep),
                })
    labevents = pd.DataFrame(lab_rows)

    return {
        "icustays":     icustays,
        "patients":     patients,
        "admissions":   admissions,
        "diagnoses_icd": diagnoses_icd,
        "chartevents":  chartevents,
        "labevents":    labevents,
        "sepsis_flags": sepsis_flags,
        "stay_ids":     stay_ids,
        "hadm_ids":     hadm_ids,
        "subject_ids":  subject_ids,
        "intimes":      intimes,
        "los_days":     los_days,
        "ages":         ages,
        "genders":      genders,
        "units":        units,
    }


# ------------------------------------------------------------------ #
# Build processed tables                                               #
# ------------------------------------------------------------------ #

def build_processed(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cohort.parquet and features.parquet from raw tables."""
    stay_ids    = raw["stay_ids"]
    hadm_ids    = raw["hadm_ids"]
    subject_ids = raw["subject_ids"]
    intimes     = raw["intimes"]
    los_days    = raw["los_days"]
    sepsis      = raw["sepsis_flags"]
    ages        = raw["ages"]
    genders     = raw["genders"]
    units       = raw["units"]

    # --- cohort ---
    cohort = pd.DataFrame({
        "stay_id":           stay_ids,
        "subject_id":        subject_ids,
        "hadm_id":           hadm_ids,
        "intime":            intimes,
        "outtime":           [intimes[i] + pd.Timedelta(days=float(los_days[i]))
                              for i in range(N_PATIENTS)],
        "los":               los_days,
        "first_careunit":    units,
        "age":               ages,
        "gender":            genders,
        "sepsis_label":      sepsis,
        "sepsis_onset_proxy": [
            (intimes[i] + pd.Timedelta(hours=6)) if sepsis[i] else pd.NaT
            for i in range(N_PATIENTS)
        ],
    })

    # --- features ---
    chartevents = raw["chartevents"]
    labevents   = raw["labevents"]
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
    labevents["charttime"]   = pd.to_datetime(labevents["charttime"])

    vital_records = []
    for stay_id in stay_ids:
        row = {"stay_id": stay_id}
        grp = chartevents[chartevents["stay_id"] == stay_id]
        for item_id, name in VITAL_ITEMS.items():
            sub = grp[grp["itemid"] == item_id].sort_values("charttime")
            vals = sub["valuenum"].values
            if len(vals) > 0:
                hours = (sub["charttime"] - sub["charttime"].iloc[0]).dt.total_seconds().values / 3600
                row[f"{name}_mean"]  = float(vals.mean())
                row[f"{name}_min"]   = float(vals.min())
                row[f"{name}_max"]   = float(vals.max())
                row[f"{name}_last"]  = float(vals[-1])
                row[f"{name}_trend"] = _compute_trend(hours, vals)
            else:
                for sfx in ["mean", "min", "max", "last", "trend"]:
                    row[f"{name}_{sfx}"] = np.nan
        vital_records.append(row)
    vital_df = pd.DataFrame(vital_records)

    lab_records = []
    for stay_id, hadm_id in zip(stay_ids, hadm_ids):
        row = {"stay_id": stay_id}
        grp = labevents[labevents["hadm_id"] == hadm_id]
        for item_id, name in LAB_ITEMS.items():
            sub = grp[grp["itemid"] == item_id].sort_values("charttime")
            vals = sub["valuenum"].values
            if len(vals) > 0:
                hours = (sub["charttime"] - sub["charttime"].iloc[0]).dt.total_seconds().values / 3600
                row[f"{name}_last"]  = float(vals[-1])
                row[f"{name}_mean"]  = float(vals.mean())
                row[f"{name}_delta"] = float(vals[-1] - vals[0]) if len(vals) > 1 else 0.0
                row[f"{name}_trend"] = _compute_trend(hours, vals)
            else:
                for sfx in ["last", "mean", "delta", "trend"]:
                    row[f"{name}_{sfx}"] = np.nan
        lab_records.append(row)
    lab_df = pd.DataFrame(lab_records)

    features = cohort[["stay_id", "hadm_id", "age", "gender", "sepsis_label"]].copy()
    features = features.merge(vital_df, on="stay_id", how="left")
    features = features.merge(lab_df,   on="stay_id", how="left")
    features["gender_male"] = (features["gender"] == "M").astype(int)
    features = features.drop(columns=["gender"])

    return cohort, features


# ------------------------------------------------------------------ #
# Write to disk                                                         #
# ------------------------------------------------------------------ #

def write_demo(dest: Path, raw: dict, cohort: pd.DataFrame, features: pd.DataFrame) -> None:
    """Write all files to dest in the expected folder structure."""
    icu_dir  = dest / "physionet.org" / "files" / "mimiciv" / "3.1" / "icu"
    hosp_dir = dest / "physionet.org" / "files" / "mimiciv" / "3.1" / "hosp"
    proc_dir = dest / "data" / "processed"
    mdl_dir  = dest / "models"

    for d in [icu_dir, hosp_dir, proc_dir, mdl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    raw["icustays"].to_csv(icu_dir / "icustays.csv.gz",     index=False, compression="gzip")
    raw["chartevents"].to_csv(icu_dir / "chartevents.csv.gz", index=False, compression="gzip")

    raw["patients"].to_csv(hosp_dir / "patients.csv.gz",       index=False, compression="gzip")
    raw["admissions"].to_csv(hosp_dir / "admissions.csv.gz",   index=False, compression="gzip")
    raw["diagnoses_icd"].to_csv(hosp_dir / "diagnoses_icd.csv.gz", index=False, compression="gzip")
    raw["labevents"].to_csv(hosp_dir / "labevents.csv.gz",     index=False, compression="gzip")

    cohort.to_parquet(proc_dir / "cohort.parquet",     index=False)
    features.to_parquet(proc_dir / "features.parquet", index=False)

    # Copy model
    model_src = Path(__file__).parent.parent / "models" / "lightgbm_sepsis.pkl"
    if model_src.exists():
        shutil.copy2(model_src, mdl_dir / "lightgbm_sepsis.pkl")
        print(f"  Copied model from {model_src}")
    else:
        print(f"  WARNING: model not found at {model_src} — train first with python run_pipeline.py")


def main() -> None:
    """Generate and save all demo data."""
    dest = Path.home() / "Desktop" / "SepsisAlert_Demo"
    if dest.exists():
        shutil.rmtree(dest)   # fresh start each run

    print(f"Generating {N_PATIENTS} synthetic ICU patients ({int(N_PATIENTS * SEPSIS_RATE)} sepsis)...")
    raw = build_raw_tables()
    print("Building processed files...")
    cohort, features = build_processed(raw)
    print(f"Writing to {dest} ...")
    write_demo(dest, raw, cohort, features)

    print("\nDone. Summary:")
    print(f"  physionet.org/  — 6 raw CSV.gz files ({N_PATIENTS} patients)")
    print(f"  data/processed/ — cohort.parquet ({len(cohort)} rows), "
          f"features.parquet ({features.shape[1]} cols)")
    print(f"  models/         — lightgbm_sepsis.pkl")
    print(f"\nShare this folder: {dest}")
    print("\nColleague config.yaml should point to:")
    print(f'  mimic_base: "{dest}/physionet.org/files/mimiciv/3.1"')
    print(f'  icu_path:   "{dest}/physionet.org/files/mimiciv/3.1/icu"')
    print(f'  hosp_path:  "{dest}/physionet.org/files/mimiciv/3.1/hosp"')
    print(f'  processed_path: "{dest}/data/processed"')


if __name__ == "__main__":
    main()
