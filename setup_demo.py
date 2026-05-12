"""
SepsisAlert — One-command demo setup.

LEGAL / DATA PRIVACY NOTICE
============================
This script generates FULLY SYNTHETIC, artificially created data.
No real patient records, no data derived from MIMIC-IV or any other
clinical database, and no personally identifiable information (PII)
are produced, stored, or distributed by this script.

All values (ages, vital signs, lab results, ICD codes, timestamps) are
sampled from statistical distributions chosen to be physiologically
plausible for demonstration purposes only. They do not represent any
real individual, hospital visit, or clinical outcome.

This output is therefore NOT subject to:
  - HIPAA (US Health Insurance Portability and Accountability Act)
  - GDPR (EU General Data Protection Regulation)
  - PhysioNet / MIMIC-IV data use agreements
  - Any other patient data protection regulation

PURPOSE
=======
Generates 5000 synthetic ICU patients and trains a demo model so the
React + FastAPI dashboard works immediately without MIMIC-IV credentials.

Usage:
    python setup_demo.py
    uvicorn src.api.main:app --reload --port 8000
    cd frontend && npm install && npm run dev

What this creates (all gitignored — no real patient data):
    data/processed/cohort.parquet    ← 5000 synthetic ICU stays
    data/processed/features.parquet  ← feature matrix (same schema as real)
    models/sepsis_model.pkl       ← demo model trained on synthetic data
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

RNG = np.random.default_rng(42)
N = 5000          # number of synthetic patients
SEPSIS_RATE = 0.22

CARE_UNITS = [
    "Medical/Surgical ICU", "Cardiac Vascular ICU",
    "Neuro ICU", "Medical ICU", "Surgical ICU",
]

# Vitals: (mean_normal, std_normal, mean_sepsis, std_sepsis)
VITAL_PARAMS: dict[str, tuple] = {
    "heart_rate":    (78,  12,  118,  15),
    "map":           (85,  10,   57,  10),
    "resp_rate":     (15,   3,   26,   4),
    "temperature_f": (98.4, 0.5, 101.8, 1.0),
    "spo2":          (97.5, 1.5,  91.0, 2.5),
}

LAB_PARAMS: dict[str, tuple] = {
    "lactate":     (1.0, 0.4,  4.8, 1.5),
    "wbc":         (7.5, 2.0, 17.0, 4.0),
    "creatinine":  (0.9, 0.2,  2.2, 0.8),
    "bilirubin":   (0.6, 0.3,  2.8, 1.2),
    "platelets":   (250, 50,  105, 40),
    "bicarbonate": (25,  2,   18,  3),
    "glucose":     (105, 20,  175, 40),
}


def _sample(mean_n, std_n, mean_s, std_s, sepsis: bool, clip_lo=None, clip_hi=None) -> float:
    """Draw a single value from normal or sepsis distribution."""
    val = RNG.normal(mean_s if sepsis else mean_n, std_s if sepsis else std_n)
    if clip_lo is not None:
        val = max(val, clip_lo)
    if clip_hi is not None:
        val = min(val, clip_hi)
    return float(val)


def build_features(n: int = N) -> tuple[pd.DataFrame, pd.DataFrame]:  # pylint: disable=too-many-locals
    """Build synthetic cohort.parquet and features.parquet."""
    sepsis = (RNG.random(n) < SEPSIS_RATE).astype(int)
    ages   = RNG.integers(22, 86, n)
    genders = RNG.choice(["M", "F"], n)
    units  = RNG.choice(CARE_UNITS, n)
    stay_ids = np.arange(30_000_001, 30_000_001 + n)
    hadm_ids = np.arange(20_000_001, 20_000_001 + n)
    intimes  = [pd.Timestamp("2021-01-01") + pd.Timedelta(days=float(d))
                for d in RNG.uniform(0, 700, n)]
    los_days = RNG.uniform(0.5, 20, n)

    cohort = pd.DataFrame({
        "stay_id":        stay_ids,
        "subject_id":     np.arange(10_000_001, 10_000_001 + n),
        "hadm_id":        hadm_ids,
        "intime":         intimes,
        "outtime":        [intimes[i] + pd.Timedelta(days=float(los_days[i])) for i in range(n)],
        "los":            los_days,
        "first_careunit": units,
        "age":            ages,
        "gender":         genders,
        "sepsis_label":   sepsis,
        "sepsis_onset_proxy": [
            intimes[i] + pd.Timedelta(hours=6) if sepsis[i] else pd.NaT
            for i in range(n)
        ],
    })

    rows = []
    for i in range(n):
        s = bool(sepsis[i])
        row: dict = {
            "stay_id":      stay_ids[i],
            "hadm_id":      hadm_ids[i],
            "age":          float(ages[i]),
            "gender_male":  1 if genders[i] == "M" else 0,
            "sepsis_label": int(sepsis[i]),
        }

        for name, (mn, sn, ms, ss) in VITAL_PARAMS.items():
            base = _sample(mn, sn, ms, ss, s)
            trend = _sample(0.1, 0.05, 0.4, 0.15, s) if name != "spo2" else _sample(-0.02, 0.01, -0.15, 0.05, s)
            row[f"{name}_mean"]  = base
            row[f"{name}_min"]   = base - abs(_sample(0, sn * 0.3, 0, ss * 0.3, s))
            row[f"{name}_max"]   = base + abs(_sample(0, sn * 0.3, 0, ss * 0.3, s))
            row[f"{name}_last"]  = base + _sample(0, sn * 0.1, 0, ss * 0.2, s)
            row[f"{name}_trend"] = trend

        for name, (mn, sn, ms, ss) in LAB_PARAMS.items():
            lo = 0.0
            base = _sample(mn, sn, ms, ss, s, clip_lo=lo)
            delta = _sample(0.0, sn * 0.1, sn * 0.3, ss * 0.2, s)
            row[f"{name}_last"]  = base
            row[f"{name}_mean"]  = base - abs(_sample(0, sn * 0.1, 0, ss * 0.1, s))
            row[f"{name}_delta"] = delta
            row[f"{name}_trend"] = delta / 12.0

        rows.append(row)

    features = pd.DataFrame(rows)
    return cohort, features


def train_demo_model(features: pd.DataFrame) -> dict:
    """Train a quick HistGradientBoosting on the synthetic feature matrix."""
    meta_cols = {"stay_id", "hadm_id", "sepsis_label"}
    feature_cols = [c for c in features.columns if c not in meta_cols]

    X = features[feature_cols]
    y = features["sepsis_label"]

    # 3-way split: 72% train / 8% calibration / 20% test — matches train.py
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
    )

    model = HistGradientBoostingClassifier(
        max_leaf_nodes=31,
        learning_rate=0.05,
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)

    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(x_cal, y_cal)

    proba = calibrated.predict_proba(x_test)[:, 1]
    auroc = float(roc_auc_score(y_test, proba))
    print(f"  Demo model test AUROC (calibrated): {auroc:.3f}")

    training_stats = {
        col: {"mean": float(x_train[col].mean()), "std": float(x_train[col].std())}
        for col in feature_cols
    }
    feat_matrix = x_train.fillna(x_train.mean()).values.astype(float)
    training_mean = feat_matrix.mean(axis=0)
    try:
        cov = np.cov(feat_matrix, rowvar=False)
        training_cov_inv = np.linalg.pinv(cov)
    except Exception:  # pylint: disable=broad-except
        training_cov_inv = None

    return {
        "model":            calibrated,
        "base_model":       model,
        "feature_cols":     feature_cols,
        "auroc":            auroc,
        "training_stats":   training_stats,
        "training_mean":    training_mean,
        "training_cov_inv": training_cov_inv,
    }


def write_local(cohort: pd.DataFrame, features: pd.DataFrame, artifact: dict) -> None:
    """Write all files to local data/ and models/ for the dashboard."""
    proc_dir = Path("data/processed")
    mdl_dir  = Path("models")
    proc_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)

    cohort.to_parquet(proc_dir / "cohort.parquet",     index=False)
    features.to_parquet(proc_dir / "features.parquet", index=False)
    joblib.dump(artifact, mdl_dir / "sepsis_model.pkl")

    print(f"  cohort.parquet  → {proc_dir / 'cohort.parquet'}")
    print(f"  features.parquet→ {proc_dir / 'features.parquet'}")
    print(f"  model artifact  → {mdl_dir / 'sepsis_model.pkl'}")


def main() -> None:
    """Run the full demo setup."""
    print("=" * 55)
    print("  SepsisAlert — Demo Setup")
    print("=" * 55)

    features_path = Path("data/processed/features.parquet")
    cohort_path   = Path("data/processed/cohort.parquet")
    model_path    = Path("models/sepsis_model.pkl")

    # Skip synthetic data generation if real processed data already exists.
    # This preserves MIMIC-IV processed data when running setup_demo.py in a
    # hospital deployment that has already run the real data pipeline.
    data_exists = features_path.exists() and cohort_path.exists()
    if data_exists:
        print("\n  Processed data found at data/processed/ — skipping synthetic generation.")
        print("  (delete data/processed/*.parquet to regenerate with synthetic data)")
        features = pd.read_parquet(features_path)
        cohort   = pd.read_parquet(cohort_path)
    else:
        print(f"\n  Generating {N} synthetic ICU patients "
              f"({int(N * SEPSIS_RATE)} sepsis)...")
        cohort, features = build_features()

    if model_path.exists():
        print(f"  Model found at {model_path} — skipping demo training.")
        artifact = joblib.load(model_path)
    else:
        print("  No trained model found — training demo model on loaded data...")
        artifact = train_demo_model(features)

    if not data_exists:
        print("  Writing to local data/ and models/...")
        write_local(cohort, features, artifact)
    elif not model_path.exists():
        # Data exists but model is missing — only write the model
        mdl_dir = Path("models")
        mdl_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, mdl_dir / "sepsis_model.pkl")
        print(f"  model artifact  → {mdl_dir / 'sepsis_model.pkl'}")

    print("\n  Done. Launch the dashboard with:")
    print("    uvicorn src.api.main:app --reload --port 8000")
    print("    cd frontend && npm install && npm run dev")
    print("    Open http://localhost:5173\n")


if __name__ == "__main__":
    main()
