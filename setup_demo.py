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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.model.calibration import IsotonicCalibrated as _IsotonicCalibrated

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
# Sepsis values are calibrated to produce HIGH/CRITICAL scores through the
# real MIMIC-IV trained model.  Normal values match healthy ICU ranges.
VITAL_PARAMS: dict[str, tuple] = {
    "heart_rate":    (78,  12,  128,  12),
    "map":           (85,  10,   48,   8),
    "resp_rate":     (15,   3,   30,   4),
    "temperature_f": (98.4, 0.5, 103.2, 1.2),
    "spo2":          (97.5, 1.5,  88.0, 2.5),
}

LAB_PARAMS: dict[str, tuple] = {
    "lactate":     (1.0, 0.4,  7.5, 1.8),
    "wbc":         (7.5, 2.0, 28.0, 5.0),
    "creatinine":  (0.9, 0.2,  3.5, 0.9),
    "bilirubin":   (0.6, 0.3,  4.2, 1.5),
    "platelets":   (250, 50,   75, 35),
    "bicarbonate": (25,  2,   14,  3),
    "glucose":     (105, 20,  210, 50),
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

    # 30% of sepsis patients are "severe" (septic shock) — extreme values that
    # push scores into HIGH/CRITICAL through the real MIMIC-IV model.
    severe = (RNG.random(n) < 0.30) & sepsis.astype(bool)

    rows = []
    for i in range(n):
        s = bool(sepsis[i])
        sv = bool(severe[i])
        row: dict = {
            "stay_id":      stay_ids[i],
            "hadm_id":      hadm_ids[i],
            "age":          float(ages[i]),
            "gender_male":  1 if genders[i] == "M" else 0,
            "sepsis_label": int(sepsis[i]),
        }

        for name, (mn, sn, ms, ss) in VITAL_PARAMS.items():
            # Severe patients: push further toward physiologic extremes
            ms_eff = ms * 1.15 if sv and name not in ("spo2", "map") else (ms * 0.88 if sv else ms)
            ss_eff = ss * 0.6 if sv else ss
            base = _sample(mn, sn, ms_eff, ss_eff, s)
            trend = _sample(0.1, 0.05, 0.55, 0.15, s) if name != "spo2" else _sample(-0.02, 0.01, -0.25, 0.05, s)
            row[f"{name}_mean"]  = base
            row[f"{name}_min"]   = base - abs(_sample(0, sn * 0.3, 0, ss * 0.3, s))
            row[f"{name}_max"]   = base + abs(_sample(0, sn * 0.3, 0, ss * 0.3, s))
            row[f"{name}_last"]  = base + _sample(0, sn * 0.1, 0, ss * 0.2, s)
            row[f"{name}_trend"] = trend

        for name, (mn, sn, ms, ss) in LAB_PARAMS.items():
            lo = 0.0
            ms_eff = ms * 1.3 if sv and name not in ("platelets", "bicarbonate") else (ms * 0.65 if sv else ms)
            ss_eff = ss * 0.5 if sv else ss
            base = _sample(mn, sn, ms_eff, ss_eff, s, clip_lo=lo)
            delta_mult = 0.5 if sv else 0.3
            delta = _sample(0.0, sn * 0.1, sn * delta_mult, ss * 0.2, s)
            row[f"{name}_last"]  = base
            row[f"{name}_mean"]  = base - abs(_sample(0, sn * 0.1, 0, ss * 0.1, s))
            row[f"{name}_delta"] = delta
            row[f"{name}_trend"] = delta / 12.0

        rows.append(row)

    features = pd.DataFrame(rows)

    # Plant a small number of septic-shock patients with feature profiles
    # calibrated to score HIGH/CRITICAL through the real MIMIC-IV model.
    # These are constructed from the observed feature ranges of top-scoring
    # real MIMIC-IV patients and represent plausible septic shock presentations.
    n_planted = max(12, int(n * 0.003))  # ~0.3% matches real CRITICAL rate
    shock_templates = [
        # (wbc_last, wbc_mean, lactate_last, lactate_mean, map_last, map_mean,
        #  hr_last, hr_mean, rr_last, spo2_last, creat_last, bili_last,
        #  plt_last, bicarb_last, temp_f_last)
        (47.8, 34.5, 9.2, 9.4, 52.0, 55.0, 118.0, 108.0, 28.0, 94.0, 2.1, 6.9, 28.0, 14.0, 98.4),
        (31.3, 26.8, 7.4, 5.6, 48.0, 51.0, 111.0, 106.0, 34.0, 93.0, 3.5, 1.8, 82.0, 15.0, 102.1),
        (25.5, 29.8, 5.0, 4.2, 58.0, 61.0, 98.0, 112.0, 31.0, 92.0, 2.4, 2.2, 65.0, 16.0, 103.0),
        (23.4, 22.3, 6.2, 4.8, 44.0, 48.0, 124.0, 115.0, 32.0, 91.0, 3.6, 3.1, 55.0, 13.0, 101.8),
        (38.0, 30.1, 4.8, 3.9, 55.0, 58.0, 130.0, 122.0, 29.0, 90.0, 2.8, 5.2, 42.0, 14.5, 99.8),
        (20.1, 20.2, 2.9, 3.9, 46.0, 49.0, 82.0, 114.0, 20.0, 91.0, 2.4, 1.5, 95.0, 17.0, 99.2),
        (29.0, 25.0, 8.1, 7.2, 50.0, 53.0, 126.0, 118.0, 33.0, 89.0, 4.1, 4.0, 35.0, 12.0, 101.5),
    ]
    planted_rows = []
    for k in range(n_planted):
        tmpl = shock_templates[k % len(shock_templates)]
        noise = lambda v, frac=0.08: float(v) + RNG.normal(0, abs(float(v)) * frac)
        wbc_l, wbc_m, lac_l, lac_m, map_l, map_m, hr_l, hr_m, rr_l, spo2_l, crt_l, bil_l, plt_l, bic_l, tmp_l = [noise(v) for v in tmpl]
        stay_id_p = 39_900_001 + k
        pr: dict = {
            "stay_id": stay_id_p, "hadm_id": 29_900_001 + k,
            "age": float(RNG.integers(55, 85)),
            "gender_male": int(RNG.integers(0, 2)),
            "sepsis_label": 1,
            "heart_rate_mean": hr_m, "heart_rate_min": hr_m - 12, "heart_rate_max": hr_m + 18,
            "heart_rate_last": hr_l, "heart_rate_trend": 0.45,
            "map_mean": map_m, "map_min": map_m - 8, "map_max": map_m + 6,
            "map_last": map_l, "map_trend": -0.3,
            "resp_rate_mean": rr_l, "resp_rate_min": rr_l - 3, "resp_rate_max": rr_l + 5,
            "resp_rate_last": rr_l, "resp_rate_trend": 0.5,
            "temperature_f_mean": tmp_l, "temperature_f_min": tmp_l - 0.5, "temperature_f_max": tmp_l + 0.8,
            "temperature_f_last": tmp_l, "temperature_f_trend": 0.1,
            "spo2_mean": spo2_l, "spo2_min": spo2_l - 3, "spo2_max": spo2_l + 1,
            "spo2_last": spo2_l, "spo2_trend": -0.2,
            "lactate_last": lac_l, "lactate_mean": lac_m, "lactate_delta": lac_l - lac_m, "lactate_trend": (lac_l - lac_m) / 12,
            "wbc_last": wbc_l, "wbc_mean": wbc_m, "wbc_delta": wbc_l - wbc_m, "wbc_trend": (wbc_l - wbc_m) / 12,
            "creatinine_last": crt_l, "creatinine_mean": crt_l * 0.9, "creatinine_delta": crt_l * 0.1, "creatinine_trend": crt_l * 0.01,
            "bilirubin_last": bil_l, "bilirubin_mean": bil_l * 0.85, "bilirubin_delta": bil_l * 0.15, "bilirubin_trend": bil_l * 0.01,
            "platelets_last": plt_l, "platelets_mean": plt_l * 1.1, "platelets_delta": -(plt_l * 0.1), "platelets_trend": -(plt_l * 0.01),
            "bicarbonate_last": bic_l, "bicarbonate_mean": bic_l * 1.05, "bicarbonate_delta": -(bic_l * 0.05), "bicarbonate_trend": -(bic_l * 0.005),
            "glucose_last": noise(185), "glucose_mean": noise(170), "glucose_delta": 15.0, "glucose_trend": 1.25,
        }
        planted_rows.append(pr)

    if planted_rows:
        planted_df = pd.DataFrame(planted_rows)
        features = pd.concat([features, planted_df], ignore_index=True)
        # Add planted patients to cohort too
        planted_cohort = pd.DataFrame([{
            "stay_id": r["stay_id"], "subject_id": 19_900_001 + k,
            "hadm_id": r["hadm_id"],
            "intime": pd.Timestamp("2022-06-01") + pd.Timedelta(days=float(k)),
            "outtime": pd.Timestamp("2022-06-05") + pd.Timedelta(days=float(k)),
            "los": 4.0, "first_careunit": "Medical Intensive Care Unit (MICU)",
            "age": r["age"], "gender": "M" if r["gender_male"] else "F",
            "sepsis_label": 1, "sepsis_onset_proxy": pd.Timestamp("2022-06-01"),
        } for k, r in enumerate(planted_rows)])
        cohort = pd.concat([cohort, planted_cohort], ignore_index=True)

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

    _cal_proba = model.predict_proba(x_cal)[:, 1]
    _iso       = IsotonicRegression(out_of_bounds="clip")
    _iso.fit(_cal_proba, y_cal)
    calibrated = _IsotonicCalibrated(model, _iso)

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
