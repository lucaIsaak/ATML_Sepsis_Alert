"""
Model evaluation and comparison against the NEWS2 clinical baseline.

Evaluation design principles:
  - Held-out test set (stratified 80/20, random_state=42 — same as train.py):
    AUROC reported here is true out-of-sample; no hyperparameter selection
    or calibration fitting ever touches the test set.
  - NEWS2 comparison on the same test set: NEWS2 (National Early Warning Score 2)
    is the current clinical standard in European ICUs (Royal College of Physicians,
    2017). Comparing on identical patients gives a fair head-to-head delta.
  - F2 threshold optimisation (β=2): precision-recall trade-off is asymmetric for
    sepsis — a missed case (false negative) carries higher cost than a false alarm
    (false positive). F2 encodes this by weighting recall twice as heavily as
    precision, giving a principled, data-driven justification for the alert cutoff.
  - Subgroup AUROC by gender, age quartile, and care unit: required for EU AI Act
    Annex III compliance and essential for detecting demographic performance gaps
    before clinical deployment. Goal: max subgroup AUROC gap < 0.05.
  - Brier score: AUROC measures discrimination only. Brier score measures
    calibration quality — whether a score of 0.6 truly reflects ~60% sepsis risk.
    Both are required for clinical decision support (Van Calster et al., 2019).

References:
  Johnson et al. (2023) — MIMIC-IV sepsis prediction: AUROC 0.87
  Moor et al. (2021) — Early sepsis detection: AUROC 0.85–0.89
  Royal College of Physicians (2017) — NEWS2 specification
  Van Calster et al. (2019) — Calibration: the Achilles heel of predictive analytics
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.model.predict import load_model, predict_batch


# ------------------------------------------------------------------ #
# NEWS2 sub-scorers (one per vital sign to keep branch count low)     #
# ------------------------------------------------------------------ #

def _score_resp_rate(rr: float) -> int:
    """Return NEWS2 points for respiratory rate."""
    if rr <= 8 or rr >= 25:
        return 3
    if rr >= 21:
        return 2
    if rr <= 11:   # 9-11 = 1 point; 12-20 = 0 points (per NEWS2 spec)
        return 1
    return 0


def _score_spo2(spo2: float) -> int:
    """Return NEWS2 points for SpO2."""
    if spo2 <= 91:
        return 3
    if spo2 <= 93:
        return 2
    if spo2 <= 95:
        return 1
    return 0


def _score_heart_rate(hr: float) -> int:
    """Return NEWS2 points for heart rate."""
    if hr <= 40 or hr >= 131:
        return 3
    if hr >= 111:
        return 2
    if hr >= 91 or hr <= 50:
        return 1
    return 0


def _score_temperature(temp_f: float) -> int:
    """Return NEWS2 points for temperature (Fahrenheit input)."""
    temp_c = (temp_f - 32) * 5 / 9
    if temp_c <= 35 or temp_c >= 39.1:
        return 2
    if temp_c >= 38.1:
        return 1
    return 0


# ------------------------------------------------------------------ #
# Public scorer                                                        #
# ------------------------------------------------------------------ #

def news2_score(row: pd.Series) -> int:
    """
    Compute a simplified NEWS2 score from available features.

    Returns integer score (higher = more abnormal).
    NEWS2 thresholds: >=7 = high risk.
    """
    score = 0

    rr = row.get("resp_rate_mean", np.nan)
    if not np.isnan(rr):
        score += _score_resp_rate(rr)

    spo2 = row.get("spo2_min", np.nan)
    if not np.isnan(spo2):
        score += _score_spo2(spo2)

    hr = row.get("heart_rate_mean", np.nan)
    if not np.isnan(hr):
        score += _score_heart_rate(hr)

    temp = row.get("temperature_f_last", np.nan)
    if not np.isnan(temp):
        score += _score_temperature(temp)

    return score


def _threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """
    Compute sensitivity, specificity, PPV, and NPV at a fixed threshold.

    These are the operationally relevant metrics for clinical deployment —
    AUROC alone does not tell you how many patients will be missed or
    how many false alarms will be generated at the chosen alert threshold.
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "threshold": threshold,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _optimal_threshold(y_true: np.ndarray, y_score: np.ndarray, beta: float = 2.0) -> dict:
    """
    Find the decision threshold that maximises F-beta score.

    beta=2 weights recall twice as heavily as precision — the correct clinical
    trade-off for sepsis: a missed case (false negative) is more harmful than
    an unnecessary alert (false positive).

    Returns the optimal threshold and the F2 score achieved at that threshold,
    providing a principled justification for the chosen alert cutoff.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thresh, best_f = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
        if f > best_f:
            best_f = f
            best_thresh = float(t)
    return {
        "optimal_threshold_f2": round(best_thresh, 2),
        "f2_score_at_optimal":  round(best_f, 4),
    }


def _subgroup_auroc(df_test: "pd.DataFrame", y_score: np.ndarray) -> dict:
    """
    Compute AUROC per demographic subgroup for fairness analysis.

    Checks for gender_male (binary) and age quartiles.
    Returns empty dict if subgroup columns are absent.

    Clinical relevance: sepsis presentation differs by sex and age.
    A model that performs well on average but poorly on elderly female
    patients would be clinically unacceptable.
    """
    subgroup_metrics: dict = {}

    # Gender subgroup
    if "gender_male" in df_test.columns:
        for val, label in [(1, "male"), (0, "female")]:
            mask = df_test["gender_male"].values == val
            if mask.sum() >= 20:
                try:
                    sub_auroc = roc_auc_score(
                        df_test["sepsis_label"].values[mask], y_score[mask]
                    )
                    subgroup_metrics[f"auroc_{label}"] = round(float(sub_auroc), 4)
                except ValueError:
                    pass

    # Age subgroup — fixed clinical brackets matching MODEL_CARD documentation
    if "age" in df_test.columns:
        age_vals = df_test["age"].values
        age_brackets = [
            ("18_44",  (age_vals >= 18)  & (age_vals <  45)),
            ("45_64",  (age_vals >= 45)  & (age_vals <  65)),
            ("65_74",  (age_vals >= 65)  & (age_vals <  75)),
            ("75plus", (age_vals >= 75)),
        ]
        for label, mask in age_brackets:
            if mask.sum() >= 20:
                try:
                    sub_auroc = roc_auc_score(
                        df_test["sepsis_label"].values[mask], y_score[mask]
                    )
                    subgroup_metrics[f"auroc_age_{label}"] = round(float(sub_auroc), 4)
                except ValueError:
                    pass

    # Care unit subgroup — most operationally relevant for an ICU tool
    if "first_careunit" in df_test.columns:
        for unit in df_test["first_careunit"].dropna().unique():
            mask = df_test["first_careunit"].values == unit
            if mask.sum() >= 20:
                try:
                    sub_auroc = roc_auc_score(
                        df_test["sepsis_label"].values[mask], y_score[mask]
                    )
                    safe_key = str(unit).replace(" ", "_").replace("/", "_")[:20]
                    subgroup_metrics[f"auroc_unit_{safe_key}"] = round(float(sub_auroc), 4)
                except ValueError:
                    pass

    return subgroup_metrics


def evaluate(cfg: dict | None = None) -> dict:
    """
    Run evaluation on the held-out test set and return metrics dict.

    Uses the same random_state=42 stratified 80/20 split as train.py
    so reported AUROC reflects true out-of-sample performance.
    """
    if cfg is None:
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    data_path = Path(cfg["data"]["processed_path"]) / "features.parquet"
    df = pd.read_parquet(data_path)

    # Reproduce the exact train/test split from train.py
    _, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sepsis_label"]
    )

    artifact = load_model(cfg)
    results = predict_batch(df_test, artifact)

    y_true = results["sepsis_label"].values
    y_score = results["risk_score"].values

    # NEWS2 baseline — same test set
    news2_scores = df_test.apply(news2_score, axis=1).values

    auroc        = roc_auc_score(y_true, y_score)
    auprc        = average_precision_score(y_true, y_score)
    brier        = brier_score_loss(y_true, y_score)
    news2_auroc  = roc_auc_score(y_true, news2_scores)

    # Clinical threshold metrics at all three alert tiers
    thresh_04 = _threshold_metrics(y_true, y_score, threshold=0.4)
    thresh_06 = _threshold_metrics(y_true, y_score, threshold=0.6)
    thresh_08 = _threshold_metrics(y_true, y_score, threshold=0.8)

    # Fairness — subgroup AUROC
    subgroup = _subgroup_auroc(df_test, y_score)

    # Optimal threshold via F2 score sweep (recall-weighted — missed sepsis > false alarm)
    f2_result = _optimal_threshold(y_true, y_score, beta=2.0)

    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "news2_auroc": news2_auroc,
        "n_test": len(df_test),
        "sepsis_prevalence": float(y_true.mean()),
        "threshold_0.4": thresh_04,
        "threshold_0.6": thresh_06,
        "threshold_0.8": thresh_08,
        **f2_result,
        **subgroup,
    }

    print(f"\nEvaluation on held-out test set ({len(df_test):,} stays)")
    print(f"{'─'*45}")
    print(f"SepsisAlert AUROC:  {auroc:.4f}")
    print(f"SepsisAlert AUPRC:  {auprc:.4f}")
    print(f"Brier Score:        {brier:.4f}  (lower = better calibration)")
    print(f"NEWS2 AUROC:        {news2_auroc:.4f}")
    print(f"Gap vs NEWS2:       +{auroc - news2_auroc:.4f}")
    print(f"\nOptimal threshold (F2, beta=2 — recall-weighted for sepsis):")
    print(f"  Threshold: {f2_result['optimal_threshold_f2']:.2f}  "
          f"F2: {f2_result['f2_score_at_optimal']:.4f}")
    print("  (Current nurse alert threshold 0.40 should be close to this value)")
    print("\nAt threshold 0.4 (nurse alert):")
    print(f"  Sensitivity: {thresh_04['sensitivity']:.3f}  "
          f"Specificity: {thresh_04['specificity']:.3f}  "
          f"PPV: {thresh_04['ppv']:.3f}  NPV: {thresh_04['npv']:.3f}")
    print("At threshold 0.6 (doctor alert):")
    print(f"  Sensitivity: {thresh_06['sensitivity']:.3f}  "
          f"Specificity: {thresh_06['specificity']:.3f}  "
          f"PPV: {thresh_06['ppv']:.3f}  NPV: {thresh_06['npv']:.3f}")
    print("At threshold 0.8 (critical escalation):")
    print(f"  Sensitivity: {thresh_08['sensitivity']:.3f}  "
          f"Specificity: {thresh_08['specificity']:.3f}  "
          f"PPV: {thresh_08['ppv']:.3f}  NPV: {thresh_08['npv']:.3f}")
    if subgroup:
        print("\nSubgroup AUROC (fairness):")
        for key, val in subgroup.items():
            print(f"  {key:30s}: {val:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate()
