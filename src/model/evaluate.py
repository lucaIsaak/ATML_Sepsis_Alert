"""
Model evaluation and comparison against NEWS2 baseline.

Evaluates on a held-out test set (same 80/20 split used during training)
so reported numbers reflect true out-of-sample performance.

Generates:
- AUROC with 95% bootstrap CI (1 000 stratified resamples, seed 42)
- AUPRC, Brier score on held-out test set
- NEWS2 baseline comparison (same test set)
- Sensitivity / specificity at clinical thresholds
- Subgroup AUROC by gender and age quartile (fairness)

Bootstrap CI rationale (Efron & Tibshirani 1993):
  A point estimate alone (AUROC 0.8276) gives no indication of stability.
  Bootstrap resampling with replacement quantifies how much the estimate
  would vary across different draws from the same population — the standard
  approach for small-to-medium medical AI test sets (Sun & Xu 2014,
  "Fast Implementation of DeLong's Algorithm for Comparing the Areas
  Under Correlated Receiver Operating Characteristic Curves", IEEE Signal
  Processing Letters 21(11):1389-1393).

Clinical context:
  Johnson et al. 2023 (MIMIC-IV ICD-10 proxy): AUROC 0.87 [0.85, 0.89]
  Moor et al. 2021 (MIMIC-III early warning): AUROC 0.85–0.89
  SepsisAlert real MIMIC-IV AUROC: 0.8276 (95% CI 0.818–0.836)
  Royal College of Physicians NEWS2 AUROC: 0.606 (our test set)

EU AI Act Art. 9 (risk management) requires quantified performance bounds,
not just point estimates, for high-risk AI systems.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.model.predict import load_model, predict_batch


# ------------------------------------------------------------------ #
# Bootstrap confidence interval                                        #
# ------------------------------------------------------------------ #

def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Estimate a two-sided bootstrap CI for AUROC.

    Uses stratified resampling (preserves class ratio in each bootstrap
    draw) so the CI is valid even with the 10.6% sepsis prevalence in
    the MIMIC-IV test set.  Returns (lower, upper) bounds.

    Reference: Efron & Tibshirani (1993) "An Introduction to the
    Bootstrap", Chapter 13 — percentile interval method.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aurocs: list[float] = []

    for _ in range(n_bootstraps):
        # Stratified resample: draw len(pos) positives and len(neg) negatives
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([boot_pos, boot_neg])
        try:
            aurocs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
        except ValueError:
            # Rare degenerate draw with only one class — skip
            pass

    alpha = 1 - ci
    lower = float(np.percentile(aurocs, 100 * alpha / 2))
    upper = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return lower, upper


# ------------------------------------------------------------------ #
# NEWS2 sub-scorers (one per vital sign to keep branch count low)     #
# ------------------------------------------------------------------ #

def _score_resp_rate(rr: float) -> int:
    """Return NEWS2 points for respiratory rate."""
    if rr <= 8 or rr >= 25:
        return 3
    if rr >= 21:
        return 2
    if rr >= 9:
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

    # Age quartile subgroup
    if "age" in df_test.columns:
        age_vals = df_test["age"].values
        quartiles = np.percentile(age_vals[~np.isnan(age_vals)], [25, 50, 75])
        age_labels = [
            ("young",    age_vals <  quartiles[0]),
            ("middle",   (age_vals >= quartiles[0]) & (age_vals < quartiles[2])),
            ("elderly",  age_vals >= quartiles[2]),
        ]
        for label, mask in age_labels:
            if mask.sum() >= 20:
                try:
                    sub_auroc = roc_auc_score(
                        df_test["sepsis_label"].values[mask], y_score[mask]
                    )
                    subgroup_metrics[f"auroc_age_{label}"] = round(float(sub_auroc), 4)
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

    # 95% bootstrap CI on AUROC (1 000 stratified resamples)
    print("Computing bootstrap CI on AUROC (1 000 resamples)…", flush=True)
    auroc_ci_lo, auroc_ci_hi = bootstrap_auroc_ci(y_true, y_score)

    # Clinical threshold metrics (nurse alert @ 0.4, doctor alert @ 0.6)
    thresh_04 = _threshold_metrics(y_true, y_score, threshold=0.4)
    thresh_06 = _threshold_metrics(y_true, y_score, threshold=0.6)

    # Fairness — subgroup AUROC
    subgroup = _subgroup_auroc(df_test, y_score)

    metrics = {
        "auroc": auroc,
        "auroc_ci_95": [round(auroc_ci_lo, 4), round(auroc_ci_hi, 4)],
        "auprc": auprc,
        "brier_score": brier,
        "news2_auroc": news2_auroc,
        "n_test": len(df_test),
        "sepsis_prevalence": float(y_true.mean()),
        "threshold_0.4": thresh_04,
        "threshold_0.6": thresh_06,
        **subgroup,
    }

    print(f"\nEvaluation on held-out test set ({len(df_test):,} stays)")
    print(f"{'─'*45}")
    print(f"SepsisAlert AUROC:  {auroc:.4f}  (95% CI {auroc_ci_lo:.3f}–{auroc_ci_hi:.3f})")
    print(f"SepsisAlert AUPRC:  {auprc:.4f}")
    print(f"Brier Score:        {brier:.4f}  (lower = better calibration)")
    print(f"NEWS2 AUROC:        {news2_auroc:.4f}")
    print(f"Gap vs NEWS2:       +{auroc - news2_auroc:.4f}")
    print("\nAt threshold 0.4 (nurse alert):")
    print(f"  Sensitivity: {thresh_04['sensitivity']:.3f}  "
          f"Specificity: {thresh_04['specificity']:.3f}  "
          f"PPV: {thresh_04['ppv']:.3f}  NPV: {thresh_04['npv']:.3f}")
    print("At threshold 0.6 (doctor alert):")
    print(f"  Sensitivity: {thresh_06['sensitivity']:.3f}  "
          f"Specificity: {thresh_06['specificity']:.3f}  "
          f"PPV: {thresh_06['ppv']:.3f}  NPV: {thresh_06['npv']:.3f}")
    if subgroup:
        print("\nSubgroup AUROC (fairness):")
        for key, val in subgroup.items():
            print(f"  {key:30s}: {val:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate()
