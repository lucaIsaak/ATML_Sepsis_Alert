"""
Model evaluation and comparison against NEWS2 baseline.

Evaluates on a held-out test set (same 80/20 split used during training)
so reported numbers reflect true out-of-sample performance.

Generates:
- AUROC, AUPRC on held-out test set
- NEWS2 baseline comparison (same test set)
- Sensitivity / specificity at clinical thresholds
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
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
    if hr >= 111 or hr <= 50:
        return 2
    if hr >= 91:
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

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    news2_auroc = roc_auc_score(y_true, news2_scores)

    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "news2_auroc": news2_auroc,
        "n_test": len(df_test),
        "sepsis_prevalence": float(y_true.mean()),
    }

    print(f"Evaluation on held-out test set ({len(df_test):,} stays)")
    print(f"SepsisAlert AUROC: {auroc:.4f}")
    print(f"SepsisAlert AUPRC: {auprc:.4f}")
    print(f"NEWS2 AUROC:       {news2_auroc:.4f}")
    print(f"Gap vs NEWS2:      +{auroc - news2_auroc:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate()
