"""
Model evaluation and comparison against NEWS2 baseline.

Generates:
- AUROC, AUPRC curves
- Sensitivity / specificity at different thresholds
- Alert fatigue comparison vs NEWS2 rule-based threshold
- Calibration plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

from src.model.predict import load_model, predict_batch


def news2_score(row: pd.Series) -> int:
    """
    Compute a simplified NEWS2 score from available features.
    Returns integer score (higher = more abnormal).
    NEWS2 thresholds: >=7 = high risk
    """
    score = 0

    # Respiratory rate
    rr = row.get("resp_rate_mean", np.nan)
    if not np.isnan(rr):
        if rr <= 8 or rr >= 25:
            score += 3
        elif rr >= 21:
            score += 2
        elif rr >= 9:
            score += 1

    # SpO2
    spo2 = row.get("spo2_min", np.nan)
    if not np.isnan(spo2):
        if spo2 <= 91:
            score += 3
        elif spo2 <= 93:
            score += 2
        elif spo2 <= 95:
            score += 1

    # Heart rate
    hr = row.get("heart_rate_mean", np.nan)
    if not np.isnan(hr):
        if hr <= 40 or hr >= 131:
            score += 3
        elif hr >= 111 or hr <= 50:
            score += 2
        elif hr >= 91 or hr <= 50:
            score += 1

    # Consciousness / temp (simplified)
    temp = row.get("temperature_f_last", np.nan)
    if not np.isnan(temp):
        temp_c = (temp - 32) * 5 / 9
        if temp_c <= 35 or temp_c >= 39.1:
            score += 2
        elif temp_c >= 38.1:
            score += 1

    return score


def evaluate(cfg: dict | None = None) -> dict:
    """Run full evaluation and return metrics dict."""
    import yaml
    if cfg is None:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

    data_path = Path(cfg["data"]["processed_path"]) / "features.parquet"
    df = pd.read_parquet(data_path)

    artifact = load_model(cfg)
    results = predict_batch(df, artifact)

    y_true = results["sepsis_label"].values
    y_score = results["risk_score"].values

    # NEWS2 baseline
    news2_scores = df.apply(news2_score, axis=1).values
    news2_binary = (news2_scores >= 7).astype(int)

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    news2_auroc = roc_auc_score(y_true, news2_scores)

    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "news2_auroc": news2_auroc,
        "n_patients": len(df),
        "sepsis_prevalence": float(y_true.mean()),
    }

    print(f"SepsisAlert AUROC: {auroc:.4f}")
    print(f"SepsisAlert AUPRC: {auprc:.4f}")
    print(f"NEWS2 AUROC:       {news2_auroc:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate()
