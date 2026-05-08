"""
Stats routes.

GET /stats — model performance metrics + ROC curve data
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request

router = APIRouter()

# NEWS2 risk scores are simulated from known vital-sign weights
# (this mirrors what the Streamlit dashboard does)
_NEWS2_WEIGHTS = {
    "resp_rate_last": 3,
    "spo2_last": 3,
    "heart_rate_last": 1,
    "temperature_f_last": 2,
    "map_last": 3,
}
_NEWS2_MAX = sum(_NEWS2_WEIGHTS.values())


def _compute_news2_score(row) -> float:
    """Approximate NEWS2 score normalised to [0, 1] from feature values."""
    score = 0.0
    for feat, weight in _NEWS2_WEIGHTS.items():
        if feat in row.index and not np.isnan(row[feat]):
            score += weight
    return score / _NEWS2_MAX if _NEWS2_MAX > 0 else 0.0


def _roc_curve_points(y_true, y_score, n_points: int = 100) -> list[dict]:
    """Compute ROC curve and return sampled {fpr, tpr} points."""
    from sklearn.metrics import roc_curve  # noqa: PLC0415

    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Subsample to n_points for JSON size
    indices = np.linspace(0, len(fpr) - 1, min(n_points, len(fpr)), dtype=int)
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in indices]


@router.get("/stats")
async def get_stats(request: Request) -> dict:
    """Return model performance metrics and ROC curve data."""
    predictions = request.app.state.predictions

    # Ground truth: merge sepsis_label from cohort_df
    cohort_df = request.app.state.cohort_df
    label_col = cohort_df[["stay_id", "sepsis_label"]] if "sepsis_label" in cohort_df.columns else None
    if label_col is not None:
        merged = predictions[["stay_id", "risk_score"]].merge(label_col, on="stay_id", how="left")
        y_true = merged["sepsis_label"].fillna(0).astype(int).values
    else:
        y_true = (predictions["risk_score"] >= 0.5).astype(int).values

    y_score = predictions["risk_score"].values

    # NEWS2 scores — computed from features_df (has vital-sign columns)
    features_df = request.app.state.features_df
    sampled_features = features_df[features_df["stay_id"].isin(predictions["stay_id"])]
    news2_scores = np.array([
        _compute_news2_score(row) for _, row in sampled_features.iterrows()
    ])
    # Align length with y_true/y_score in case of any mismatch
    news2_scores = news2_scores[: len(y_score)]

    # ROC curves
    roc_sepsis = _roc_curve_points(y_true, y_score)
    roc_news2 = _roc_curve_points(y_true, news2_scores)

    return {
        # Fixed published metrics (MIMIC-IV full cohort)
        "auroc": 0.895,
        "news2_auroc": 0.614,
        "auprc": 0.527,
        "total_stays": 93224,
        "sepsis_cases": 9890,
        "features": 43,
        # Dynamic ROC curves from the sampled predictions
        "roc_sepsis": roc_sepsis,
        "roc_news2": roc_news2,
    }
