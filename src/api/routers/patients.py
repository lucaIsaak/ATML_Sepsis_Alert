"""
Patient routes — list and detail endpoints.

GET /patients            — list all sampled patients with risk scores
GET /patients/{stay_id}  — patient detail + SHAP features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

# Module-level caches so SHAP is only computed once per patient per server start
_shap_cache: dict[int, list[dict]] = {}
_explainer = None  # lazily initialised on first SHAP request


def _get_explainer(artifact: dict, features_df: pd.DataFrame):
    """Build (or return cached) SHAP explainer backed by a 100-row background."""
    global _explainer  # noqa: PLW0603
    if _explainer is None:
        from src.explainability.shap_explainer import build_explainer  # noqa: PLC0415
        feature_cols = artifact["feature_cols"]
        background = (
            features_df[feature_cols]
            .dropna()
            .sample(n=min(100, len(features_df)), random_state=42)
        )
        _explainer = build_explainer(artifact["model"], background)
    return _explainer


def _risk_label(score: float) -> str:
    if score >= 0.6:
        return "HIGH"
    if score >= 0.4:
        return "MODERATE"
    return "LOW"


def _row_to_patient(row: pd.Series) -> dict:
    return {
        "stay_id": int(row["stay_id"]),
        "risk_score": float(row["risk_score"]),
        "risk_label": _risk_label(float(row["risk_score"])),
        "age": float(row["age"]) if "age" in row.index and pd.notna(row["age"]) else None,
        "first_careunit": (
            str(row["first_careunit"])
            if "first_careunit" in row.index and pd.notna(row["first_careunit"])
            else "Unknown"
        ),
        "gender": (
            str(row["gender"])
            if "gender" in row.index and pd.notna(row["gender"])
            else None
        ),
    }


@router.get("/patients")
async def list_patients(request: Request) -> list[dict]:
    """Return all sampled patients sorted by risk_score descending."""
    predictions: pd.DataFrame = request.app.state.predictions
    patients = [_row_to_patient(row) for _, row in predictions.iterrows()]
    patients.sort(key=lambda p: p["risk_score"], reverse=True)
    return patients


@router.get("/patients/{stay_id}")
async def get_patient(stay_id: int, request: Request) -> dict:
    """Return patient detail with top/bottom SHAP features."""
    predictions: pd.DataFrame = request.app.state.predictions
    features_df: pd.DataFrame = request.app.state.features_df
    artifact = request.app.state.artifact

    # Look up in predictions for risk score / display columns
    pred_row = predictions[predictions["stay_id"] == stay_id]
    if pred_row.empty:
        raise HTTPException(status_code=404, detail=f"Patient {stay_id} not found")

    patient = _row_to_patient(pred_row.iloc[0])
    risk_score = patient["risk_score"]

    # SHAP computation
    if stay_id not in _shap_cache:
        # Feature values come from features_df, not predictions
        feat_row = features_df[features_df["stay_id"] == stay_id]
        if feat_row.empty:
            raise HTTPException(status_code=404, detail=f"Feature data for {stay_id} not found")

        from src.explainability.shap_explainer import explain_patient  # noqa: PLC0415

        feature_cols = artifact["feature_cols"]
        feature_vector = feat_row.iloc[0][feature_cols].values.astype(float)
        explainer = _get_explainer(artifact, features_df)

        explanation = explain_patient(
            explainer=explainer,
            feature_vector=feature_vector,
            feature_names=list(feature_cols),
            risk_score=risk_score,
            stay_id=str(stay_id),
            top_n=len(feature_cols),
        )
        _shap_cache[stay_id] = explanation.top_features

    all_features = _shap_cache[stay_id]

    # top = highest |shap|, bottom = lowest |shap|
    sorted_desc = sorted(all_features, key=lambda f: abs(f["shap"]), reverse=True)
    sorted_asc  = sorted(all_features, key=lambda f: abs(f["shap"]))

    def to_shap_dict(f: dict) -> dict:
        return {
            "label":   f.get("label", f.get("feature", "")),
            "shap":    float(f["shap"]),
            "value":   float(f["value"]) if f.get("value") is not None else 0.0,
            "feature": f.get("feature", f.get("label", "")),
        }

    return {
        **patient,
        "shap_top":    [to_shap_dict(f) for f in sorted_desc[:16]],
        "shap_bottom": [to_shap_dict(f) for f in sorted_asc[:8]],
    }
