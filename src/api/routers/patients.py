"""
Patient routes — list and detail endpoints.

GET /patients            — list all sampled patients with risk scores
GET /patients/{stay_id}  — patient detail + SHAP features + epistemic uncertainty
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.safety.guardrails import AuditLogger

router = APIRouter()

# Module-level caches so SHAP, OOD, and uncertainty are computed once per patient
_shap_cache:        dict[int, list[dict]] = {}
_ood_cache:         dict[int, dict] = {}   # stay_id → {flag, outlier_features, ...}
_uncertainty_cache: dict[int, dict] = {}   # stay_id → uncertainty dict
_cache_lock       = threading.Lock()       # guards all three caches + lazy singletons
_explainer        = None   # lazily initialised on first SHAP request
_explainer_lock   = threading.Lock()
_input_guard      = None   # lazily initialised on first OOD request
_audit_logger = AuditLogger(log_path="logs/audit.jsonl")


def _get_input_guard(artifact: dict):
    """Return (or build) the InputGuard for the loaded artifact."""
    global _input_guard  # noqa: PLW0603
    if _input_guard is None:
        from src.safety.guardrails import InputGuard  # noqa: PLC0415
        _input_guard = InputGuard.from_artifact(artifact)
    return _input_guard


def _get_explainer(artifact: dict, features_df: pd.DataFrame):
    """Build (or return cached) SHAP TreeExplainer (thread-safe lazy init)."""
    global _explainer  # noqa: PLW0603
    with _explainer_lock:
        if _explainer is None:
            from src.explainability.shap_explainer import build_explainer  # noqa: PLC0415
            _explainer = build_explainer(artifact["model"])
    return _explainer


def _risk_label(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
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
    """Return patient detail with top/bottom SHAP features and epistemic uncertainty."""
    predictions: pd.DataFrame = request.app.state.predictions
    features_df: pd.DataFrame = request.app.state.features_df
    artifact = request.app.state.artifact

    # Look up in predictions for risk score / display columns
    pred_row = predictions[predictions["stay_id"] == stay_id]
    if pred_row.empty:
        raise HTTPException(status_code=404, detail=f"Patient {stay_id} not found")

    patient = _row_to_patient(pred_row.iloc[0])
    risk_score = patient["risk_score"]

    # Fetch the raw feature row once — needed for OOD, SHAP, and uncertainty
    feat_row = features_df[features_df["stay_id"] == stay_id]
    if feat_row.empty:
        raise HTTPException(status_code=404, detail=f"Feature data for {stay_id} not found")

    feature_cols   = artifact["feature_cols"]
    feature_vector = feat_row.iloc[0][feature_cols].values.astype(float)

    # ── OOD / InputGuard check ────────────────────────────────────────
    with _cache_lock:
        if stay_id not in _ood_cache:
            guard        = _get_input_guard(artifact)
            feature_dict = feat_row.iloc[0].to_dict()
            ood_result   = guard.check(feature_dict)
            _ood_cache[stay_id] = {
                "ood_flag":             ood_result.confidence_flag,
                "outlier_features":     ood_result.outlier_features,
                "mahalanobis_distance": ood_result.mahalanobis_distance,
                "multivariate_novel":   ood_result.multivariate_novel,
            }
        ood_info = _ood_cache[stay_id]

    # ── Epistemic uncertainty (MC perturbation) ───────────────────────
    with _cache_lock:
        if stay_id not in _uncertainty_cache:
            try:
                from src.model.uncertainty import estimate_uncertainty  # noqa: PLC0415
                _uncertainty_cache[stay_id] = estimate_uncertainty(
                    model=artifact["model"],
                    feature_vector=feature_vector,
                    feature_names=list(feature_cols),
                    training_stats=artifact.get("training_stats", {}),
                )
            except Exception as _exc:  # pylint: disable=broad-except
                _uncertainty_cache[stay_id] = {
                    "uncertainty_flag": "LOW",
                    "is_uncertain": False,
                    "ci_lower": None,
                    "ci_upper": None,
                    "ci_width": None,
                    "variance": None,
                    "std": None,
                    "point_estimate": risk_score,
                    "n_samples": 0,
                    "error": str(_exc),
                }
        epistemic = _uncertainty_cache[stay_id]

    # ── SHAP computation ──────────────────────────────────────────────
    with _cache_lock:
        if stay_id not in _shap_cache:
            from src.explainability.shap_explainer import explain_patient  # noqa: PLC0415

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

    shap_top    = [to_shap_dict(f) for f in sorted_desc[:16]]
    shap_bottom = [to_shap_dict(f) for f in sorted_asc[:8]]

    # ── Audit log (GDPR Art. 22 / EU AI Act) ─────────────────────────
    _audit_logger.log_prediction(
        stay_id=str(stay_id),
        risk_score=risk_score,
        risk_tier=patient["risk_label"],
        ood_flag=ood_info.get("ood_flag", "NORMAL"),
        outlier_features=ood_info.get("outlier_features", []),
        top_features=shap_top[:5],
    )

    return {
        **patient,
        **ood_info,
        "epistemic_uncertainty": epistemic,
        "shap_top":    shap_top,
        "shap_bottom": shap_bottom,
    }
