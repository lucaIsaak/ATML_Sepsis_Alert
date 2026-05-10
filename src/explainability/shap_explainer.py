"""
SHAP explainability for the sepsis model.

Uses shap.Explainer (auto-selects best method for HistGradientBoosting).
Returns top N feature contributions per patient for the narrative layer.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


FEATURE_LABELS = {
    "heart_rate_mean": "Heart Rate (mean)",
    "heart_rate_min": "Heart Rate (min)",
    "heart_rate_max": "Heart Rate (max)",
    "heart_rate_last": "Heart Rate (last)",
    "map_mean": "Mean Art. Pressure (mean)",
    "map_min": "Mean Art. Pressure (min)",
    "map_max": "Mean Art. Pressure (max)",
    "map_last": "Mean Art. Pressure (last)",
    "resp_rate_mean": "Respiratory Rate (mean)",
    "resp_rate_min": "Respiratory Rate (min)",
    "resp_rate_max": "Respiratory Rate (max)",
    "resp_rate_last": "Respiratory Rate (last)",
    "temperature_f_mean": "Temperature (mean)",
    "temperature_f_min": "Temperature (min)",
    "temperature_f_max": "Temperature (max)",
    "temperature_f_last": "Temperature (last)",
    "spo2_mean": "SpO2 (mean)",
    "spo2_min": "SpO2 (min)",
    "spo2_max": "SpO2 (max)",
    "spo2_last": "SpO2 (last)",
    "lactate_last": "Lactate (last)",
    "lactate_mean": "Lactate (mean)",
    "lactate_delta": "Lactate (change)",
    "wbc_last": "WBC (last)",
    "wbc_mean": "WBC (mean)",
    "wbc_delta": "WBC (change)",
    "creatinine_last": "Creatinine (last)",
    "creatinine_mean": "Creatinine (mean)",
    "creatinine_delta": "Creatinine (change)",
    "bilirubin_last": "Bilirubin (last)",
    "bilirubin_mean": "Bilirubin (mean)",
    "bilirubin_delta": "Bilirubin (change)",
    "platelets_last": "Platelets (last)",
    "platelets_mean": "Platelets (mean)",
    "platelets_delta": "Platelets (change)",
    "bicarbonate_last": "Bicarbonate (last)",
    "bicarbonate_mean": "Bicarbonate (mean)",
    "bicarbonate_delta": "Bicarbonate (change)",
    "glucose_last": "Glucose (last)",
    "glucose_mean": "Glucose (mean)",
    "glucose_delta": "Glucose (change)",
    "heart_rate_trend": "Heart Rate (trend)",
    "map_trend": "Mean Art. Pressure (trend)",
    "resp_rate_trend": "Respiratory Rate (trend)",
    "temperature_f_trend": "Temperature (trend)",
    "spo2_trend": "SpO2 (trend)",
    "lactate_trend": "Lactate (trend)",
    "wbc_trend": "WBC (trend)",
    "creatinine_trend": "Creatinine (trend)",
    "bilirubin_trend": "Bilirubin (trend)",
    "platelets_trend": "Platelets (trend)",
    "bicarbonate_trend": "Bicarbonate (trend)",
    "glucose_trend": "Glucose (trend)",
    "age": "Age",
    "gender_male": "Gender (Male)",
}

FEATURE_UNITS = {
    "heart_rate": "bpm",
    "map": "mmHg",
    "resp_rate": "breaths/min",
    "temperature_f": "°F",
    "spo2": "%",
    "lactate": "mmol/L",
    "wbc": "K/µL",
    "creatinine": "mg/dL",
    "bilirubin": "mg/dL",
    "platelets": "K/µL",
    "bicarbonate": "mEq/L",
    "glucose": "mg/dL",
}


@dataclass
class SHAPExplanation:
    """SHAP-based explanation for a single patient's risk prediction."""

    stay_id: str
    risk_score: float
    risk_label: str
    top_features: list[dict]
    base_value: float


def _get_unit(feat_name: str) -> str:
    """Look up the clinical unit for a feature by checking name prefix."""
    for key, unit in FEATURE_UNITS.items():
        if feat_name.startswith(key):
            return unit
    return ""


def _extract_shap_vals(shap_values) -> tuple[np.ndarray, float]:
    """Extract class-1 SHAP values and base value from a shap output object."""
    if shap_values.values.ndim == 3:
        return shap_values.values[0, :, 1], float(shap_values.base_values[0, 1])
    return shap_values.values[0], float(shap_values.base_values[0])


def build_explainer(model, x_background=None) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for HistGradientBoosting (100× faster than Permutation).

    x_background is accepted for API compatibility but TreeExplainer does not
    require a background dataset — it uses the model's own tree structure.
    """
    return shap.TreeExplainer(model)


def explain_patient(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    explainer: shap.Explainer,
    feature_vector: np.ndarray,
    feature_names: list[str],
    risk_score: float,
    stay_id: str = "unknown",
    top_n: int = 5,
) -> SHAPExplanation:
    """Compute SHAP values for a single patient."""
    feat_df = pd.DataFrame([feature_vector], columns=feature_names)
    shap_values = explainer(feat_df)

    vals, base = _extract_shap_vals(shap_values)
    indices = np.argsort(np.abs(vals))[::-1][:top_n]

    top_features = []
    for idx in indices:
        feat_name = feature_names[idx]
        feat_val = feature_vector[idx]
        shap_val = float(vals[idx])
        top_features.append({
            "feature": feat_name,
            "label": FEATURE_LABELS.get(feat_name, feat_name),
            "value": float(feat_val) if not np.isnan(feat_val) else None,
            "unit": _get_unit(feat_name),
            "shap": shap_val,
            "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
        })

    label = "HIGH" if risk_score >= 0.6 else "MODERATE" if risk_score >= 0.4 else "LOW"

    return SHAPExplanation(
        stay_id=stay_id,
        risk_score=risk_score,
        risk_label=label,
        top_features=top_features,
        base_value=base,
    )


def format_for_narrative(explanation: SHAPExplanation) -> str:
    """Format SHAP explanation as structured string for LLM prompt."""
    lines = [
        f"Risk score: {explanation.risk_score:.2f} ({explanation.risk_label})",
        "Key drivers:",
    ]
    for feat in explanation.top_features:
        val_str = f"{feat['value']:.1f} {feat['unit']}" if feat["value"] is not None else "N/A"
        arrow = "↑" if feat["direction"] == "increases_risk" else "↓"
        lines.append(f"  - {feat['label']} = {val_str}  [{feat['shap']:+.3f} {arrow} risk]")
    return "\n".join(lines)
