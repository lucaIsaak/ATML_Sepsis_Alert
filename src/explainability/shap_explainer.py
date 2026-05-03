"""
SHAP explainability wrapper for the LightGBM sepsis model.

For each prediction, returns the top N feature contributions
that drove the risk score — used as input to the narrative generator.
"""

import shap
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SHAPExplanation:
    """Structured output of SHAP analysis for one patient."""
    stay_id: str
    risk_score: float
    top_features: list[dict]   # [{"feature": str, "value": float, "shap": float, "direction": str}]
    base_value: float          # expected model output (population mean risk)


def build_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for LightGBM.

    Args:
        model: trained LGBMClassifier
        X_background: sample of training data for background distribution
                      (100–500 rows is usually enough)
    """
    return shap.TreeExplainer(model, data=X_background, model_output="probability")


def explain_patient(
    explainer: shap.TreeExplainer,
    feature_vector: np.ndarray,
    feature_names: list[str],
    risk_score: float,
    stay_id: str = "unknown",
    top_n: int = 5,
) -> SHAPExplanation:
    """
    Compute SHAP values for a single patient and return structured explanation.

    Args:
        explainer: SHAP TreeExplainer
        feature_vector: 1D array of feature values (aligned to feature_names)
        feature_names: list of feature names
        risk_score: model output probability
        stay_id: patient identifier
        top_n: number of top features to return

    Returns:
        SHAPExplanation with top contributing features
    """
    X = pd.DataFrame([feature_vector], columns=feature_names)
    shap_values = explainer(X)

    # shap_values.values shape: (1, n_features) for binary classification probability output
    vals = shap_values.values[0]
    base = float(shap_values.base_values[0])

    # Sort by absolute contribution
    indices = np.argsort(np.abs(vals))[::-1][:top_n]

    top_features = []
    for i in indices:
        feat_val = float(feature_vector[i]) if not np.isnan(feature_vector[i]) else None
        shap_val = float(vals[i])
        top_features.append({
            "feature": feature_names[i],
            "value": feat_val,
            "shap": shap_val,
            "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
        })

    return SHAPExplanation(
        stay_id=stay_id,
        risk_score=risk_score,
        top_features=top_features,
        base_value=base,
    )


def format_for_narrative(explanation: SHAPExplanation) -> str:
    """
    Format SHAP explanation as a structured string for the LLM prompt.

    Example output:
        Risk score: 0.73 (HIGH)
        Key drivers (top 5):
        - lactate_last = 4.2 mmol/L  [+0.18 ↑ risk]
        - wbc_last = 18.5 k/uL       [+0.12 ↑ risk]
        - map_min = 58 mmHg           [+0.09 ↑ risk]
        - heart_rate_max = 128 bpm    [+0.07 ↑ risk]
        - creatinine_last = 2.1 mg/dL [+0.06 ↑ risk]
    """
    label = "HIGH" if explanation.risk_score >= 0.6 else "MODERATE" if explanation.risk_score >= 0.4 else "LOW"
    lines = [
        f"Risk score: {explanation.risk_score:.2f} ({label})",
        "Key drivers:",
    ]
    for feat in explanation.top_features:
        val_str = f"{feat['value']:.2f}" if feat["value"] is not None else "N/A"
        arrow = "↑" if feat["direction"] == "increases_risk" else "↓"
        lines.append(f"  - {feat['feature']} = {val_str}  [{feat['shap']:+.3f} {arrow} risk]")
    return "\n".join(lines)
