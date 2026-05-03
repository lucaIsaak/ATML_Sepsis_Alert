"""
Model inference for a single patient or batch.

Loads the saved LightGBM artifact and returns a risk score (0–1)
along with the feature vector used (needed for SHAP).
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict | None = None) -> dict:
    """Load model artifact. Returns dict with 'model', 'feature_cols', 'auroc'."""
    if cfg is None:
        cfg = load_config()
    path = Path(cfg["model"]["artifact_path"])
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run src/model/train.py first.")
    return joblib.load(path)


def predict_patient(features: dict, artifact: dict | None = None) -> dict:
    """
    Predict sepsis risk for a single patient.

    Args:
        features: dict of {feature_name: value} — must include all model features
        artifact: loaded model artifact (will load from disk if None)

    Returns:
        {
            "risk_score": float,       # 0–1 probability
            "risk_label": str,         # "HIGH" / "MODERATE" / "LOW"
            "feature_vector": ndarray, # aligned feature array for SHAP
        }
    """
    if artifact is None:
        artifact = load_model()

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    # Align to expected feature order, fill missing with NaN
    X = pd.DataFrame([features])[feature_cols]
    risk_score = float(model.predict_proba(X)[0, 1])

    if risk_score >= 0.6:
        label = "HIGH"
    elif risk_score >= 0.4:
        label = "MODERATE"
    else:
        label = "LOW"

    return {
        "risk_score": risk_score,
        "risk_label": label,
        "feature_vector": X.values[0],
        "feature_names": feature_cols,
    }


def predict_batch(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """
    Predict for a DataFrame of patients.

    Returns df with added columns: risk_score, risk_label.
    """
    if artifact is None:
        artifact = load_model()

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["risk_score"] = proba
    result["risk_label"] = pd.cut(
        proba,
        bins=[0, 0.4, 0.6, 1.0],
        labels=["LOW", "MODERATE", "HIGH"],
        include_lowest=True,
    )
    return result
