"""
Gradient boosting training for sepsis prediction.

Uses sklearn HistGradientBoostingClassifier — same algorithm as LightGBM,
pure Python, no native library dependencies, natively handles NaN values.

Input:  feature matrix (stay_id + features + sepsis_label)
Output: trained model saved to models/sepsis_model.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file and return as dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns (exclude meta columns)."""
    exclude = {"stay_id", "hadm_id", "sepsis_label"}
    return [c for c in df.columns if c not in exclude]


def train(cfg: dict | None = None) -> HistGradientBoostingClassifier:
    """Train the sepsis model and save the artifact to disk."""
    if cfg is None:
        cfg = load_config()

    data_path = Path(cfg["data"]["processed_path"]) / "features.parquet"
    df = pd.read_parquet(data_path)

    feature_cols = get_feature_cols(df)
    features = df[feature_cols]
    y = df["sepsis_label"]

    print(f"Training on {len(df):,} stays | "
          f"Sepsis prevalence: {y.mean():.1%} | "
          f"Features: {len(feature_cols)}")

    # 3-way split: 72% train / 8% calibration / 20% test
    # The 20% test set (random_state=42, test_size=0.2) matches evaluate.py exactly.
    # Calibration is fitted on held-out data the model never saw — no data leakage.
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
    )

    model_cfg = cfg["model"]
    # early_stopping disabled — max_iter is Optuna-tuned (859); using full 80%
    # training set ensures the documented 80/20 split is accurate.
    model = HistGradientBoostingClassifier(
        max_leaf_nodes=model_cfg["num_leaves"],
        learning_rate=model_cfg["learning_rate"],
        max_iter=model_cfg["n_estimators"],
        min_samples_leaf=model_cfg["min_child_samples"],
        max_depth=model_cfg.get("max_depth"),           # None = no depth limit
        l2_regularization=model_cfg.get("l2_regularization", 0.0),
        class_weight=model_cfg["class_weight"],
        random_state=42,
        early_stopping=False,
        verbose=1,
    )

    model.fit(x_train, y_train)

    # Isotonic calibration on the held-out calibration split.
    # Makes risk scores empirically meaningful: 0.6 ≈ 60% actual sepsis rate.
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(x_cal, y_cal)

    val_proba = calibrated.predict_proba(x_test)[:, 1]
    auroc = roc_auc_score(y_test, val_proba)
    print(f"\nValidation AUROC (calibrated): {auroc:.4f}")

    print(classification_report(
        y_test, (val_proba >= 0.4).astype(int), target_names=["No Sepsis", "Sepsis"]
    ))

    # Training statistics computed from x_train only (the model's actual training data).
    training_stats = {
        col: {"mean": float(x_train[col].mean()), "std": float(x_train[col].std())}
        for col in feature_cols
    }

    # Covariance for Mahalanobis OOD check — from x_train only.
    feat_matrix = x_train.fillna(x_train.mean()).values.astype(float)
    training_mean = feat_matrix.mean(axis=0)
    try:
        cov = np.cov(feat_matrix, rowvar=False)
        training_cov_inv = np.linalg.pinv(cov)
    except Exception:  # pylint: disable=broad-except
        training_cov_inv = None

    # Save both calibrated model (for inference) and raw base model (for SHAP).
    artifact_path = Path(model_cfg["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":             calibrated,   # calibrated — use for predict_proba
            "base_model":        model,         # raw HistGBM — use for SHAP
            "feature_cols":      feature_cols,
            "auroc":             auroc,
            "training_stats":    training_stats,
            "training_mean":     training_mean,
            "training_cov_inv":  training_cov_inv,
        },
        artifact_path,
    )
    print(f"Saved model to {artifact_path}")

    return model


if __name__ == "__main__":
    train()
