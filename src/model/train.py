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

    x_train, x_val, y_train, y_val = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
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

    val_proba = model.predict_proba(x_val)[:, 1]
    auroc = roc_auc_score(y_val, val_proba)
    print(f"\nValidation AUROC: {auroc:.4f}")

    print(classification_report(
        y_val, (val_proba >= 0.4).astype(int), target_names=["No Sepsis", "Sepsis"]
    ))

    # Compute per-feature training statistics for InputGuard z-score checks
    # and epistemic uncertainty perturbation scaling.
    training_stats = {
        col: {"mean": float(x_train[col].mean()), "std": float(x_train[col].std())}
        for col in feature_cols
    }

    # Compute training centroid and inverse covariance for Mahalanobis OOD check.
    feat_matrix = x_train.fillna(x_train.mean()).values.astype(float)
    training_mean = feat_matrix.mean(axis=0)
    try:
        cov = np.cov(feat_matrix, rowvar=False)
        training_cov_inv = np.linalg.pinv(cov)   # pseudo-inverse — handles correlated features
    except Exception:  # pylint: disable=broad-except
        training_cov_inv = None

    # Save
    artifact_path = Path(model_cfg["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":             model,
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
