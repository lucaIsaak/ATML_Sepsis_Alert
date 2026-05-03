"""
LightGBM training for sepsis prediction.

Input:  feature matrix (stay_id + features + sepsis_label)
Output: trained model saved to models/lightgbm_sepsis.pkl
"""

import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns (exclude meta columns)."""
    exclude = {"stay_id", "hadm_id", "sepsis_label"}
    return [c for c in df.columns if c not in exclude]


def train(cfg: dict | None = None) -> lgb.LGBMClassifier:
    if cfg is None:
        cfg = load_config()

    data_path = Path(cfg["data"]["processed_path"]) / "features.parquet"
    df = pd.read_parquet(data_path)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df["sepsis_label"]

    print(f"Training on {len(df):,} stays | "
          f"Sepsis prevalence: {y.mean():.1%} | "
          f"Features: {len(feature_cols)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_cfg = cfg["model"]
    model = lgb.LGBMClassifier(
        num_leaves=model_cfg["num_leaves"],
        learning_rate=model_cfg["learning_rate"],
        n_estimators=model_cfg["n_estimators"],
        min_child_samples=model_cfg["min_child_samples"],
        class_weight=model_cfg["class_weight"],
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, val_proba)
    print(f"\nValidation AUROC: {auroc:.4f}")

    val_pred = (val_proba >= 0.4).astype(int)
    print(classification_report(y_val, val_pred, target_names=["No Sepsis", "Sepsis"]))

    # Save
    artifact_path = Path(model_cfg["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols, "auroc": auroc}, artifact_path)
    print(f"Saved model to {artifact_path}")

    return model


if __name__ == "__main__":
    train()
