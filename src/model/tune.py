# pylint: disable=duplicate-code
"""
Optuna hyperparameter tuning for the sepsis model.

Uses Bayesian optimisation — much smarter than grid search.
Runs N trials, each trying a different set of hyperparameters,
and learns from previous trials to focus on the most promising regions.

Usage:
    python -m src.model.tune              # default: 50 trials
    python -m src.model.tune --trials 100

After tuning, the best model is saved to the same artifact path as train.py.
The best hyperparameters are printed and can be pasted into config.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file and return as dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns (exclude meta columns)."""
    exclude = {"stay_id", "hadm_id", "sepsis_label"}
    return [c for c in df.columns if c not in exclude]


def _make_objective(features: pd.DataFrame, labels: pd.Series):
    """Return an Optuna objective function closed over the training data."""

    def objective(trial: optuna.Trial) -> float:
        """Suggest hyperparameters and return cross-validated AUROC."""
        params = {
            "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 15, 255),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter":          trial.suggest_int("max_iter", 200, 1000),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "class_weight":      "balanced",
            "random_state":      42,
            "early_stopping":    False,   # handled by max_iter in CV context
        }

        model = HistGradientBoostingClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, features, labels,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        return float(scores.mean())

    return objective


def tune(n_trials: int = 50, cfg: dict | None = None) -> dict:
    """
    Run Optuna hyperparameter search and return the best params.

    Uses 5-fold stratified cross-validation as the objective — more
    reliable than a single train/val split.
    """
    if cfg is None:
        cfg = load_config()

    data_path = Path(cfg["data"]["processed_path"]) / "features.parquet"
    df = pd.read_parquet(data_path)

    feature_cols = get_feature_cols(df)

    # Split first — CV and final fit both on train split only to avoid leakage
    x_train, _, y_train, _ = train_test_split(
        df[feature_cols], df["sepsis_label"],
        test_size=0.2, random_state=42, stratify=df["sepsis_label"],
    )

    print(f"Tuning on {len(x_train):,} train stays | {n_trials} Optuna trials | 5-fold CV")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        _make_objective(x_train, y_train), n_trials=n_trials, show_progress_bar=True
    )

    best = study.best_params
    print(f"\nBest AUROC (CV): {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for k, v in best.items():
        print(f"  {k:25s}: {v}")

    # Retrain on train split only (not full dataset) to preserve test-set integrity
    best_model = HistGradientBoostingClassifier(
        **best,
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )
    best_model.fit(x_train, y_train)

    # Compute training stats for InputGuard and epistemic uncertainty
    training_stats = {
        col: {"mean": float(x_train[col].mean()), "std": float(x_train[col].std())}
        for col in feature_cols
    }
    feat_matrix = x_train.fillna(x_train.mean()).values.astype(float)
    training_mean = feat_matrix.mean(axis=0)
    try:
        cov = np.cov(feat_matrix, rowvar=False)
        training_cov_inv = np.linalg.pinv(cov)
    except Exception:  # pylint: disable=broad-except
        training_cov_inv = None

    artifact_path = Path(cfg["model"]["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model":            best_model,
            "feature_cols":     feature_cols,
            "auroc":            study.best_value,
            "training_stats":   training_stats,
            "training_mean":    training_mean,
            "training_cov_inv": training_cov_inv,
        },
        artifact_path,
    )
    print(f"\nSaved tuned model to {artifact_path}")
    print("\nPaste into config.yaml → model section:")
    print(f"  num_leaves:         {best.get('max_leaf_nodes', 64)}")
    print(f"  learning_rate:      {best.get('learning_rate', 0.05):.4f}")
    print(f"  n_estimators:       {best.get('max_iter', 500)}")
    print(f"  min_child_samples:  {best.get('min_samples_leaf', 20)}")
    print(f"  max_depth:          {best.get('max_depth', 'null')}")
    print(f"  l2_regularization:  {best.get('l2_regularization', 0.0):.4f}")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()
    tune(n_trials=args.trials)
