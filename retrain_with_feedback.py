"""
SepsisAlert — Feedback-Driven Retraining Script.

PURPOSE
=======
Incorporates clinician feedback collected via the dashboard into the next
model training cycle.  Run this manually whenever enough new feedback has
accumulated — it is intentionally NOT called automatically.

HOW IT WORKS
============
1. Load the original feature matrix (data/processed/features.parquet)
   with its original MIMIC-IV / synthetic labels.
2. Load all clinician feedback from data/feedback/feedback.csv.
3. For every patient that has been labelled by a clinician:
     - Override their original label with the clinician's label.
     - Assign a higher sample weight so the model pays extra attention
       to these hand-verified examples.
     - "flagged_wrong" labels get a reduced weight (0.5) because a
       clinician saying the alert was wrong does not definitively mean
       the patient never had sepsis.
4. Retrain a new HistGradientBoostingClassifier on the combined dataset.
5. Compare AUROC of old model vs. new model on a held-out validation set.
6. If the new model is better (or --force is passed), back up the old
   model and save the new one to models/sepsis_model.pkl.

SIGNAL QUALITY LIMITATION (ARCHITECTURAL NOTE)
===============================================
This script retrains on the *full* 93,224-patient MIMIC-IV dataset, with
clinician feedback rows merely overriding a small number of labels (typically
5–50 in a production deployment, even fewer in a demo).  At that ratio the
per-feedback sample weight of 3.0 can nudge the loss function by only a tiny
fraction — it is unlikely to measurably change the learned decision boundary,
and the AUROC delta between old and new model will usually be noise.

A more effective approach for future iterations:

  Option A — Threshold calibration only
      Keep the pre-trained model frozen.  Fit a Platt-scaling layer or
      isotonic regression exclusively on the labelled feedback rows to
      recalibrate the output probability.  Requires as few as ~20 labels.

  Option B — Fine-tuning on feedback only
      Hold the base model fixed and train a small meta-learner (logistic
      regression over the top SHAP features) on feedback rows only.
      This avoids catastrophic forgetting of the MIMIC-IV training signal.

  Option C — Active learning queue
      Surface the model's most uncertain predictions (risk ≈ 0.5) to
      clinicians first, maximising label efficiency before triggering
      a retrain.

Until one of these strategies is adopted, AUROC comparisons after a
feedback-driven retrain should be interpreted with caution — a stable
(near-zero delta) result is expected and does not indicate model degradation.

USAGE
=====
    # Dry run — prints everything but does NOT overwrite the model
    python retrain_with_feedback.py --dry-run

    # Full retrain — saves new model only if AUROC improves
    python retrain_with_feedback.py

    # Force save even if AUROC is lower (use with caution)
    python retrain_with_feedback.py --force

SAMPLE WEIGHTS EXPLAINED
=========================
  Original labels (automated):   weight = 1.0
  Clinician confirmed_sepsis:     weight = 3.0  (high-confidence positive)
  Clinician flagged_wrong:        weight = 0.5  (provisional negative —
                                                  absence of diagnosis ≠
                                                  absence of disease)

OUTPUT
======
  models/sepsis_model.pkl            ← updated model (if saved)
  models/sepsis_model_backup_<ts>.pkl ← backup of previous model
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------ #
# Path setup — allow running from repo root                           #
# ------------------------------------------------------------------ #
sys.path.insert(0, str(Path(__file__).parent))

from src.data.feedback import load_training_labels  # noqa: E402

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

ARTIFACT_PATH   = Path("models/sepsis_model.pkl")
FEATURES_PATH   = Path("data/processed/features.parquet")
CONFIG_PATH     = Path("config.yaml")

# Sample weights
W_ORIGINAL          = 1.0   # automated MIMIC-IV / synthetic label
W_CONFIRMED_SEPSIS  = 3.0   # clinician says: yes, this was sepsis
W_FLAGGED_WRONG     = 0.5   # clinician says: alert was probably wrong


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def _backup_model(artifact_path: Path) -> Path:
    """Copy current model to a timestamped backup file."""
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = artifact_path.parent / f"sepsis_model_backup_{ts}.pkl"
    shutil.copy2(artifact_path, backup)
    return backup


# ------------------------------------------------------------------ #
# Core logic                                                           #
# ------------------------------------------------------------------ #

def build_training_set() -> tuple[pd.DataFrame, pd.Series, np.ndarray, list[str]]:
    """
    Merge original features with clinician feedback labels.

    Returns
    -------
    X            : feature DataFrame
    y            : label Series  (0 / 1)
    weights      : sample weight array
    feature_cols : list of feature column names
    """
    # ── Original data ──────────────────────────────────────────
    features_df = pd.read_parquet(FEATURES_PATH)
    meta_cols   = {"stay_id", "hadm_id", "sepsis_label"}
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    # ── Feedback labels ────────────────────────────────────────
    feedback = load_training_labels()   # stay_id, sepsis_label, low_confidence

    n_feedback = len(feedback)
    n_confirmed = int((feedback["feedback_type"] == "confirmed_sepsis").sum())
    n_flagged   = int((feedback["feedback_type"] == "flagged_wrong").sum())

    print(f"  Feedback rows loaded : {n_feedback}")
    print(f"    confirmed_sepsis   : {n_confirmed}")
    print(f"    flagged_wrong      : {n_flagged}")

    # ── Merge: clinician labels override original labels ───────
    df = features_df.copy()

    if n_feedback > 0:
        # Build a mapping: stay_id → (new_label, low_confidence)
        fb_map = feedback.set_index("stay_id")[
            ["sepsis_label", "low_confidence"]
        ].to_dict(orient="index")

        original_labels  = df["sepsis_label"].copy()
        overridden_count = 0

        for idx, row in df.iterrows():
            sid = int(row["stay_id"])
            if sid in fb_map:
                df.at[idx, "sepsis_label"] = fb_map[sid]["sepsis_label"]
                overridden_count += 1

        print(f"  Labels overridden by clinician feedback: {overridden_count}")
    else:
        print("  No feedback yet — training on original labels only.")

    # ── Sample weights ─────────────────────────────────────────
    weights = np.full(len(df), W_ORIGINAL)

    if n_feedback > 0:
        for idx, row in df.iterrows():
            sid = int(row["stay_id"])
            if sid in fb_map:
                if fb_map[sid]["low_confidence"]:
                    weights[idx] = W_FLAGGED_WRONG
                else:
                    weights[idx] = W_CONFIRMED_SEPSIS

    X = df[feature_cols]
    y = df["sepsis_label"]

    print(f"\n  Training set size    : {len(df):,} patients")
    print(f"  Sepsis prevalence    : {y.mean():.1%}")
    print(f"  Features             : {len(feature_cols)}")

    return X, y, weights, feature_cols


def train_new_model(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    cfg: dict,
) -> tuple[HistGradientBoostingClassifier, float, pd.Index, pd.Series]:
    """
    Train a new model and return it together with its validation AUROC.

    Returns
    -------
    model       : trained classifier
    val_auroc   : AUROC on the held-out validation split
    x_val_idx   : index of validation rows (for weight-aware reporting)
    y_val       : validation labels
    """
    x_train, x_val, y_train, y_val, w_train, _ = train_test_split(
        X, y, weights,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_cfg = cfg["model"]
    model = HistGradientBoostingClassifier(
        max_leaf_nodes    = model_cfg["num_leaves"],
        learning_rate     = model_cfg["learning_rate"],
        max_iter          = model_cfg["n_estimators"],
        min_samples_leaf  = model_cfg["min_child_samples"],
        class_weight      = model_cfg["class_weight"],
        random_state      = 42,
        early_stopping    = True,
        validation_fraction = 0.1,
        n_iter_no_change  = 50,
        verbose           = 0,          # quiet — we print our own summary
    )

    print("\n  Training new model…")
    model.fit(x_train, y_train, sample_weight=w_train)

    val_proba = model.predict_proba(x_val)[:, 1]
    val_auroc = float(roc_auc_score(y_val, val_proba))

    print(f"  New model validation AUROC : {val_auroc:.4f}")
    print()
    print(classification_report(
        y_val,
        (val_proba >= 0.4).astype(int),
        target_names=["No Sepsis", "Sepsis"],
        digits=3,
    ))

    return model, val_auroc, x_val.index, y_val


def evaluate_old_model(
    x_val_idx: pd.Index,
    y_val: pd.Series,
    X: pd.DataFrame,
) -> float:
    """Run the existing saved model on the same validation split to compare."""
    if not ARTIFACT_PATH.exists():
        print("  No existing model found — skipping comparison.")
        return 0.0

    old_artifact = joblib.load(ARTIFACT_PATH)
    old_model    = old_artifact["model"]
    old_cols     = old_artifact["feature_cols"]

    x_val_old = X.loc[x_val_idx, old_cols]
    old_proba = old_model.predict_proba(x_val_old)[:, 1]
    old_auroc = float(roc_auc_score(y_val, old_proba))
    print(f"  Old model validation AUROC : {old_auroc:.4f}")
    return old_auroc


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(dry_run: bool = False, force: bool = False) -> None:
    cfg = _load_config()

    # ── Step 1: Build dataset ──────────────────────────────────
    _section("Step 1 / 3  —  Build Training Set")
    X, y, weights, feature_cols = build_training_set()

    # ── Step 2: Train ──────────────────────────────────────────
    _section("Step 2 / 3  —  Train New Model")
    new_model, new_auroc, x_val_idx, y_val = train_new_model(X, y, weights, cfg)

    # ── Step 3: Compare & save ─────────────────────────────────
    _section("Step 3 / 3  —  Compare & Save")
    old_auroc = evaluate_old_model(x_val_idx, y_val, X)

    delta = new_auroc - old_auroc
    if delta >= 0:
        print(f"\n  ✅ New model is better by {delta:+.4f} AUROC")
    else:
        print(f"\n  ⚠️  New model is worse by {delta:+.4f} AUROC")

    if dry_run:
        print("\n  DRY RUN — no files written.")
        print("  Remove --dry-run to actually save the model.")
        return

    should_save = force or (delta >= 0)

    if should_save:
        # Back up old model first
        if ARTIFACT_PATH.exists():
            backup = _backup_model(ARTIFACT_PATH)
            print(f"\n  Old model backed up → {backup}")

        artifact = {
            "model":        new_model,
            "feature_cols": feature_cols,
            "auroc":        new_auroc,
        }
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, ARTIFACT_PATH)
        print(f"  New model saved      → {ARTIFACT_PATH}")

        if force and delta < 0:
            print("  (Saved via --force despite lower AUROC)")
    else:
        print(
            "\n  Model NOT saved (new AUROC is lower).\n"
            "  Use --force to save anyway."
        )

    print(f"\n{'='*55}")
    print("  Retraining complete.")
    print(f"{'='*55}\n")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain SepsisAlert model with clinician feedback."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing any files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Save the new model even if its AUROC is lower than the old one.",
    )
    args = parser.parse_args()

    if args.dry_run and args.force:
        print("Error: --dry-run and --force cannot be used together.")
        sys.exit(1)

    main(dry_run=args.dry_run, force=args.force)
