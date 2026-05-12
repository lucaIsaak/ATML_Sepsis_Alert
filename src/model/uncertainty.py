"""
Epistemic uncertainty estimation for the SepsisAlert model.

WHY THIS EXISTS
===============
A HistGradientBoostingClassifier outputs a single probability estimate with no
attached confidence.  When the model is asked about a patient whose feature
*combination* it has rarely seen during training, it produces an authoritative-
looking number — and has no way to signal "I don't really know."

This module closes that gap by estimating *local sensitivity*: how much does the
prediction change when we add small, realistic noise to the feature values?

  High variance  → model is near many decision boundaries simultaneously
                   → operating in a sparse region of training-feature space
                   → epistemic uncertainty is high → flag for human review

  Low variance   → model output is stable across realistic perturbations
                   → prediction is reliable

APPROACH: Monte Carlo Feature Perturbation
==========================================
For each patient we draw N perturbed copies of the feature vector:

    x' ~ N(x, (σ_train × α)²)

where σ_train is the per-feature training standard deviation and α=0.10
(10%).  We run predict_proba on all N copies at once (vectorised — one model
call) and report the variance, 5th/95th percentile interval, and a flag.

This is a model-agnostic proxy: no retraining or architecture changes required.
It is not theoretically identical to Bayesian epistemic uncertainty (which would
require an ensemble or conformal prediction), but it is well-calibrated for
gradient-boosted trees because perturbations that cross a split boundary produce
a measurable output change, while perturbations within the same leaf cancel out.

THRESHOLDS
==========
  LOW      variance < 0.003   — prediction is stable; use risk score normally
  MODERATE 0.003 ≤ var < 0.01 — some sensitivity; note uncertainty in narrative
  HIGH     var ≥ 0.01         — near many boundaries; flag for human review
                                 regardless of the point estimate's magnitude

These thresholds are conservative defaults derived from the expected variance
of a well-calibrated GBDT under 10% Gaussian noise (α=0.10).  They should be
validated against your deployment data and tightened as labelled feedback
accumulates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_N_SAMPLES             = 50     # perturbations per patient (vectorised — fast)
_PERTURBATION_FRACTION = 0.10   # 10 % of training std per feature
_VAR_MODERATE          = 0.003
_VAR_HIGH              = 0.010


def estimate_uncertainty(
    model,
    feature_vector: np.ndarray,
    feature_names: list[str],
    training_stats: dict,
    n_samples: int = _N_SAMPLES,
    seed: int = 42,
) -> dict:
    """
    Estimate epistemic uncertainty for one patient via MC perturbation.

    Parameters
    ----------
    model          : fitted sklearn classifier (supports predict_proba)
    feature_vector : 1-D float array aligned to feature_names
    feature_names  : list of feature column names
    training_stats : {feature_name: {"mean": float, "std": float}} from artifact
    n_samples      : number of Monte Carlo perturbations
    seed           : RNG seed for reproducibility across calls

    Returns
    -------
    dict
        point_estimate   – original unperturbed risk score (0–1)
        variance         – prediction variance across perturbed samples
        std              – standard deviation (√variance)
        ci_lower         – 5th percentile of perturbed predictions
        ci_upper         – 95th percentile of perturbed predictions
        ci_width         – ci_upper − ci_lower (convenient summary)
        is_uncertain     – True if variance ≥ _VAR_MODERATE
        uncertainty_flag – "LOW" | "MODERATE" | "HIGH"
        n_samples        – number of perturbations used
    """
    # ── Point estimate ─────────────────────────────────────────────────
    feat_df = pd.DataFrame([feature_vector], columns=feature_names)
    point_estimate = float(model.predict_proba(feat_df)[0, 1])

    # ── Perturbation scales ────────────────────────────────────────────
    # Each feature is perturbed by N(0, (α × σ_train)²).
    # Features absent from training_stats get a 1.0 std fallback.
    scales = np.array(
        [training_stats.get(f, {}).get("std", 1.0) * _PERTURBATION_FRACTION
         for f in feature_names],
        dtype=float,
    )
    scales = np.where(scales < 1e-8, 1e-8, scales)   # guard against zero

    # ── Vectorised MC evaluation ───────────────────────────────────────
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scales, size=(n_samples, len(feature_names)))
    X_perturbed = feature_vector + noise             # (n_samples, n_features)
    perturbed_df = pd.DataFrame(X_perturbed, columns=feature_names)
    perturbed_preds = model.predict_proba(perturbed_df)[:, 1]  # one call

    # ── Aggregate statistics ───────────────────────────────────────────
    variance = float(np.var(perturbed_preds))
    std      = float(np.std(perturbed_preds))
    ci_lower = float(np.percentile(perturbed_preds, 5))
    ci_upper = float(np.percentile(perturbed_preds, 95))

    if variance >= _VAR_HIGH:
        uncertainty_flag = "HIGH"
        is_uncertain = True
    elif variance >= _VAR_MODERATE:
        uncertainty_flag = "MODERATE"
        is_uncertain = True
    else:
        uncertainty_flag = "LOW"
        is_uncertain = False

    return {
        "point_estimate":   round(point_estimate, 4),
        "variance":         round(variance, 6),
        "std":              round(std, 4),
        "ci_lower":         round(ci_lower, 4),
        "ci_upper":         round(ci_upper, 4),
        "ci_width":         round(ci_upper - ci_lower, 4),
        "is_uncertain":     is_uncertain,
        "uncertainty_flag": uncertainty_flag,
        "n_samples":        n_samples,
    }
