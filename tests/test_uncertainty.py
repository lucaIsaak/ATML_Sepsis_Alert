"""
Tests for epistemic uncertainty estimation (MC perturbation).

Covers:
  - Return structure (all expected keys present)
  - LOW flag for a model whose output is stable under perturbation
  - HIGH flag for a model that is highly sensitive to feature changes
  - CI ordering (ci_lower ≤ point_estimate ≤ ci_upper, approximately)
  - n_samples parameter is respected
  - Variance is non-negative
  - Missing training_stats falls back gracefully (std=1.0)
"""

import numpy as np
import pytest

from src.model.uncertainty import estimate_uncertainty


# ------------------------------------------------------------------ #
# Mock models                                                          #
# ------------------------------------------------------------------ #

class _ConstantModel:
    """Always predicts 0.6 — variance under any perturbation is zero."""

    def predict_proba(self, df):
        n = len(df)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


class _SensitiveModel:
    """
    p = sigmoid(20 * x[0]) — highly sensitive to the first feature.

    At x[0]=0 the sigmoid slope is 20 * 0.5 * 0.5 = 5, so a perturbation
    of 0.1 (10% of std=1.0) produces ~0.5 shift in logit → measurable
    variance that reliably exceeds the HIGH threshold of 0.01.
    """

    def predict_proba(self, df):
        x = df.iloc[:, 0].values
        p = 1.0 / (1.0 + np.exp(-20.0 * x))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

FEATURE_NAMES = ["f0", "f1", "f2"]
TRAINING_STATS = {
    "f0": {"mean": 0.0, "std": 1.0},
    "f1": {"mean": 0.0, "std": 1.0},
    "f2": {"mean": 0.0, "std": 1.0},
}
FEATURE_VECTOR = np.array([0.0, 0.0, 0.0])


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestUncertaintyReturnStructure:
    """estimate_uncertainty must return a dict with all documented keys."""

    EXPECTED_KEYS = {
        "point_estimate",
        "variance",
        "std",
        "ci_lower",
        "ci_upper",
        "ci_width",
        "is_uncertain",
        "uncertainty_flag",
        "n_samples",
    }

    def test_all_keys_present(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert self.EXPECTED_KEYS == set(result.keys())

    def test_uncertainty_flag_is_valid_string(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert result["uncertainty_flag"] in ("LOW", "MODERATE", "HIGH")

    def test_is_uncertain_is_bool(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert isinstance(result["is_uncertain"], bool)


class TestLowUncertainty:
    """A constant model produces near-zero variance → LOW flag."""

    def test_low_flag(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert result["uncertainty_flag"] == "LOW"

    def test_not_uncertain(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert result["is_uncertain"] is False

    def test_variance_near_zero(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert result["variance"] < 1e-6

    def test_point_estimate_correct(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert abs(result["point_estimate"] - 0.6) < 1e-4


class TestHighUncertainty:
    """A highly sensitive model produces large variance → HIGH flag."""

    def test_high_flag(self):
        result = estimate_uncertainty(
            _SensitiveModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert result["uncertainty_flag"] == "HIGH"

    def test_is_uncertain_true(self):
        result = estimate_uncertainty(
            _SensitiveModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert result["is_uncertain"] is True

    def test_variance_above_high_threshold(self):
        result = estimate_uncertainty(
            _SensitiveModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert result["variance"] >= 0.01


class TestCIBounds:
    """Confidence interval ordering must hold."""

    def test_ci_lower_le_ci_upper(self):
        for model in [_ConstantModel(), _SensitiveModel()]:
            result = estimate_uncertainty(
                model, FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
            )
            assert result["ci_lower"] <= result["ci_upper"]

    def test_ci_width_matches_bounds(self):
        result = estimate_uncertainty(
            _SensitiveModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        expected_width = round(result["ci_upper"] - result["ci_lower"], 4)
        assert abs(result["ci_width"] - expected_width) < 1e-6

    def test_variance_non_negative(self):
        for model in [_ConstantModel(), _SensitiveModel()]:
            result = estimate_uncertainty(
                model, FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
            )
            assert result["variance"] >= 0.0


class TestNSamples:
    """n_samples parameter is respected and reflected in the return dict."""

    def test_n_samples_recorded(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS,
            n_samples=10,
        )
        assert result["n_samples"] == 10

    def test_default_n_samples(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, TRAINING_STATS
        )
        assert result["n_samples"] == 50


class TestMissingTrainingStats:
    """Features absent from training_stats get std=1.0 fallback — no crash."""

    def test_empty_stats_does_not_raise(self):
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, training_stats={}
        )
        assert "uncertainty_flag" in result

    def test_partial_stats_does_not_raise(self):
        partial = {"f0": {"mean": 0.0, "std": 0.5}}
        result = estimate_uncertainty(
            _ConstantModel(), FEATURE_VECTOR, FEATURE_NAMES, training_stats=partial
        )
        assert result["variance"] >= 0.0
