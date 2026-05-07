"""
Tests for the AI safety guardrails module.

Covers all three protection layers:
  - InputGuard  : OOD detection
  - NarrativeGuard : prohibited-pattern detection and safe fallback
  - AuditLogger    : append-only log write / read round-trip
"""

import json

from src.safety.guardrails import (
    AuditLogger, InputGuard, NarrativeGuard, NarrativeResult, OODResult,
)


# ------------------------------------------------------------------ #
# Layer 1 — InputGuard                                                #
# ------------------------------------------------------------------ #

class TestInputGuard:
    """Tests for the out-of-distribution input detector (Layer 1)."""

    def _guard(self, stats=None):
        """Return an InputGuard with optional training statistics."""
        return InputGuard(training_stats=stats)

    def test_normal_vitals_returns_normal_flag(self):
        """Physiologically normal values produce NORMAL confidence flag."""
        guard = self._guard()
        result = guard.check({
            "heart_rate_mean": 80.0,
            "map_mean": 75.0,
            "resp_rate_mean": 16.0,
            "spo2_min": 97.0,
            "lactate_last": 1.2,
        })
        assert result.confidence_flag == "NORMAL"
        assert not result.is_ood

    def test_single_extreme_value_returns_caution(self):
        """One OOD feature produces CAUTION flag."""
        guard = self._guard()
        result = guard.check({
            "heart_rate_mean": 350.0,   # above hard bound 250
        })
        assert result.confidence_flag in ("CAUTION", "LOW_CONFIDENCE")
        assert result.is_ood
        assert "heart_rate_mean" in result.outlier_features

    def test_three_extreme_values_returns_low_confidence(self):
        """Three or more OOD features trigger LOW_CONFIDENCE."""
        guard = self._guard()
        result = guard.check({
            "heart_rate_mean": 400.0,
            "map_mean": 5.0,        # below hard bound 15
            "spo2_min": 10.0,       # below hard bound 50
        })
        assert result.confidence_flag == "LOW_CONFIDENCE"
        assert result.n_outlier_features >= 3

    def test_nan_features_are_ignored(self):
        """NaN (missing) features do not trigger OOD — handled by model."""
        guard = self._guard()
        result = guard.check({"heart_rate_mean": float("nan"), "map_mean": 75.0})
        assert result.confidence_flag == "NORMAL"

    def test_z_score_detection_with_training_stats(self):
        """Extreme z-score (>3.5σ) triggers OOD when training stats provided."""
        stats = {"lactate_last": {"mean": 1.5, "std": 0.5}}
        guard = self._guard(stats=stats)
        # lactate = 10.0 → z = (10 - 1.5) / 0.5 = 17 → OOD
        result = guard.check({"lactate_last": 10.0})
        assert result.is_ood
        assert "lactate_last" in result.outlier_features

    def test_z_score_within_bounds_is_normal(self):
        """Value within 3σ of training mean is not flagged as OOD."""
        stats = {"lactate_last": {"mean": 1.5, "std": 0.5}}
        guard = self._guard(stats=stats)
        # lactate = 2.0 → z = 1.0 → within 3σ
        result = guard.check({"lactate_last": 2.0})
        assert not result.is_ood

    def test_from_artifact_without_stats(self):
        """Artifact without training_stats falls back to hard bounds silently."""
        artifact = {"model": None, "feature_cols": []}
        guard = InputGuard.from_artifact(artifact)
        assert isinstance(guard, InputGuard)

    def test_from_artifact_with_stats(self):
        """Artifact with training_stats initialises z-score detection."""
        artifact = {
            "model": None,
            "feature_cols": [],
            "training_stats": {"heart_rate_mean": {"mean": 80.0, "std": 15.0}},
        }
        guard = InputGuard.from_artifact(artifact)
        result = guard.check({"heart_rate_mean": 80.0})
        assert result.confidence_flag == "NORMAL"


# ------------------------------------------------------------------ #
# Layer 2 — NarrativeGuard                                            #
# ------------------------------------------------------------------ #

class TestNarrativeGuard:
    """Tests for the LLM narrative validator (Layer 2)."""

    def _guard(self):
        """Return a NarrativeGuard instance."""
        return NarrativeGuard()

    def test_clean_narrative_passes_through(self):
        """Safe narrative is returned unchanged."""
        guard = self._guard()
        clean = (
            "SITUATION: Patient shows possible sepsis risk.\n"
            "CONCERN: Elevated lactate 4.2 mmol/L (normal <2).\n"
            "ACTIONS:\n1. Reassess at bedside.\n2. Notify physician."
        )
        result = guard.validate(clean, shap_summary="lactate 4.2")
        assert not result.was_replaced
        assert result.text == clean
        assert not result.is_fallback

    def test_confirmed_diagnosis_triggers_replacement(self):
        """Narratives containing confirmed diagnosis must be replaced."""
        guard = self._guard()
        bad = "Patient has sepsis. Start antibiotics immediately."
        result = guard.validate(bad, shap_summary="lactate 4.2")
        assert result.was_replaced
        assert result.is_fallback
        assert len(result.violations_found) >= 1

    def test_diagnosed_with_triggers_replacement(self):
        """'Diagnosed with sepsis' phrase triggers replacement."""
        guard = self._guard()
        bad = "The patient has been diagnosed with sepsis based on elevated lactate."
        result = guard.validate(bad, shap_summary="")
        assert result.was_replaced

    def test_treatment_order_triggers_replacement(self):
        """Definitive treatment orders must be rejected."""
        guard = self._guard()
        bad = "Start IV antibiotics within 30 minutes."
        result = guard.validate(bad, shap_summary="")
        assert result.was_replaced

    def test_fallback_contains_no_diagnosis(self):
        """Safe fallback must not contain confirmed diagnosis language."""
        guard = self._guard()
        result = guard.validate("diagnosed with sepsis", shap_summary="lactate 4.2")
        assert "confirmed" not in result.text.lower()
        assert "diagnosed with" not in result.text.lower()

    def test_fallback_includes_shap_summary(self):
        """Fallback must include SHAP summary so clinician sees the evidence."""
        guard = self._guard()
        shap = "lactate 4.2 mmol/L"
        result = guard.validate("confirmed sepsis", shap_summary=shap)
        assert shap in result.text

    def test_empty_narrative_is_safe(self):
        """Empty narrative contains no prohibited patterns — should pass."""
        guard = self._guard()
        result = guard.validate("", shap_summary="hr=110")
        assert not result.was_replaced


# ------------------------------------------------------------------ #
# Layer 3 — AuditLogger                                               #
# ------------------------------------------------------------------ #

class TestAuditLogger:
    """Tests for the append-only audit logger (Layer 3)."""

    def _logger(self, tmp_path):
        """Create a logger writing to a temp path."""
        return AuditLogger(log_path=tmp_path / "audit.jsonl")

    def _make_ood(self, flag="NORMAL"):
        """Build an OODResult with the given confidence flag."""
        return OODResult(
            is_ood=(flag != "NORMAL"),
            n_outlier_features=0,
            outlier_features=[],
            confidence_flag=flag,
        )

    def _make_nar(self, replaced=False):
        """Build a NarrativeResult with optional replacement flag."""
        return NarrativeResult(
            text="SITUATION: test",
            was_replaced=replaced,
            violations_found=[],
            is_fallback=replaced,
        )

    def test_log_creates_file(self, tmp_path):
        """Logger creates the JSONL file on first write."""
        logger = self._logger(tmp_path)
        logger.log_alert("stay-1", 0.72, "DOCTOR", [], self._make_ood(), self._make_nar())
        assert (tmp_path / "audit.jsonl").exists()

    def test_log_is_valid_json(self, tmp_path):
        """Each log line is valid JSON with required fields."""
        logger = self._logger(tmp_path)
        logger.log_alert("stay-1", 0.72, "DOCTOR", [], self._make_ood(), self._make_nar())
        lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["stay_id"] == "stay-1"
        assert record["risk_score"] == 0.72

    def test_log_is_append_only(self, tmp_path):
        """Multiple calls append new lines without overwriting."""
        logger = self._logger(tmp_path)
        logger.log_alert("stay-1", 0.5, "NURSE", [], self._make_ood(), self._make_nar())
        logger.log_alert("stay-2", 0.8, "CRITICAL", [], self._make_ood("LOW_CONFIDENCE"),
                         self._make_nar(replaced=True))
        lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_read_recent_returns_list(self, tmp_path):
        """read_recent returns the last n records in chronological order."""
        logger = self._logger(tmp_path)
        for i in range(5):
            logger.log_alert(f"stay-{i}", 0.5 + i * 0.05, "NURSE",
                             [], self._make_ood(), self._make_nar())
        records = logger.read_recent(n=3)
        assert len(records) == 3
        assert records[-1]["stay_id"] == "stay-4"

    def test_read_recent_empty_log(self, tmp_path):
        """read_recent returns empty list when no log file exists."""
        logger = self._logger(tmp_path)
        records = logger.read_recent()
        assert not records

    def test_ood_flag_recorded(self, tmp_path):
        """OOD confidence flag is persisted in the audit record."""
        logger = self._logger(tmp_path)
        logger.log_alert(
            "stay-ood", 0.65, "DOCTOR", [],
            self._make_ood("LOW_CONFIDENCE"),
            self._make_nar(),
        )
        record = logger.read_recent(1)[0]
        assert record["ood_flag"] == "LOW_CONFIDENCE"

    def test_narrative_replacement_recorded(self, tmp_path):
        """Narrative replacement flag is persisted for audit review."""
        logger = self._logger(tmp_path)
        logger.log_alert(
            "stay-bad", 0.75, "DOCTOR", [],
            self._make_ood(),
            self._make_nar(replaced=True),
        )
        record = logger.read_recent(1)[0]
        assert record["narrative_was_replaced"] is True
