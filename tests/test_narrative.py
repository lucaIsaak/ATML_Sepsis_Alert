"""Tests for narrative prompt construction and LLM client."""
import pytest
from unittest.mock import MagicMock, patch
from src.narrative.prompts import (
    build_nurse_prompt, build_doctor_prompt,
    enrich_shap_summary, CLINICAL_THRESHOLDS
)
from src.explainability.shap_explainer import SHAPExplanation


def make_explanation(risk_score=0.75, risk_label="HIGH"):
    return SHAPExplanation(
        stay_id="test-001",
        risk_score=risk_score,
        risk_label=risk_label,
        base_value=0.1,
        top_features=[
            {
                "feature": "lactate_last",
                "label": "Lactate (last)",
                "value": 4.2,
                "unit": "mmol/L",
                "shap": 0.18,
                "direction": "increases_risk",
            },
            {
                "feature": "map_min",
                "label": "Mean Art. Pressure (min)",
                "value": 58.0,
                "unit": "mmHg",
                "shap": 0.12,
                "direction": "increases_risk",
            },
        ],
    )


class TestPromptBuilders:
    def test_nurse_prompt_contains_sbar_keywords(self):
        expl = make_explanation()
        from src.explainability.shap_explainer import format_for_narrative
        summary = format_for_narrative(expl)
        prompt = build_nurse_prompt(summary, "65yo M | MICU")
        assert "SITUATION" in prompt or "nurse" in prompt.lower() or "ACTIONS" in prompt

    def test_doctor_prompt_contains_assessment(self):
        expl = make_explanation()
        from src.explainability.shap_explainer import format_for_narrative
        summary = format_for_narrative(expl)
        prompt = build_doctor_prompt(summary)
        assert "ASSESSMENT" in prompt or "WORKUP" in prompt or "ORGAN" in prompt

    def test_patient_context_included_in_prompt(self):
        expl = make_explanation()
        from src.explainability.shap_explainer import format_for_narrative
        summary = format_for_narrative(expl)
        prompt = build_nurse_prompt(summary, patient_context="72yo F")
        assert "72yo F" in prompt


class TestEnrichSHAPSummary:
    def test_enriched_summary_contains_reference_ranges(self):
        expl = make_explanation()
        from src.explainability.shap_explainer import format_for_narrative
        summary = format_for_narrative(expl)
        enriched = enrich_shap_summary(summary, expl.top_features)
        assert "normal" in enriched.lower()
        assert "mmol/L" in enriched or "mmHg" in enriched

    def test_enriched_summary_longer_than_original(self):
        expl = make_explanation()
        from src.explainability.shap_explainer import format_for_narrative
        summary = format_for_narrative(expl)
        enriched = enrich_shap_summary(summary, expl.top_features)
        assert len(enriched) > len(summary)

    def test_all_thresholds_have_required_keys(self):
        for feature, thresholds in CLINICAL_THRESHOLDS.items():
            assert "normal" in thresholds, f"{feature} missing 'normal'"
            assert "concern" in thresholds, f"{feature} missing 'concern'"
            assert "critical" in thresholds, f"{feature} missing 'critical'"


class TestOllamaClient:
    def test_client_handles_connection_error_gracefully(self):
        from src.narrative.ollama_client import OllamaClient
        cfg = {
            "narrative": {
                "provider": "ollama",
                "ollama_model": "mistral:7b",
                "ollama_base_url": "http://localhost:9999",  # wrong port
                "max_tokens": 100,
            }
        }
        client = OllamaClient(cfg)
        expl = make_explanation()
        result = client.generate_nurse_alert(expl)
        assert "unavailable" in result.lower() or "error" in result.lower()

    def test_is_available_returns_false_when_server_down(self):
        from src.narrative.ollama_client import OllamaClient
        cfg = {
            "narrative": {
                "ollama_model": "mistral:7b",
                "ollama_base_url": "http://localhost:9999",
                "max_tokens": 100,
            }
        }
        client = OllamaClient(cfg)
        assert client.is_available() is False
