"""
Narrative generation via Anthropic Claude API (cloud option).

When to use this instead of Ollama
===================================
On-premise Ollama (mistral:7b) is the default and is always preferred for
GDPR compliance — patient feature data never leaves the hospital network.

The Claude backend is provided for deployments where:
  1. The hospital has signed a Data Processing Agreement (DPA) with Anthropic
     (required under GDPR Art. 28 before any patient-adjacent data can be
     processed by a third-party cloud service)
  2. The hospital's Data Protection Officer (DPO) has approved the transfer
  3. The hospital cannot or does not want to run an on-premise GPU node

Switch by setting in config.yaml:
    narrative:
      provider: "claude"
      claude_model: "claude-haiku-4-5-20251001"   # fast + cheap
      gdpr_cloud_dpa_acknowledged: true            # DPA must be in place

IMPORTANT: Only de-identified SHAP feature summaries are sent to Claude
(e.g. "Lactate = 4.2, WBC = 18.1"). Patient stay_id is NOT included in
the prompt. However, combinations of clinical values from a small ICU
can still be quasi-identifiable — a signed DPA with Anthropic is
mandatory before enabling this backend.

Cost comparison (vs Ollama on-prem):
  claude-haiku-4-5:  ~€0.0008 / narrative (input + output)
  claude-sonnet-4-6: ~€0.006  / narrative
  Ollama on-prem:    ~€0 / narrative (after GPU amortisation)

At 200-bed ICU with hourly re-scoring, Claude Haiku costs ~€1.40/day.
This is well under 5% of revenue at Standard tier (€275/bed/month).
"""

from __future__ import annotations

import os


# Available Claude models for the model selector dropdown
CLAUDE_MODELS = [
    "claude-haiku-4-5-20251001",   # fastest, cheapest — recommended for production
    "claude-sonnet-4-6",           # higher quality, ~7x more expensive
]


class ClaudeClient:
    """
    Drop-in replacement for OllamaClient using the Anthropic Claude API.

    Implements the same _chat / _stream_chat / is_available interface so
    NarrativeAgent works unchanged with either backend.
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            import yaml  # pylint: disable=import-outside-toplevel
            with open("config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

        narrative_cfg = cfg.get("narrative", {})
        self.model     = narrative_cfg.get("claude_model", CLAUDE_MODELS[0])
        self.max_tokens = narrative_cfg.get("max_tokens", 512)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it in your .env file or environment before using the Claude backend."
            )

        try:
            import anthropic  # pylint: disable=import-outside-toplevel
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

    def _chat(self, system: str, user: str) -> str:
        """Send a synchronous chat request to Claude and return the response."""
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text.strip()
        except Exception as exc:  # pylint: disable=broad-except
            return f"[Claude narrative error] {exc}"

    def _stream_chat(self, system: str, user: str):
        """Stream a chat response from Claude, yielding text chunks."""
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:  # pylint: disable=broad-except
            yield f"\n\n*Claude narrative error: {exc}*"

    def is_available(self) -> bool:
        """Return True if the Anthropic API key is set and the client is initialised."""
        return self._client is not None

    # ------------------------------------------------------------------ #
    # Compatibility shims — same signatures as OllamaClient               #
    # ------------------------------------------------------------------ #

    def generate_nurse_alert(self, explanation, patient_context: str = "") -> str:
        from src.narrative.prompts import NURSE_SYSTEM_PROMPT, build_nurse_prompt, enrich_shap_summary  # noqa: PLC0415
        from src.explainability.shap_explainer import format_for_narrative  # noqa: PLC0415
        shap_summary = format_for_narrative(explanation)
        enriched = enrich_shap_summary(shap_summary, explanation.top_features)
        return self._chat(NURSE_SYSTEM_PROMPT, build_nurse_prompt(enriched, patient_context))

    def stream_alert(self, explanation, patient_context: str = ""):
        from src.narrative.prompts import NURSE_SYSTEM_PROMPT, build_nurse_prompt, enrich_shap_summary  # noqa: PLC0415
        from src.explainability.shap_explainer import format_for_narrative  # noqa: PLC0415
        shap_summary = format_for_narrative(explanation)
        enriched = enrich_shap_summary(shap_summary, explanation.top_features)
        yield from self._stream_chat(NURSE_SYSTEM_PROMPT, build_nurse_prompt(enriched, patient_context))
