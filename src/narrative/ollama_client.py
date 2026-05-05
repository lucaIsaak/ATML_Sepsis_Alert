"""
Narrative generation via Ollama (local LLM).

Model: mistral:7b (on-premise, GDPR-safe, zero per-call cost)

The LLM output is the ground truth the nurse acts on.
All inference is local — no patient data leaves the machine.
"""

import yaml
import requests

from src.explainability.shap_explainer import SHAPExplanation, format_for_narrative
from src.narrative.prompts import (
    DOCTOR_SYSTEM_PROMPT,
    NURSE_SYSTEM_PROMPT,
    build_doctor_prompt,
    build_nurse_prompt,
    enrich_shap_summary,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file and return as dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class OllamaClient:
    """Client for generating clinical narratives via a local Ollama LLM."""

    def __init__(self, cfg: dict | None = None):
        """Initialise client from config (loads from disk if cfg is None)."""
        if cfg is None:
            cfg = load_config()
        self.model = cfg["narrative"]["ollama_model"]
        self.base_url = cfg["narrative"]["ollama_base_url"]
        self.max_tokens = cfg["narrative"]["max_tokens"]

    def _chat(self, system: str, user: str) -> str:
        """Send a chat request to Ollama and return the model's response."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.2,   # low — clinical text must be consistent
                "top_p": 0.9,
            },
        }
        try:
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return (
                "Narrative unavailable: Ollama server not running.\n"
                "Start with: ollama serve"
            )
        except requests.exceptions.RequestException as exc:
            return f"Narrative error: {exc}"

    def generate_nurse_alert(
        self,
        explanation: SHAPExplanation,
        patient_context: str = "",
    ) -> str:
        """
        Generate a nurse-facing SBAR alert.

        The nurse reads this at the bedside and decides whether to
        escalate to the physician. It must be readable in <30 seconds.
        """
        shap_summary = format_for_narrative(explanation)
        enriched = enrich_shap_summary(shap_summary, explanation.top_features)
        user_prompt = build_nurse_prompt(enriched, patient_context)
        return self._chat(NURSE_SYSTEM_PROMPT, user_prompt)

    def generate_doctor_summary(
        self,
        explanation: SHAPExplanation,
        patient_context: str = "",
    ) -> str:
        """
        Generate a physician-facing detailed summary.

        References Sepsis-3 criteria and suggests workup steps.
        """
        shap_summary = format_for_narrative(explanation)
        enriched = enrich_shap_summary(shap_summary, explanation.top_features)
        user_prompt = build_doctor_prompt(enriched, patient_context)
        return self._chat(DOCTOR_SYSTEM_PROMPT, user_prompt)

    def generate_alert(self, explanation: SHAPExplanation, patient_context: str = "") -> str:
        """Alias for generate_nurse_alert (backward compatibility)."""
        return self.generate_nurse_alert(explanation, patient_context)

    def is_available(self) -> bool:
        """Return True if Ollama is running and the configured model is loaded."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            models = [m["name"] for m in response.json().get("models", [])]
            return any(self.model in m for m in models)
        except requests.exceptions.RequestException:
            return False
