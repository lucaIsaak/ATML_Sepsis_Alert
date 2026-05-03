"""
Narrative generation via Ollama (local LLM).

Primary model: mistral:7b
Fallback: llama3.2

All inference runs locally — no patient data leaves the machine.
This is the GDPR-compliant default for hospital deployments.
"""

import requests
import yaml
from src.narrative.prompts import SYSTEM_PROMPT, build_alert_prompt, build_explanation_prompt
from src.explainability.shap_explainer import SHAPExplanation, format_for_narrative


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class OllamaClient:
    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = load_config()
        self.model = cfg["narrative"]["ollama_model"]
        self.base_url = cfg["narrative"]["ollama_base_url"]
        self.max_tokens = cfg["narrative"]["max_tokens"]

    def _chat(self, system: str, user: str) -> str:
        """Send a chat request to the local Ollama server."""
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
                "temperature": 0.3,   # low temperature for clinical consistency
            },
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return (
                "[Narrative unavailable: Ollama server not running. "
                "Start with: ollama serve]"
            )
        except Exception as e:
            return f"[Narrative error: {e}]"

    def generate_alert(
        self,
        explanation: SHAPExplanation,
        patient_context: str = "",
    ) -> str:
        """
        Generate a nurse-facing alert narrative from a SHAP explanation.

        Args:
            explanation: SHAPExplanation from shap_explainer.explain_patient()
            patient_context: optional non-identifying context string

        Returns:
            Plain-language alert string
        """
        shap_summary = format_for_narrative(explanation)
        user_prompt = build_alert_prompt(shap_summary, patient_context)
        return self._chat(SYSTEM_PROMPT, user_prompt)

    def generate_explanation(self, explanation: SHAPExplanation) -> str:
        """Generate a more detailed physician-facing explanation."""
        shap_summary = format_for_narrative(explanation)
        user_prompt = build_explanation_prompt(shap_summary)
        return self._chat(SYSTEM_PROMPT, user_prompt)

    def is_available(self) -> bool:
        """Check if Ollama server is running and model is loaded."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            models = [m["name"] for m in response.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False
