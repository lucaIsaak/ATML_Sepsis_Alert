"""
Clinical prompt templates for the narrative generator.

The LLM receives only the SHAP summary — no raw patient identifiers
or free-text notes. This keeps the prompt short and GDPR-safe.
"""

SYSTEM_PROMPT = """You are a clinical decision support assistant embedded in an ICU early warning system.
Your role is to translate an AI model's sepsis risk assessment into a clear, concise alert
that a bedside nurse can immediately act on.

Guidelines:
- Be factual and direct. No fluff.
- Explain WHY the risk is elevated using the provided lab/vital values.
- Suggest 1–2 immediate next steps (e.g., "consider lactate recheck", "notify physician").
- Never diagnose. Use language like "suggests elevated risk" not "patient has sepsis".
- Maximum 4 sentences.
- Write in plain English, avoid jargon where possible."""


def build_alert_prompt(shap_summary: str, patient_context: str = "") -> str:
    """
    Build the user-facing prompt for narrative generation.

    Args:
        shap_summary: formatted string from shap_explainer.format_for_narrative()
        patient_context: optional non-identifying context (e.g., "72yo M, post-surgical")
    """
    context_line = f"Patient context: {patient_context}\n" if patient_context else ""
    return f"""{context_line}AI sepsis model output:
{shap_summary}

Generate a brief clinical alert narrative for the bedside nurse."""


def build_explanation_prompt(shap_summary: str) -> str:
    """Prompt for a more detailed explanation (for physician view)."""
    return f"""AI sepsis model output:
{shap_summary}

Explain in 2–3 sentences why these findings suggest elevated sepsis risk,
referencing the Sepsis-3 criteria where relevant."""
