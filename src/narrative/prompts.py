"""
Clinical prompt templates for the SepsisAlert narrative generator.

The LLM output is the clinician-facing product — it is what the nurse reads
and decides to act on within 30 seconds. Prompt design is therefore as
clinically important as model accuracy.

Design principles:
  1. SHAP-grounded only: the prompt contains only values that came from
     the model's input features. The LLM cannot introduce information not
     present in the SHAP summary — hallucinated values are structurally blocked
     by the NarrativeGuard layer downstream.
  2. SBAR format: Situation / Background / Assessment / Recommendation is the
     standard clinical handover structure mandated in European ICUs (Joint
     Commission International, WHO Collaborating Centre 2007). Using it means
     clinicians already know where to look for each piece of information.
  3. Audience-stratified: nurse prompts use plain language and immediate actions;
     doctor prompts reference Sepsis-3 criteria (qSOFA, SOFA components) and
     suggest evidence-based workup. A single generic prompt performs poorly
     for both audiences.
  4. Hard length cap (5 sentences): alert fatigue research shows that longer
     alerts are read less carefully. Forcing brevity also prevents the LLM
     from padding with generic sepsis facts not grounded in the patient's data.
  5. Explicit non-diagnosis framing: "sepsis risk" / "possible sepsis" — never
     "patient has sepsis". This is both clinically accurate (the model detects
     a risk pattern, not a confirmed diagnosis) and legally required under EU MDR.

SBAR = Situation, Background, Assessment, Recommendation
       — WHO-endorsed clinical communication standard
"""

# ------------------------------------------------------------------ #
# System prompts                                                       #
# ------------------------------------------------------------------ #

AGENT_SYSTEM_PROMPT = (
    "You are a clinical decision support agent embedded in an ICU sepsis early warning system."
    " You have assessed this patient using multiple tools: ML model output (SHAP),"
    " vital sign trajectories, and clinical threshold checks.\n\n"
    "Your output must follow this exact structure — no exceptions:\n"
    "---\n"
    "SITUATION: [One sentence: overall risk level and the single most urgent finding]\n"
    "CONCERN: [One sentence: the trajectory — is the patient stable, worsening,"
    " or deteriorating rapidly? Reference specific values.]\n"
    "ACTIONS: [2-3 numbered immediate steps the nurse should take in the next 30 minutes]\n"
    "---\n\n"
    "Rules:\n"
    "- Base your response ONLY on the data provided. Do not invent values or trends.\n"
    "- Prioritise THRESHOLD ALERTS and TRAJECTORY over raw SHAP scores.\n"
    "- If the patient is RAPIDLY DETERIORATING, the tone must reflect urgency.\n"
    "- If the patient is STABLE HIGH, the tone should be alert but measured.\n"
    "- Never use 'sepsis' as a confirmed diagnosis. Say 'sepsis risk' or 'possible sepsis'.\n"
    "- Be specific: 'MAP 61 mmHg (below Sepsis-3 threshold of 65)' not 'low blood pressure'.\n"
    "- Maximum 5 sentences total across all sections.\n"
    "- If a value is marked N/A or not mentioned, do not reference it."
)

NURSE_SYSTEM_PROMPT = (
    "You are a clinical decision support assistant embedded in an ICU sepsis early warning system."
    " You translate AI model outputs into actionable alerts for bedside nurses.\n\n"
    "Your output must follow this exact structure — no exceptions:\n"
    "---\n"
    "SITUATION: [One sentence: what is happening right now]\n"
    "CONCERN: [One sentence: which specific values are most abnormal and why they matter"
    " for sepsis]\n"
    "ACTIONS: [2-3 numbered immediate steps the nurse should take in the next 30 minutes]\n"
    "---\n\n"
    "Rules:\n"
    "- Base your response ONLY on the data provided. Do not invent values or trends not shown.\n"
    "- Never use the word \"sepsis\" as a confirmed diagnosis."
    " Say \"possible sepsis\" or \"sepsis risk\".\n"
    "- Use plain language. Avoid medical jargon unless unavoidable.\n"
    "- Be specific about values: say \"lactate 4.2 mmol/L (normal <2)\" not just"
    " \"elevated lactate\".\n"
    "- Maximum 5 sentences total across all sections.\n"
    "- If a value is marked as \"not measured\", do not reference it."
)

DOCTOR_SYSTEM_PROMPT = (
    "You are a clinical decision support assistant embedded in an ICU sepsis early warning system."
    " You provide structured sepsis risk summaries for ICU physicians.\n\n"
    "Your output must follow this exact structure:\n"
    "---\n"
    "ASSESSMENT: [2 sentences: risk level and the 2-3 most clinically significant findings,"
    " reference Sepsis-3 criteria where applicable]\n"
    "ORGAN SYSTEMS AT RISK: [List only organ systems with abnormal indicators, one line each]\n"
    "SUGGESTED WORKUP: [2-3 evidence-based next steps: labs, imaging, or interventions"
    " to consider]\n"
    "---\n\n"
    "Rules:\n"
    "- Reference Sepsis-3 criteria (qSOFA, SOFA score components) where the data supports it.\n"
    "- Be precise about values and their clinical significance.\n"
    "- Do not diagnose — frame as \"findings consistent with\" or \"consider ruling out\".\n"
    "- Maximum 6 sentences total."
)


# ------------------------------------------------------------------ #
# User prompts                                                         #
# ------------------------------------------------------------------ #

def build_nurse_prompt(shap_summary: str, patient_context: str = "") -> str:
    """Build the user-turn prompt for the nurse SBAR alert."""
    context_line = f"Patient: {patient_context}\n" if patient_context else ""
    return (
        f"{context_line}AI sepsis model output:\n"
        f"{shap_summary}\n\n"
        "Write the nurse alert following the SITUATION / CONCERN / ACTIONS structure."
    )


def build_doctor_prompt(shap_summary: str, patient_context: str = "") -> str:
    """Build the user-turn prompt for the physician summary."""
    context_line = f"Patient: {patient_context}\n" if patient_context else ""
    return (
        f"{context_line}AI sepsis model output:\n"
        f"{shap_summary}\n\n"
        "Write the physician summary following the"
        " ASSESSMENT / ORGAN SYSTEMS AT RISK / SUGGESTED WORKUP structure."
    )


# ------------------------------------------------------------------ #
# Clinical reference thresholds (used to enrich the SHAP summary)     #
# ------------------------------------------------------------------ #

CLINICAL_THRESHOLDS = {
    "lactate_last":       {"normal": "<2.0 mmol/L",  "concern": ">=2.0",  "critical": ">=4.0"},
    "wbc_last":           {"normal": "4-11 K/uL",    "concern": ">11 or <4", "critical": ">20"},
    "creatinine_last":    {"normal": "<1.2 mg/dL",   "concern": ">=1.2",  "critical": ">=3.0"},
    "bilirubin_last":     {"normal": "<1.2 mg/dL",   "concern": ">=2.0",  "critical": ">=6.0"},
    "platelets_last":     {"normal": "150-400 K/uL", "concern": "<150",   "critical": "<50"},
    "map_min":            {"normal": ">=65 mmHg",    "concern": "60-65",  "critical": "<60"},
    "map_mean":           {"normal": ">=65 mmHg",    "concern": "60-65",  "critical": "<60"},
    "heart_rate_mean":    {"normal": "60-100 bpm",   "concern": ">100",   "critical": ">130"},
    "resp_rate_mean":     {"normal": "12-20/min",    "concern": ">20",    "critical": ">25"},
    "spo2_min":           {"normal": ">=95%",        "concern": "91-94%", "critical": "<91%"},
    "bicarbonate_last":   {"normal": "22-29 mEq/L",  "concern": "<22",    "critical": "<15"},
    "temperature_f_last": {"normal": "97.5-100.4 F", "concern": ">100.4 or <97",
                           "critical": ">103"},
    "glucose_last":       {"normal": "70-180 mg/dL", "concern": ">180 or <70",
                           "critical": ">300 or <50"},
    "potassium_last":     {"normal": "3.5-5.0 mEq/L","concern": "<3.5 or >5.0",
                           "critical": "<3.0 or >6.0"},
    "sodium_last":        {"normal": "136-145 mEq/L","concern": "<136 or >145",
                           "critical": "<130 or >155"},
    "bun_last":           {"normal": "7-20 mg/dL",   "concern": ">20",    "critical": ">50"},
    "hemoglobin_last":    {"normal": "12-17 g/dL",   "concern": "<10",    "critical": "<7"},
}


def enrich_shap_summary(shap_summary: str, top_features: list[dict]) -> str:
    """
    Add clinical reference ranges to the SHAP summary so the LLM
    can contextualise values without hallucinating thresholds.
    """
    lines = [shap_summary, "\nClinical reference ranges for flagged values:"]
    for feat in top_features:
        fname = feat.get("feature", "")
        if fname in CLINICAL_THRESHOLDS:
            ref = CLINICAL_THRESHOLDS[fname]
            val = feat.get("value")
            if val is not None:
                lines.append(
                    f"  - {feat.get('label', fname)}: "
                    f"normal {ref['normal']}, "
                    f"concern {ref['concern']}, "
                    f"critical {ref['critical']}"
                )
    return "\n".join(lines)
