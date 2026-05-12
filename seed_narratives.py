"""
SepsisAlert — Seed Narrative Feedback Store.

Seeds logs/narrative_feedback.jsonl with clinically plausible, high-quality
SBAR examples so the RAG few-shot system works from first use.

Without seeded examples the few-shot context is always empty on a fresh
install, making narrative quality dependent entirely on zero-shot prompting.
These examples cover the main clinical presentation archetypes:
  - Septic shock (HIGH risk, elevated lactate + haemodynamic instability)
  - Early sepsis (MODERATE risk, fever + rising WBC)
  - Respiratory sepsis (HIGH risk, low SpO2 + high respiratory rate)
  - Renal sepsis (MODERATE risk, rising creatinine + oliguria)
  - Post-op sepsis (HIGH risk, rapid deterioration)
  - Near-miss (MODERATE risk, borderline but deteriorating)

Usage:
    python seed_narratives.py              # seed if log is empty
    python seed_narratives.py --force      # overwrite existing examples
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path("logs/narrative_feedback.jsonl")

_SEED_EXAMPLES = [
    # ── 1. Septic shock — HIGH risk ──────────────────────────────────
    {
        "stay_id": "seed_001",
        "model_used": "mistral:7b",
        "rating": 5,
        "correction_note": "",
        "shap_summary": "Risk score: 0.87 (HIGH)\nKey drivers:\n  - Lactate (last) = 5.2 mmol/L  [+0.312 ↑ risk]\n  - Mean Art. Pressure (min) = 48 mmHg  [+0.287 ↑ risk]\n  - Heart Rate (mean) = 128 bpm  [+0.198 ↑ risk]\n  - Creatinine (last) = 2.8 mg/dL  [+0.143 ↑ risk]\n  - Respiratory Rate (mean) = 28 breaths/min  [+0.112 ↑ risk]",
        "shap_vector": {
            "Lactate (last)": 0.312,
            "Mean Art. Pressure (min)": 0.287,
            "Heart Rate (mean)": 0.198,
            "Creatinine (last)": 0.143,
            "Respiratory Rate (mean)": 0.112,
            "SpO2 (min)": -0.031,
            "WBC (last)": 0.087,
        },
        "narrative_text": (
            "SITUATION: Patient in MICU shows HIGH sepsis risk (score 0.87). "
            "Haemodynamic compromise is the dominant concern.\n\n"
            "BACKGROUND: Lactate has risen to 5.2 mmol/L (normal <2.0) with MAP "
            "dropping to 48 mmHg — below the 65 mmHg threshold for septic shock. "
            "Heart rate is 128 bpm and respiratory rate 28 breaths/min, consistent "
            "with a high metabolic demand state. Creatinine 2.8 mg/dL suggests acute "
            "kidney injury, likely from hypoperfusion.\n\n"
            "ASSESSMENT: Feature pattern is consistent with early septic shock. "
            "Elevated lactate combined with refractory hypotension is the primary "
            "driver of the alert. This combination carries an independently elevated "
            "30-day mortality risk.\n\n"
            "RECOMMENDATION: Notify attending physician immediately. "
            "Consider fluid resuscitation, vasopressor review, and blood cultures "
            "before antibiotic administration. Monitor urine output closely.\n\n"
            "NOTE: This is AI decision support — not a diagnosis. "
            "Clinical assessment by a physician is required before any intervention."
        ),
    },

    # ── 2. Early sepsis — MODERATE risk ──────────────────────────────
    {
        "stay_id": "seed_002",
        "model_used": "mistral:7b",
        "rating": 5,
        "correction_note": "",
        "shap_summary": "Risk score: 0.52 (MODERATE)\nKey drivers:\n  - WBC (last) = 18.4 K/µL  [+0.178 ↑ risk]\n  - Temperature (last) = 101.8 °F  [+0.142 ↑ risk]\n  - Heart Rate (trend) = +8.2 bpm/h  [+0.118 ↑ risk]\n  - Lactate (mean) = 2.3 mmol/L  [+0.091 ↑ risk]\n  - Respiratory Rate (mean) = 22 breaths/min  [+0.068 ↑ risk]",
        "shap_vector": {
            "WBC (last)": 0.178,
            "Temperature (last)": 0.142,
            "Heart Rate (trend)": 0.118,
            "Lactate (mean)": 0.091,
            "Respiratory Rate (mean)": 0.068,
            "Mean Art. Pressure (mean)": -0.022,
        },
        "narrative_text": (
            "SITUATION: Patient shows MODERATE sepsis risk (score 0.52) with an "
            "evolving infectious picture.\n\n"
            "BACKGROUND: WBC is elevated at 18.4 K/µL with fever of 101.8 °F. "
            "Heart rate is trending upward at +8.2 bpm/hour over the past 6 hours — "
            "a pattern of gradual deterioration rather than acute decompensation. "
            "Lactate is borderline at 2.3 mmol/L (normal <2.0). "
            "MAP remains preserved at this time.\n\n"
            "ASSESSMENT: Presentation is consistent with systemic inflammatory "
            "response, possibly early-stage sepsis. The rising heart rate trend is "
            "the key early warning signal — the patient may still be compensating.\n\n"
            "RECOMMENDATION: Reassess at bedside. Review recent cultures and "
            "antibiotic coverage. Increase monitoring frequency and recheck lactate "
            "in 2 hours to determine if the trend continues.\n\n"
            "NOTE: This is AI decision support — not a diagnosis. "
            "Clinical assessment by a nurse and physician is required."
        ),
    },

    # ── 3. Respiratory sepsis — HIGH risk ────────────────────────────
    {
        "stay_id": "seed_003",
        "model_used": "mistral:7b",
        "rating": 5,
        "correction_note": "",
        "shap_summary": "Risk score: 0.74 (HIGH)\nKey drivers:\n  - SpO2 (min) = 84%  [+0.261 ↑ risk]\n  - Respiratory Rate (mean) = 32 breaths/min  [+0.233 ↑ risk]\n  - Respiratory Rate (trend) = +4.1 breaths/h  [+0.189 ↑ risk]\n  - WBC (last) = 22.1 K/µL  [+0.134 ↑ risk]\n  - Temperature (mean) = 102.4 °F  [+0.099 ↑ risk]",
        "shap_vector": {
            "SpO2 (min)": 0.261,
            "Respiratory Rate (mean)": 0.233,
            "Respiratory Rate (trend)": 0.189,
            "WBC (last)": 0.134,
            "Temperature (mean)": 0.099,
            "Lactate (last)": 0.072,
        },
        "narrative_text": (
            "SITUATION: Patient shows HIGH sepsis risk (score 0.74) driven by "
            "severe respiratory compromise.\n\n"
            "BACKGROUND: SpO2 has dropped to a minimum of 84% with respiratory "
            "rate at 32 breaths/min, trending upward at +4.1 breaths/hour. "
            "WBC is markedly elevated at 22.1 K/µL with persistent fever of 102.4 °F. "
            "This combination suggests a pulmonary source (pneumonia, ARDS) with "
            "systemic sepsis response.\n\n"
            "ASSESSMENT: Worsening hypoxia and escalating respiratory effort are "
            "the primary drivers. The rate of change in respiratory rate is "
            "particularly concerning — this patient may decompensate rapidly.\n\n"
            "RECOMMENDATION: Notify attending physician urgently. "
            "Assess for need for escalation to non-invasive or invasive ventilation. "
            "Obtain chest imaging and review oxygenation parameters. "
            "Blood cultures and respiratory cultures if not already collected.\n\n"
            "NOTE: This is AI decision support — not a diagnosis."
        ),
    },

    # ── 4. Renal / metabolic sepsis — MODERATE risk ──────────────────
    {
        "stay_id": "seed_004",
        "model_used": "mistral:7b",
        "rating": 5,
        "correction_note": "",
        "shap_summary": "Risk score: 0.48 (MODERATE)\nKey drivers:\n  - Creatinine (last) = 3.4 mg/dL  [+0.201 ↑ risk]\n  - Creatinine (trend) = +0.6 mg/dL/h  [+0.167 ↑ risk]\n  - Bicarbonate (last) = 16 mEq/L  [+0.143 ↑ risk]\n  - Lactate (delta) = +1.8 mmol/L  [+0.128 ↑ risk]\n  - Platelets (last) = 87 K/µL  [+0.092 ↑ risk]",
        "shap_vector": {
            "Creatinine (last)": 0.201,
            "Creatinine (trend)": 0.167,
            "Bicarbonate (last)": 0.143,
            "Lactate (change)": 0.128,
            "Platelets (last)": 0.092,
            "Heart Rate (mean)": 0.044,
        },
        "narrative_text": (
            "SITUATION: Patient shows MODERATE sepsis risk (score 0.48) with "
            "a pattern of progressive end-organ dysfunction.\n\n"
            "BACKGROUND: Creatinine has risen to 3.4 mg/dL and is trending upward "
            "at +0.6 mg/dL per hour, consistent with acute kidney injury. "
            "Bicarbonate is low at 16 mEq/L, indicating metabolic acidosis. "
            "Lactate has increased by 1.8 mmol/L since last measurement, and "
            "platelets are falling (87 K/µL) — a pattern associated with early "
            "DIC in septic patients. Haemodynamics remain borderline.\n\n"
            "ASSESSMENT: Multi-organ signal is more concerning than the overall "
            "risk score suggests. The combination of worsening renal function, "
            "acidosis, and thrombocytopenia is consistent with Sepsis-3 organ "
            "dysfunction criteria.\n\n"
            "RECOMMENDATION: Notify physician. Review fluid balance and nephrology "
            "input. Recheck coagulation panel. Monitor lactate closely.\n\n"
            "NOTE: This is AI decision support — not a diagnosis."
        ),
    },

    # ── 5. Post-op sepsis — HIGH risk, rapidly deteriorating ─────────
    {
        "stay_id": "seed_005",
        "model_used": "mistral:7b",
        "rating": 5,
        "correction_note": "",
        "shap_summary": "Risk score: 0.81 (HIGH)\nKey drivers:\n  - Heart Rate (trend) = +22 bpm/h  [+0.298 ↑ risk]\n  - Lactate (last) = 4.7 mmol/L  [+0.271 ↑ risk]\n  - Mean Art. Pressure (trend) = -6.4 mmHg/h  [+0.244 ↑ risk]\n  - Temperature (last) = 103.1 °F  [+0.187 ↑ risk]\n  - WBC (delta) = +6.2 K/µL  [+0.143 ↑ risk]",
        "shap_vector": {
            "Heart Rate (trend)": 0.298,
            "Lactate (last)": 0.271,
            "Mean Art. Pressure (trend)": 0.244,
            "Temperature (last)": 0.187,
            "WBC (change)": 0.143,
            "Creatinine (last)": 0.089,
        },
        "narrative_text": (
            "SITUATION: Patient is at HIGH sepsis risk (score 0.81) with rapid "
            "haemodynamic deterioration — CRITICAL escalation may be required.\n\n"
            "BACKGROUND: Heart rate is escalating at +22 bpm/hour and MAP is "
            "falling at -6.4 mmHg/hour simultaneously — a convergent pattern of "
            "cardiovascular decompensation. Lactate is 4.7 mmol/L (severe). "
            "Fever of 103.1 °F with a WBC rise of +6.2 K/µL since last check "
            "indicates an active and worsening infectious process.\n\n"
            "ASSESSMENT: The rate of change across multiple parameters is the "
            "primary concern. This patient is on a decompensation trajectory. "
            "Lactate above 4 mmol/L with falling MAP meets criteria for septic "
            "shock requiring urgent intervention.\n\n"
            "RECOMMENDATION: Immediate physician notification required. "
            "Initiate septic shock protocol. Secure IV access, begin fluid "
            "resuscitation, and prepare vasopressors if MAP drops below 65 mmHg. "
            "Blood cultures before antibiotics. ICU attending at bedside.\n\n"
            "NOTE: This is AI decision support — not a diagnosis. "
            "Physician assessment is required before any intervention."
        ),
    },

    # ── 6. Near-miss — MODERATE but deteriorating ────────────────────
    {
        "stay_id": "seed_006",
        "model_used": "mistral:7b",
        "rating": 5,
        "correction_note": "",
        "shap_summary": "Risk score: 0.42 (MODERATE)\nKey drivers:\n  - Lactate (trend) = +0.4 mmol/L/h  [+0.156 ↑ risk]\n  - Heart Rate (trend) = +5.8 bpm/h  [+0.122 ↑ risk]\n  - Respiratory Rate (last) = 23 breaths/min  [+0.098 ↑ risk]\n  - Temperature (last) = 100.8 °F  [+0.071 ↑ risk]\n  - SpO2 (trend) = -0.8 %/h  [+0.065 ↑ risk]",
        "shap_vector": {
            "Lactate (trend)": 0.156,
            "Heart Rate (trend)": 0.122,
            "Respiratory Rate (last)": 0.098,
            "Temperature (last)": 0.071,
            "SpO2 (trend)": 0.065,
            "Mean Art. Pressure (mean)": -0.018,
        },
        "narrative_text": (
            "SITUATION: Patient is at MODERATE sepsis risk (score 0.42) but the "
            "trajectory of multiple markers is more concerning than the current "
            "score suggests.\n\n"
            "BACKGROUND: Lactate is rising at +0.4 mmol/L per hour, heart rate "
            "is trending upward at +5.8 bpm/hour, and SpO2 is drifting downward "
            "at -0.8% per hour. No single value has yet crossed a critical "
            "threshold, but the concurrent directional change across lactate, "
            "heart rate, respiratory rate, and oxygenation is an early warning "
            "pattern. Low-grade fever (100.8 °F) supports an infectious process.\n\n"
            "ASSESSMENT: This patient is in a compensated but worsening state. "
            "If current trends continue for another 2–3 hours, risk will likely "
            "cross into HIGH territory. Early intervention now is preferable to "
            "a reactive response later.\n\n"
            "RECOMMENDATION: Bedside nursing assessment. Increase vital sign "
            "monitoring frequency. Notify physician if trends continue. "
            "Review recent cultures and consider early antibiotic review.\n\n"
            "NOTE: This is AI decision support — not a diagnosis."
        ),
    },

    # ── 7. Elderly patient — HIGH risk, atypical presentation ────────
    {
        "stay_id": "seed_007",
        "model_used": "mistral:7b",
        "rating": 4,
        "correction_note": "Good narrative — note that elderly patients may not mount fever; the low temperature here is also abnormal.",
        "shap_summary": "Risk score: 0.69 (HIGH)\nKey drivers:\n  - Lactate (last) = 3.8 mmol/L  [+0.241 ↑ risk]\n  - Mean Art. Pressure (min) = 58 mmHg  [+0.198 ↑ risk]\n  - Age = 81  [+0.162 ↑ risk]\n  - WBC (last) = 3.1 K/µL  [+0.134 ↑ risk]\n  - Temperature (last) = 97.1 °F  [-0.041 ↑ risk]",
        "shap_vector": {
            "Lactate (last)": 0.241,
            "Mean Art. Pressure (min)": 0.198,
            "Age": 0.162,
            "WBC (last)": 0.134,
            "Temperature (last)": -0.041,
            "Heart Rate (mean)": 0.088,
        },
        "narrative_text": (
            "SITUATION: Elderly patient (81 years) shows HIGH sepsis risk (score 0.69) "
            "with an atypical presentation.\n\n"
            "BACKGROUND: Lactate is elevated at 3.8 mmol/L with MAP dropping to "
            "58 mmHg — concerning for early haemodynamic compromise. "
            "Notably, WBC is low at 3.1 K/µL (leucopenia) and temperature is "
            "97.1 °F — both can indicate a blunted immune response in elderly patients, "
            "which paradoxically correlates with higher mortality in sepsis. "
            "The absence of fever does not exclude severe infection in this age group.\n\n"
            "ASSESSMENT: Elevated lactate and low MAP are the primary risk drivers. "
            "Leucopenia in an older patient with haemodynamic instability warrants "
            "high clinical suspicion for Gram-negative or atypical sepsis.\n\n"
            "RECOMMENDATION: Notify physician. Obtain blood cultures (consider "
            "anaerobic bottles). Review antibiotic coverage appropriate for "
            "immunosenescence. Monitor MAP and consider early vasopressor "
            "readiness.\n\n"
            "NOTE: This is AI decision support — not a diagnosis."
        ),
    },

    # ── 8. Abdominal / surgical sepsis — MODERATE risk ───────────────
    {
        "stay_id": "seed_008",
        "model_used": "mistral:7b",
        "rating": 5,
        "correction_note": "",
        "shap_summary": "Risk score: 0.56 (MODERATE)\nKey drivers:\n  - Bilirubin (last) = 4.2 mg/dL  [+0.189 ↑ risk]\n  - WBC (last) = 20.7 K/µL  [+0.176 ↑ risk]\n  - Lactate (mean) = 2.6 mmol/L  [+0.141 ↑ risk]\n  - Temperature (mean) = 101.2 °F  [+0.112 ↑ risk]\n  - Heart Rate (mean) = 112 bpm  [+0.087 ↑ risk]",
        "shap_vector": {
            "Bilirubin (last)": 0.189,
            "WBC (last)": 0.176,
            "Lactate (mean)": 0.141,
            "Temperature (mean)": 0.112,
            "Heart Rate (mean)": 0.087,
            "Creatinine (last)": 0.053,
        },
        "narrative_text": (
            "SITUATION: Patient shows MODERATE sepsis risk (score 0.56) with "
            "a hepatic/abdominal pattern of organ stress.\n\n"
            "BACKGROUND: Bilirubin is elevated at 4.2 mg/dL, consistent with "
            "hepatic involvement or biliary obstruction in the context of sepsis. "
            "WBC is 20.7 K/µL with fever averaging 101.2 °F, and lactate is "
            "borderline at 2.6 mmol/L. Tachycardia at 112 bpm. "
            "This combination raises concern for an abdominal source "
            "(biliary, hepatic, or surgical) driving the systemic response.\n\n"
            "ASSESSMENT: Elevated bilirubin as the top SHAP feature is unusual "
            "in non-abdominal presentations — consider cholangitis, hepatic "
            "abscess, or post-operative abdominal complication as the source.\n\n"
            "RECOMMENDATION: Notify physician. Review recent abdominal imaging "
            "and surgical history. Consider gastroenterology or surgical input. "
            "Blood cultures and hepatic function panel.\n\n"
            "NOTE: This is AI decision support — not a diagnosis."
        ),
    },
]


def seed(force: bool = False) -> None:
    """Write seed examples to the narrative feedback log."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _LOG_PATH.exists() and not force:
        existing = sum(1 for _ in _LOG_PATH.open())
        if existing > 0:
            print(
                f"  {_LOG_PATH} already contains {existing} records. "
                "Use --force to overwrite. Skipping."
            )
            return

    ts = datetime.now(timezone.utc).isoformat()
    written = 0

    with _LOG_PATH.open("w", encoding="utf-8") as f:
        for ex in _SEED_EXAMPLES:
            record = {
                "stay_id":         ex["stay_id"],
                "model_used":      ex["model_used"],
                "rating":          ex["rating"],
                "correction_note": ex.get("correction_note", ""),
                "shap_summary":    ex["shap_summary"],
                "shap_vector":     ex["shap_vector"],
                "narrative_text":  ex["narrative_text"],
                "timestamp":       ts,
                "seeded":          True,
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"  Seeded {written} narrative examples → {_LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed the narrative feedback store with clinically plausible examples."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing examples even if the log already contains records.",
    )
    args = parser.parse_args()
    seed(force=args.force)
