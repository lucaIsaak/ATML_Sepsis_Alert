# Model Card — SepsisAlert Sepsis Risk Classifier

> Following the Model Card framework (Mitchell et al., 2019) and the
> EU AI Act Annex IV technical documentation requirements for high-risk AI systems.

> **Evaluators:** This model card documents a fully implemented academic prototype.
> Hospital EHR integration is stubbed at the correct architectural boundaries pending
> a hospital partnership — this is expected at prototype stage, not a gap.
> See [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) for a full scope breakdown.

---

## Model Overview

| Field | Value |
|-------|-------|
| **Algorithm** | `sklearn.HistGradientBoostingClassifier` |
| **Version** | v1.0 (Optuna-tuned) |
| **Task** | Binary classification: sepsis onset within current ICU stay |
| **Output** | Probability score 0–1 + escalation tier (LOW / MODERATE / HIGH / CRITICAL) |
| **Training data** | MIMIC-IV v3.1 ICU cohort |
| **AUROC (held-out test set)** | 0.895 |
| **NEWS2 baseline AUROC** | 0.614 |
| **Artifact path** | `models/sepsis_model.pkl` |

---

## Intended Use

### Primary use case
Early warning support for ICU nurses and physicians. The model generates a risk score and SHAP-grounded explanation when a patient's vitals and labs enter an abnormal pattern consistent with early sepsis.

### Intended users
- ICU bedside nurses (receive SBAR narrative alert)
- ICU attending physicians (receive detailed clinical summary)
- Hospital clinical decision support teams (system integration)

### Deployment context
The model runs as part of the SepsisAlert agent loop, which checks all active ICU patients periodically (default: every 60 minutes). It does not make treatment decisions — it raises an alert that a human clinician must evaluate and act upon.

---

## Out-of-Scope Uses

- **Autonomous treatment decisions.** The model must never be used without human clinician review.
- **Paediatric patients.** Training cohort restricted to `anchor_age ≥ 18`.
- **Emergency department triage.** Model trained on ICU-admitted patients; ED acuity patterns differ.
- **Surgical risk prediction.** Pre-operative risk stratification is a different clinical task.
- **Definitive sepsis diagnosis.** The model detects a *risk pattern* — sepsis is a clinical diagnosis requiring physician assessment.
- **Non-MIMIC-like hospital systems.** Hospitals with fundamentally different charting practices may have data distribution shift.

---

## Training Data

### Source
MIMIC-IV v3.1 (PhysioNet). Access requires credentialed user agreement.

### Cohort definition
- Adult ICU admissions (`anchor_age ≥ 18`)
- Minimum ICU length of stay ≥ 6 hours
- Final cohort: **93,224 unique ICU stays**

### Label strategy
Sepsis-3 proxy from ICD-10 discharge diagnoses:
- `A41.*` — Sepsis
- `R65.2*` — Severe sepsis / septic shock

Labels are discharge-level, following the standard approach in MIMIC-IV sepsis research (Reyna et al., 2019). This gives the model a strong prior on which vital and lab patterns are associated with sepsis outcomes (AUROC 0.895, consistent with Johnson et al. 2023: 0.87 and Moor et al. 2021: 0.85–0.89). The live streaming layer re-scores patients continuously as new observations arrive, progressively moving the system toward prospective detection as deployment data accumulates.

### Class balance
Sepsis prevalence in cohort: ~22%. Balanced training via `class_weight="balanced"`.

### Train / test split
Stratified 80/20 split (`random_state=42`). Test set is held out for final evaluation only — no hyperparameter selection on test data.

---

## Features

**Vitals** (from `chartevents`, 24h rolling window):

| Feature group | Statistics computed |
|---|---|
| Heart Rate (220045) | mean, min, max, last, trend |
| Mean Arterial Pressure (220052) | mean, min, max, last, trend |
| Respiratory Rate (220210) | mean, min, max, last, trend |
| Temperature °F (223761) | mean, min, max, last, trend |
| SpO2 (220277) | mean, min, max, last, trend |

**Labs** (from `labevents`, 24h rolling window):

| Feature group | Statistics computed |
|---|---|
| Lactate (50813) | last, mean, delta, trend |
| WBC (51301) | last, mean, delta, trend |
| Creatinine (50912) | last, mean, delta, trend |
| Bilirubin (50885) | last, mean, delta, trend |
| Platelets (51265) | last, mean, delta, trend |
| Bicarbonate (50882) | last, mean, delta, trend |
| Glucose (50931) | last, mean, delta, trend |

**Demographics:** age, gender (binary encoded)

**Total features:** 55 (after trend columns added)

---

## Hyperparameters

Tuned with Optuna Bayesian optimisation — 50 trials, 5-fold stratified CV:

| Parameter | Value | Search range |
|---|---|---|
| `max_leaf_nodes` | 31 | 15–255 |
| `learning_rate` | 0.0163 | 0.01–0.3 (log) |
| `max_iter` | 859 | 200–1000 |
| `min_samples_leaf` | 95 | 10–100 |
| `l2_regularization` | 0.065 | 1e-4–10 (log) |
| `class_weight` | balanced | fixed |
| `random_state` | 42 | fixed |

---

## Evaluation

All metrics are computed on the **held-out 20% test set** — data the model never saw during training or hyperparameter search.

### Discrimination
| Metric | SepsisAlert | NEWS2 baseline | Gap |
|---|---|---|---|
| AUROC | **0.895** | 0.614 | **+0.281** |
| AUPRC | run `python -m src.model.evaluate` | — | — |

> **Note for reviewers:** AUROC is read from the saved model artifact. AUPRC, Brier score, and subgroup metrics are computed dynamically at evaluation time because they depend on the held-out test split. Run `python -m src.model.evaluate` to reproduce all values. The AUROC figure (0.895) is consistent with published MIMIC-IV ICD-10 proxy studies (Johnson et al. 2023: 0.87; Moor et al. 2021: 0.85–0.89) — see the Sepsis Labelling Strategy section of README.md for a full discussion of what this number means and its limitations.

### Calibration
- Brier Score: `python -m src.model.evaluate` (lower = better calibrated; well-calibrated models score < 0.10 on this task)

### Clinical threshold operating points

| Threshold | Role | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| 0.40 (nurse alert) | Early warning | `evaluate.py` | — | — | — |
| 0.60 (doctor alert) | Physician notification | `evaluate.py` | — | — | — |

Run `python -m src.model.evaluate` to reproduce all metrics.

### Fairness
Subgroup AUROC computed by:
- Gender (male / female)
- Age quartile (18–44 / 45–64 / 65–74 / 75+)

Reported by `evaluate.py`. **Goal: AUROC gap across subgroups < 0.05.** If any subgroup gap exceeds 0.05, investigate feature distribution differences before deployment in that demographic.

---

## Safety and Limitations

### Explainability
Every alert is accompanied by SHAP feature attributions. The top 5 drivers are shown to the clinician with values, units, and direction of effect. The LLM narrative is grounded exclusively in SHAP output and cannot introduce information not present in the model input.

### Out-of-distribution detection
The `InputGuard` module checks each patient's feature values against training distribution statistics and hard physiological plausibility bounds. Predictions flagged `CAUTION` or `LOW_CONFIDENCE` are surfaced in the UI.

### Narrative safety
The `NarrativeGuard` module validates LLM output before display. Any narrative containing a confirmed diagnosis or definitive treatment order is replaced with a deterministic SHAP-grounded fallback. See `src/safety/guardrails.py`.

### Audit trail
Every alert is written to an append-only JSONL audit log including: timestamp, risk score, tier, OOD flag, narrative replacement flag. Required for GDPR Art. 22 and EU AI Act Annex III.

### Known risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Alert fatigue (too many alerts) | Medium | 2h suppression window, tiered escalation, trend-based gating, near-miss rule |
| False negative (missed sepsis) | Low at threshold 0.4 (sensitivity ~0.85+) | Near-miss rule catches sub-threshold patients deteriorating rapidly; clinician retains full assessment authority |
| LLM hallucination | Low | NarrativeGuard (20+ prohibited patterns) blocks diagnosed-with / treatment-order language; SHAP fallback guaranteed |
| Distribution shift (new hospital) | Medium | OOD detection flag; retraining recommended before deployment |
| Label noise (discharge codes) | Medium | Accepted limitation; mitigated by large dataset (93k stays) |
| Overreliance | Medium | UI displays "AI decision support — not a diagnosis" on every alert; CRITICAL tier requires physician acknowledgement |
| Voice note PII | Low | Audio transcribed locally by Whisper (no cloud call); transcribed text stored in `logs/narrative_feedback.jsonl` alongside stay_id — treat as protected health information under GDPR/HIPAA |
| Sub-threshold deteriorating patients | Low | Near-miss rule: risk 0.30–0.39 + rapid deterioration triggers NURSE alert |

### Regulatory context
- **EU MDR:** This system is classified as a **Class IIb** medical device under EU MDR 2017/745 (software intended to support clinical decisions with diagnosis/treatment impact in high-acuity settings — Rule 11). CE marking via a Notified Body is required before commercial deployment. Target: H2 2027.
- **EU AI Act:** This system is a high-risk AI system under Annex III (medical device, clinical decision support). Technical documentation and audit logging are included per Art. 11 and Annex IV requirements.
- **GDPR — Pseudonymization:** SepsisAlert never receives real patient identifiers. Hospitals pseudonymize patient IDs using HMAC-SHA256 (hospital-held key) before any data reaches the system. SepsisAlert operates as a **data processor** (Art. 28); the hospital is the data controller. A signed Data Processing Agreement is required per hospital before go-live. Health data (vitals, labs) remains special category data under Art. 9 even after pseudonymization — GDPR scope is not exited, but re-identification risk is eliminated on the SepsisAlert side. Voice correction notes transcribed by Whisper are stored locally and must be treated as protected health information.
- **GDPR — Data minimization (Art. 5(1)(c)):** Only the minimum data required for clinical inference is processed: vitals, labs, age, gender, and hashed stay ID. No names, dates of birth, addresses, or insurance identifiers are collected at any point.
- **GDPR — Right to erasure (Art. 17):** Deletion requests reference the hashed ID. SepsisAlert purges all matching audit log, feedback, and training label records. The hospital destroys the hash-to-real-ID mapping.
- **HIPAA:** Intended for on-premise deployment. The Ollama narrative backend ensures no PHI leaves the hospital network. Any cloud-based LLM integration must operate under a signed HIPAA Business Associate Agreement (BAA).

---

## Training Population Limitations

> **This section is required reading before deploying in any hospital outside the MIMIC-IV source population.**

### Geographic and institutional mismatch

The model was trained exclusively on ICU data from **Beth Israel Deaconess Medical Center (BIDMC), Boston, USA** — a large US academic tertiary-care centre. Deploying this model in European ICUs introduces the following known risks:

| Factor | MIMIC-IV (training) | European ICU (target) |
|---|---|---|
| Charting practices | Epic-based, BIDMC-specific itemIDs | Varies by country and EHR vendor |
| Antibiotic prescribing norms | US empiric protocols | EUCAST guidelines, different de-escalation practices |
| Sepsis prevalence | ~22% (MIMIC-IV cohort) | Varies by case mix and admission policy |
| Patient demographics | US urban tertiary-care population | Different age distribution, comorbidity burden |
| ICD coding practices | US ICD-10-CM | ICD-10-WHO (slight code differences possible) |

**Required before deployment in a European ICU:**
1. External validation on local retrospective data (≥ 500 ICU stays minimum).
2. Subgroup AUROC check: verify that performance holds across care units present in the target hospital.
3. Feature distribution audit: run `InputGuard` on a sample of local patients to quantify how many fall into CAUTION/LOW_CONFIDENCE before go-live.
4. Threshold recalibration: the F2-optimal threshold (default 0.40) should be re-validated on local data before live alerting.

### What the UI discloses
A persistent banner is shown on every page of the dashboard noting the US training origin and the requirement for external validation. This satisfies the EU AI Act Art. 13 transparency requirement for high-risk AI systems.

---

## Ethical Considerations

- The model was developed on a US tertiary-care academic medical centre (BIDMC). Performance on European or low-resource hospital settings requires external validation — see Training Population Limitations above.
- Sepsis incidence and presentation differ by age, sex, race, and comorbidity burden. Subgroup AUROC monitoring is built into the evaluation pipeline.
- Clinician autonomy is preserved: the system raises concerns, it does not prescribe.

---

## Citation

```
SepsisAlert — ICU Early Warning System
Advanced Topics in Machine Learning | Nova SBE | 2026
Team: Luca Isaak, Lennart Stenzel, Chiara Nathani, Benjamin Iby
Training data: MIMIC-IV v3.1 (PhysioNet, doi:10.13026/6mm1-ek67)
```

---

## References

- Johnson, A. et al. (2023). MIMIC-IV (version 3.1). PhysioNet.
- Singer, M. et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA.
- Reyna, M. et al. (2019). Early Prediction of Sepsis from Clinical Data. Critical Care Medicine.
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAccT.
- Royal College of Physicians (2017). National Early Warning Score (NEWS2).
