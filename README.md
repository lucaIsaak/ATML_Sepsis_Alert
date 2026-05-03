# SepsisAlert — Early ICU Sepsis Detection with Explainable AI

> AI-powered real-time alerts with clinician-ready explanations
> Advanced Topics in Machine Learning | Nova SBE | 2026

---

## The Problem

Sepsis kills **11 million people/year** — 1 in 5 global deaths. In European ICUs, ~30% of patients develop sepsis and mortality reaches 41.9%. Every hour of delayed treatment increases mortality by **~7%**.

Current tools (NEWS2, SIRS) are static, rule-based, and provide **no explanation** for why an alert was triggered. Clinicians receive a score they can't trust, leading to alert fatigue.

**SepsisAlert solves this**: a LightGBM model detects sepsis 4–6 hours early, SHAP traces every alert to its exact clinical cause, and a local LLM translates the output into plain-language explanations nurses can act on immediately.

---

## Architecture

```
MIMIC-IV / Hospital EHR
         │
    ┌────▼────┐
    │  DuckDB  │   ← fast SQL directly on raw .csv.gz files
    └────┬────┘
         │  cohort + features (24h rolling windows)
    ┌────▼──────────┐
    │  LightGBM     │   ← trained on MIMIC-IV, sepsis-3 labels
    │  (local .pkl) │
    └────┬──────────┘
         │  risk score (0–1)
    ┌────▼──────┐
    │   SHAP    │   ← top contributing features per patient
    └────┬──────┘
         │  feature importances
    ┌────▼─────────────────────────┐
    │  Narrative Generator         │
    │  Ollama / mistral:7b (local) │   ← on-premise, GDPR-safe
    │  HuggingFace fallback        │
    └────┬─────────────────────────┘
         │  plain-language clinical explanation
    ┌────▼──────────────────┐
    │  PatientMonitorAgent  │   ← agentic monitoring loop
    │  (tool-use pattern)   │
    └────┬──────────────────┘
         │
    ┌────▼──────────┐
    │   Streamlit   │   ← ICU dashboard, alerts, SHAP charts
    └───────────────┘
```

---

## Agentic Approach

The core of SepsisAlert is a **PatientMonitorAgent** — not a simple pipeline but an agent that:

1. **Monitors** incoming vitals/lab data for all active ICU stays
2. **Decides** whether to run inference (new data available, patient not yet alerted)
3. **Runs** the sepsis model and generates a risk score
4. **Explains** the alert using SHAP + LLM narrative if risk exceeds threshold
5. **Dispatches** the alert to the Streamlit dashboard

This maps directly to real clinical deployment: the agent replaces a nurse manually watching a screen. Each tool the agent calls corresponds to a real clinical action.

```python
class PatientMonitorAgent:
    tools = [
        fetch_latest_vitals,      # pull new chartevents / labevents
        run_sepsis_model,         # LightGBM inference → risk score
        explain_with_shap,        # SHAP top-5 feature drivers
        generate_narrative,       # Ollama → plain-language explanation
        dispatch_alert,           # send to dashboard / log
    ]
```

---

## Sepsis Labelling Strategy

We use the **Sepsis-3 ICD-10 proxy** from `diagnoses_icd`:
- Codes: `A41.*` (sepsis), `R65.2*` (severe sepsis / septic shock)
- Cross-referenced against ICU admission timing for temporal alignment
- Labels are assigned at the *stay level*, with prediction target = onset - 6h

---

## Data Sources (MIMIC-IV 3.1)

| File | Module | Purpose |
|------|--------|---------|
| `icu/icustays.csv.gz` | ICU | Anchor table for all stays |
| `icu/chartevents.csv.gz` | ICU | Vitals: HR, MAP, temp, SpO2, RR |
| `hosp/labevents.csv.gz` | Hospital | Labs: lactate, WBC, creatinine, bilirubin, platelets |
| `hosp/diagnoses_icd.csv.gz` | Hospital | Sepsis labels (ICD-10 A41.x) |
| `hosp/patients.csv.gz` | Hospital | Age, gender |
| `hosp/admissions.csv.gz` | Hospital | Admission type, mortality flag |

---

## Features (24h Rolling Windows)

**Vitals** (from chartevents):
- Heart Rate — mean, min, max, trend
- Mean Arterial Pressure — mean, min
- Temperature — max
- SpO2 — min
- Respiratory Rate — mean, max

**Labs** (from labevents):
- Lactate — last value, delta
- WBC — last value, delta
- Creatinine — last value, delta
- Bilirubin — last value
- Platelets — last value, delta

**Clinical scores**:
- NEWS2 score (reconstructed) — baseline comparison

---

## Model

- **Algorithm**: LightGBM (gradient boosting, fast inference, works well on tabular EHR data)
- **Training data**: MIMIC-IV ICU cohort (~50k stays)
- **Label**: Sepsis-3 onset within next 6 hours (binary classification)
- **Evaluation**: AUROC, sensitivity/specificity, alert fatigue reduction vs NEWS2

---

## Narrative Layer

- **Primary**: Ollama `mistral:7b` running **locally** (no data leaves the machine/hospital)
- **Cloud option**: Claude API (for demo/non-sensitive environments)
- **HuggingFace option**: `epfl-llm/meditron-7b` (medical-domain fine-tuned)

The LLM receives only the SHAP feature importances and vital signs summary — **no raw patient data** is passed to any external API.

---

## Project Structure

```
ATML_Sepsis_Alert/
├── src/
│   ├── data/
│   │   ├── cohort.py           # DuckDB cohort extraction
│   │   ├── features.py         # Feature engineering (rolling windows)
│   │   └── labels.py           # Sepsis-3 label generation
│   ├── model/
│   │   ├── train.py            # LightGBM training
│   │   ├── evaluate.py         # AUROC, NEWS2 comparison
│   │   └── predict.py          # Inference
│   ├── explainability/
│   │   └── shap_explainer.py   # SHAP wrapper
│   ├── narrative/
│   │   ├── ollama_client.py    # Ollama narrative generation
│   │   └── prompts.py          # Clinical prompt templates
│   ├── agent/
│   │   └── monitor_agent.py    # PatientMonitorAgent
│   └── app/
│       └── dashboard.py        # Streamlit ICU dashboard
├── models/                     # Saved model artifacts (gitignored)
├── notebooks/                  # EDA and model development
├── data/                       # Processed data (gitignored)
├── tests/
├── config.yaml                 # Model + app configuration
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Data processing | DuckDB | Queries .csv.gz directly, no data loading overhead |
| ML model | LightGBM | Best-in-class for tabular EHR data, fast inference |
| Explainability | SHAP | Industry standard, directly interpretable |
| Narrative LLM | Ollama / mistral:7b | On-premise, GDPR-compliant, zero per-call cost |
| Frontend | Streamlit | Fast to build, clinical dashboard-friendly |
| Agent pattern | Custom Python | Tool-use pattern, easy to extend |

---

## Business Model

- **SaaS**: €200–350 / ICU bed / month
- **Onboarding**: €15K–30K one-time per hospital
- **Target**: >70% gross margin at scale
- **Beachhead**: DACH region (~28K ICU beds)
- **TAM**: €171M (EU27)

**Unit economics**: 20-bed ICU = €48K–84K ARR. LLM inference cost = **€0** (local Ollama). Cloud hosting ~€20–50/month. Primary cost driver = sales & implementation.

---

## Why Not Just a Wrapper

SepsisAlert is **not** a ChatGPT wrapper. The LLM only generates the explanation — the risk score comes from a validated LightGBM model trained on real ICU outcomes. The LLM is grounded on SHAP output and cannot override the model score. This satisfies EU AI Act CDSS compliance requirements that pure LLM outputs cannot.

---

## Setup

```bash
# Clone and set up environment
git clone <repo>
cd ATML_Sepsis_Alert
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit config.yaml with your MIMIC-IV data path

# Run data pipeline
python -m src.data.cohort
python -m src.data.features

# Train model
python -m src.model.train

# Launch dashboard
streamlit run src/app/dashboard.py
```

---

## Team

| Name | Role |
|------|------|
| Luca Isaak | ML Engineering |
| Lennart Stenzel | ML Engineering |
| Chiara Nathani | Market Research & Presentation |
| Benjamin Iby | Business Model & Financials |
