# SepsisAlert — Early ICU Sepsis Detection with Explainable AI

> AI-powered real-time alerts with clinician-ready explanations
> Advanced Topics in Machine Learning | Nova SBE | 2026

---

## The Problem

Sepsis kills **11 million people/year** — 1 in 5 global deaths. In European ICUs, ~30% of patients develop sepsis and mortality reaches 41.9%. Every hour of delayed treatment increases mortality by **~7%**.

Current tools (NEWS2, SIRS) are static, rule-based, and provide **no explanation** for why an alert was triggered. Clinicians receive a score they can't trust, leading to alert fatigue.

**SepsisAlert solves this**: a gradient boosting model detects sepsis 4–6 hours early, SHAP traces every alert to its exact clinical cause, and a local LLM translates the output into plain-language explanations nurses can act on immediately.

---

## Architecture

```
MIMIC-IV / Hospital EHR
         │
    ┌────▼────┐
    │  DuckDB  │   ← fast SQL directly on raw .csv.gz files
    └────┬────┘
         │  cohort + features (24h rolling windows, trend slopes)
    ┌────▼──────────────────────────┐
    │  HistGradientBoosting         │   ← trained on MIMIC-IV, Sepsis-3 labels
    │  (sklearn, Optuna-tuned)      │   ← AUROC 0.895 vs NEWS2 0.614
    └────┬──────────────────────────┘
         │  risk score (0–1)
    ┌────▼──────────────────────────┐
    │  AI Safety Guardrails         │   ← OOD detection, narrative validation,
    │  (src/safety/guardrails.py)   │      audit log (GDPR / EU AI Act)
    └────┬──────────────────────────┘
         │
    ┌────▼──────┐
    │   SHAP    │   ← top-5 feature drivers per patient
    └────┬──────┘
         │  feature importances + clinical reference ranges
    ┌────▼─────────────────────────┐
    │  Narrative Generator         │
    │  Ollama / mistral:7b (local) │   ← on-premise, GDPR-safe
    │  Claude API / HF fallback    │
    └────┬─────────────────────────┘
         │  SBAR-structured clinical explanation
    ┌────▼──────────────────┐
    │  PatientMonitorAgent  │   ← ReAct-pattern monitoring loop
    │  4-tier escalation    │      NONE → NURSE → DOCTOR → CRITICAL
    └────┬──────────────────┘
         │
    ┌────▼──────────┐
    │   Streamlit   │   ← ICU dashboard, alerts, SHAP charts
    └───────────────┘
```

---

## Agentic Approach

The core is a **PatientMonitorAgent** using the ReAct (Reason + Act) pattern:

1. **OBSERVE** — pull latest features from the patient buffer
2. **THINK** — reason about risk score, trend, alert history, escalation tier
3. **ACT** — call the right tools for the right tier
4. **REMEMBER** — update per-patient memory (trajectory, last physician notification)

Escalation tiers:

| Tier | Trigger | Action |
|---|---|---|
| NONE | risk < 0.40 | No alert |
| NURSE | risk 0.40–0.59 | SBAR narrative to bedside nurse |
| DOCTOR | risk 0.60–0.79 | Clinical summary to attending physician |
| CRITICAL | risk ≥ 0.80 OR rapid deterioration | Immediate escalation |

Alert fatigue is addressed by a 2-hour suppression window and trend-based override (rapid deterioration escalates regardless of suppression).

---

## AI Safety — Three-Layer Guardrails

Safety is built into every alert cycle (`src/safety/guardrails.py`):

| Layer | What it does | Why |
|---|---|---|
| **InputGuard** | OOD detection via z-score against training distribution and hard physiological bounds. Flags `NORMAL / CAUTION / LOW_CONFIDENCE`. | Prevents silent model extrapolation on implausible inputs |
| **NarrativeGuard** | Validates LLM output for prohibited phrases (confirmed diagnoses, definitive treatment orders). Replaces with deterministic SHAP fallback if violated. | Guarantees clinical safety even if Ollama misbehaves |
| **AuditLogger** | Append-only JSONL log: timestamp, risk score, tier, OOD flag, narrative replacement flag. | GDPR Art. 22 (automated decision transparency) + EU AI Act Annex III |

See `MODEL_CARD.md` for full safety documentation.

---

## Sepsis Labelling Strategy

Labels use the **Sepsis-3 ICD-10 proxy** from `diagnoses_icd`:
- Codes: `A41.*` (sepsis), `R65.2*` (severe sepsis / septic shock)
- Labels are assigned at the **stay level** from discharge codes.

**Known limitation**: a discharge-level label does not capture the exact time of sepsis onset. A patient labelled `sepsis=1` may appear clinically normal early in the stay. This is the standard approach in MIMIC-IV research (Reyna et al., 2019) and is accepted for a research setting. Production deployment would use rolling per-hour labels aligned to Sepsis-3 onset time.

---

## Data Sources (MIMIC-IV 3.1)

| File | Purpose |
|------|---------|
| `icu/icustays.csv.gz` | Anchor table for all stays |
| `icu/chartevents.csv.gz` | Vitals: HR, MAP, temp, SpO2, RR |
| `hosp/labevents.csv.gz` | Labs: lactate, WBC, creatinine, bilirubin, platelets, bicarbonate, glucose |
| `hosp/diagnoses_icd.csv.gz` | Sepsis labels (ICD-10 A41.x) |
| `hosp/patients.csv.gz` | Age, gender |
| `hosp/admissions.csv.gz` | Admission type, mortality flag |

---

## Features (24h Rolling Windows — 55 total)

**Vitals** (from chartevents): Heart Rate, MAP, Respiratory Rate, Temperature, SpO2
— each with: mean, min, max, last, **trend** (linear slope, units/hour)

**Labs** (from labevents): Lactate, WBC, Creatinine, Bilirubin, Platelets, Bicarbonate, Glucose
— each with: last, mean, delta, **trend**

**Demographics**: age, gender

Trend features capture whether a marker is stable, rising, or falling — rising lactate at 3.0 mmol/L is clinically different from stable lactate at 3.0 mmol/L.

---

## Model

- **Algorithm**: `sklearn.HistGradientBoostingClassifier` — pure Python, no native deps, natively handles NaN
- **Hyperparameters**: tuned with Optuna Bayesian optimisation (50-trial search, 5-fold stratified CV)
- **Training data**: MIMIC-IV ICU cohort — 93,224 stays, adults, ICU LOS ≥ 6h
- **Performance**: AUROC **0.895** vs NEWS2 baseline **0.614** (+0.281)
- **Calibration**: Brier score reported by `src/model/evaluate.py`
- **Fairness**: subgroup AUROC by gender and age quartile reported at evaluation

See `MODEL_CARD.md` for full model documentation.

---

## Clinician Feedback Loop

SepsisAlert includes a lightweight active-learning loop that lets clinicians label patients directly from the dashboard and feed those labels back into the next model training cycle.

### How it works

**1. Labelling in the dashboard**

On the Patient Detail page, two buttons appear above the risk gauge for every patient:

| Button | Meaning | Stored as |
|---|---|---|
| ✅ Confirm Sepsis | Clinician verifies this patient did develop sepsis | `confirmed_sepsis` |
| 🚩 Flag as Wrong | Alert fired but clinician believes it was a false positive | `flagged_wrong` |

Labels are saved immediately to `data/feedback/feedback.csv` (gitignored). A clinician can always click again to update a previous decision — each patient has exactly one label at any time.

**Why no "Rule Out Sepsis" button?** Because absence of a diagnosis does not equal absence of disease — the patient may have developed sepsis after discharge, or the alert may have prompted early intervention that prevented it. Only `confirmed_sepsis` is a clean ground-truth label. `flagged_wrong` is treated as a provisional negative with reduced weight.

**2. Retraining with feedback**

```bash
# See what would happen without writing any files
python retrain_with_feedback.py --dry-run

# Retrain — saves new model only if AUROC improves
python retrain_with_feedback.py

# Force save regardless of AUROC
python retrain_with_feedback.py --force
```

The script (`retrain_with_feedback.py`):
- Loads the original feature matrix and overrides labels for all clinician-labelled patients
- Applies differential sample weights so clinician labels are trusted more than automated ones:

| Label source | Sample weight | Rationale |
|---|---|---|
| Original automated label | 1.0 | Baseline |
| `confirmed_sepsis` | 3.0 | High-confidence, manually verified |
| `flagged_wrong` | 0.5 | Provisional — uncertain negative |

- Trains a new model using the same hyperparameters from `config.yaml`
- Compares old vs. new AUROC on the same validation split
- Only overwrites `models/sepsis_model.pkl` if the new model is better
- Always creates a timestamped backup (`models/sepsis_model_backup_<ts>.pkl`) before saving

**3. Feedback data structure**

`data/feedback/feedback.csv`:

| Column | Example | Description |
|---|---|---|
| `stay_id` | `30002932` | ICU stay identifier |
| `feedback_type` | `confirmed_sepsis` | Clinician decision |
| `risk_score` | `0.914` | Model score at time of labelling |
| `timestamp` | `2026-05-07T14:32:10` | When the label was given |
| `low_confidence` | `False` | True for `flagged_wrong` entries |

The feedback CSV is gitignored alongside all other patient data.

---

## Narrative Layer

- **Primary**: Ollama `mistral:7b` running **locally** — no data leaves the machine
- **Fallback 1**: Claude API (non-sensitive / demo environments only)
- **Fallback 2**: HuggingFace `epfl-llm/meditron-7b`

Prompts are SBAR-structured and grounded in SHAP output only. The `NarrativeGuard` validates every output before it reaches the clinician. The LLM **cannot** override the model score.

---

## Project Structure

```
ATML_Sepsis_Alert/
├── src/
│   ├── data/
│   │   ├── cohort.py           # DuckDB cohort extraction (MIMIC-IV)
│   │   ├── features.py         # Feature engineering (24h rolling windows + trends)
│   │   ├── feedback.py         # Clinician feedback store + retraining label bridge
│   │   ├── patient_buffer.py   # Streaming per-patient rolling buffer
│   │   └── streaming.py        # MIMIC stream simulator (FHIR-compatible)
│   ├── model/
│   │   ├── train.py            # Model training (HistGradientBoosting)
│   │   ├── tune.py             # Optuna hyperparameter search
│   │   ├── evaluate.py         # AUROC, Brier, clinical thresholds, subgroup fairness
│   │   └── predict.py          # Single-patient and batch inference
│   ├── explainability/
│   │   └── shap_explainer.py   # SHAP wrapper + feature label/unit mapping
│   ├── narrative/
│   │   ├── ollama_client.py    # Ollama narrative generation (nurse + doctor)
│   │   └── prompts.py          # SBAR prompt templates + clinical thresholds
│   ├── safety/
│   │   └── guardrails.py       # InputGuard, NarrativeGuard, AuditLogger
│   ├── agent/
│   │   └── monitor_agent.py    # PatientMonitorAgent (ReAct loop)
│   ├── integrations/
│   │   └── fhir_adapter.py     # HL7 FHIR R4 adapter (Epic / Cerner)
│   ├── app/
│   │   └── dashboard.py        # Streamlit ICU dashboard
│   └── schemas.py              # Pydantic input/output schemas
├── tests/
│   ├── test_model.py           # Inference correctness
│   ├── test_schemas.py         # Clinical validation rules
│   ├── test_narrative.py       # Prompt structure + LLM client
│   └── test_safety.py          # Guardrails (22 tests)
├── notebooks/
│   ├── 01_eda.ipynb            # Cohort exploration
│   └── 02_feature_analysis.ipynb # Feature importance (permutation)
├── models/                     # Model artifacts (gitignored)
├── data/                       # Processed data (gitignored)
├── logs/                       # Audit logs (gitignored)
├── setup_demo.py               # One-command demo setup (synthetic data, no MIMIC needed)
├── run_pipeline.py             # Full MIMIC-IV pipeline runner
├── retrain_with_feedback.py    # Feedback-driven retraining (run manually after labelling)
├── config.yaml                 # Model + app configuration
├── MODEL_CARD.md               # Model documentation (EU AI Act Annex IV)
├── TRANSPARENCY_LOG.md         # GenAI tool usage disclosure
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Data processing | DuckDB | Queries `.csv.gz` directly, no loading overhead |
| ML model | sklearn HistGradientBoosting + Optuna | Best-in-class tabular, no native deps, Bayesian-tuned |
| Explainability | SHAP | Industry standard, directly interpretable |
| AI Safety | Custom guardrails | OOD detection, narrative validation, audit log |
| Narrative LLM | Ollama / mistral:7b | On-premise, GDPR-compliant, zero per-call cost |
| EHR integration | HL7 FHIR R4 adapter | Compatible with Epic / Oracle Health (Cerner) |
| Frontend | Streamlit | Fast to build, clinical dashboard-friendly |
| Agent pattern | ReAct (custom Python) | Reason + Act loop, 4-tier escalation, trend memory |
| Validation | Pydantic v2 | Clinical bounds enforced at schema level |
| Testing | pytest (22+ tests) | Model, schemas, narrative, safety guardrails |

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

SepsisAlert is **not** a ChatGPT wrapper. The LLM only generates the explanation — the risk score comes from a validated gradient boosting model trained on 93,224 real ICU outcomes. The LLM is grounded on SHAP output and cannot override the model score. Moat: proprietary MIMIC-IV training, clinical workflow integration, FHIR adapter, EU AI Act-compliant audit trail, and switching costs from EHR integration.

---

## Quick Start (no MIMIC-IV required)

```bash
git clone <repo>
cd ATML_Sepsis_Alert
pip install -r requirements.txt

# Generate synthetic demo data + train demo model (~10 seconds)
python setup_demo.py

# Launch the dashboard
streamlit run src/app/dashboard.py
```

`setup_demo.py` generates 200 fully synthetic ICU patients (no real patient data — see legal notice in the script) and trains a demo model. The dashboard is immediately usable.

### React Frontend (alternative UI)

A modern React dashboard is available alongside the Streamlit app.

**Install dependencies (first time only):**
```bash
cd frontend && npm install
```

**Run the full stack:**
```bash
# Terminal 1 — FastAPI backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — React frontend
cd frontend && npm run dev
```

The React app runs on [http://localhost:5173](http://localhost:5173) and proxies API calls to the backend on port 8000. Make sure `setup_demo.py` has been run first so the model and data files exist.

---

## Full Pipeline (with MIMIC-IV access)

```bash
pip install -r requirements.txt

# Configure data paths
cp .env.example .env
# Edit config.yaml → set icu_path and hosp_path to your MIMIC-IV location

# Run full pipeline
python run_pipeline.py

# Or step by step:
python -m src.data.cohort
python -m src.data.features
python -m src.model.tune        # Optuna (optional, ~30 min)
python -m src.model.train
python -m src.model.evaluate    # AUROC, Brier, subgroup fairness

streamlit run src/app/dashboard.py
```

---

## Tests

```bash
pytest tests/ -v
```

Covers: model inference, clinical schema validation, SBAR prompt structure, LLM client fallback, and all three safety guardrail layers (22 tests total).

---

## Team

| Name | Role |
|------|------|
| Luca Isaak | ML Engineering |
| Lennart Stenzel | ML Engineering |
| Chiara Nathani | Market Research & Presentation |
| Benjamin Iby | Business Model & Financials |
