# SepsisAlert — Early ICU Sepsis Detection with Explainable AI

> AI-powered real-time alerts with clinician-ready explanations
> Advanced Topics in Machine Learning | Nova SBE | 2026

---

## The Problem

Sepsis kills **11 million people/year** — 1 in 5 global deaths. In European ICUs, ~30% of patients develop sepsis and mortality reaches 41.9%. Every hour of delayed treatment increases mortality by **~7%**.

Current tools (NEWS2, SIRS) are static, rule-based, and provide **no explanation** for why an alert was triggered. Clinicians receive a score they can't trust, leading to alert fatigue.

**SepsisAlert solves this**: a gradient boosting model detects risk patterns consistent with early-stage sepsis (AUROC 0.895 vs NEWS2 0.614), SHAP traces every alert to its exact clinical cause, and a local LLM translates the output into plain-language explanations nurses can act on immediately.

---

## Architecture

```
MIMIC-IV / Hospital EHR (FHIR R4)
         |
    +----v----+
    |  DuckDB  |   <- fast SQL directly on raw .csv.gz files
    +----+----+
         |  cohort + features (24h rolling windows, trend slopes)
    +----v-------------------------------+
    |  HistGradientBoosting              |   <- trained on MIMIC-IV, Sepsis-3 labels
    |  (sklearn, Optuna-tuned)           |   <- AUROC 0.895 vs NEWS2 0.614
    +----+-------------------------------+
         |  risk score (0-1)
    +----v-------------------------------+
    |  AI Safety Guardrails              |   <- OOD detection, narrative validation,
    |  (src/safety/guardrails.py)        |      audit log (GDPR / EU AI Act)
    +----+-------------------------------+
         |
    +----v------+
    |   SHAP    |   <- top feature drivers per patient
    +----+------+
         |  feature importances + clinical reference ranges
    +----v------------------------------+
    |  Narrative Generator              |
    |  Ollama / mistral:7b (local)      |   <- on-premise, GDPR-safe, streaming
    |  Few-shot + RAG context           |   <- learns from clinician ratings
    +----+------------------------------+
         |  SBAR-structured clinical explanation
    +----v---------------------+
    |  PatientMonitorAgent     |   <- ReAct-pattern monitoring loop
    |  4-tier escalation       |      NONE -> NURSE -> DOCTOR -> CRITICAL
    +----+---------------------+
         |
    +----v-----------+    +----v------------------+
    |   Streamlit    |    |  React + FastAPI       |
    |   Dashboard    |    |  (src/api + frontend/) |
    +----------------+    +-----------------------+
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
| CRITICAL | risk >= 0.80 OR rapid deterioration | Immediate escalation |

Alert fatigue is addressed by a 2-hour suppression window and trend-based override (rapid deterioration escalates regardless of suppression).

---

## AI Safety — Three-Layer Guardrails

Safety is built into every alert cycle (`src/safety/guardrails.py`) and enforced across **both** the Streamlit and React/FastAPI interfaces:

| Layer | What it does | Why |
|---|---|---|
| **InputGuard** | OOD detection via z-score against training distribution and hard physiological bounds. Flags `NORMAL / CAUTION / LOW_CONFIDENCE`. | Prevents silent model extrapolation on implausible inputs |
| **NarrativeGuard** | Validates LLM output for prohibited phrases (confirmed diagnoses, definitive treatment orders). Replaces with deterministic SHAP fallback if violated. | Guarantees clinical safety even if Ollama misbehaves |
| **AuditLogger** | Append-only JSONL log: timestamp, risk score, tier, OOD flag, narrative replacement flag. | GDPR Art. 22 (automated decision transparency) + EU AI Act Annex III |

All three layers are wired into the FastAPI routers as well as the Streamlit dashboard — no guardrail is bypassed regardless of which frontend is used. The audit log is accessible via `GET /audit`.

See `MODEL_CARD.md` for full safety documentation.

---

## Sepsis Labelling Strategy

Labels use the **Sepsis-3 ICD-10 proxy** from `diagnoses_icd`:
- Codes: `A41.*` (sepsis), `R65.2*` (severe sepsis / septic shock)
- Labels are assigned at the **stay level** from discharge codes.

**Known limitations and AUROC interpretation:**

ICD-10 codes are billing codes assigned at discharge — they carry no onset timestamp. Because of this, the feature extraction window (first 24h from ICU admission) overlaps with the disease period for patients who were already septic or deteriorating on arrival. The model therefore partially captures **concurrent sepsis presentation** alongside prospective risk, which inflates AUROC compared to a strictly prospective early-warning setup.

The reported AUROC of **0.895** is consistent with published MIMIC-IV ICD-10 proxy studies (Johnson et al. 2023: 0.87; Moor et al. 2021: 0.85–0.89) and is not an outlier. However, it should be interpreted as **sepsis risk stratification at ICU admission**, not as proof of a 6-hour advance prediction capability.

True prospective validation would require:
1. Deriving actual onset times from first antibiotic + suspected infection source (clinical Sepsis-3) — complex ETL across `prescriptions` and `microbiologyevents`.
2. Restricting feature extraction to strictly before each patient's individual onset time.

This is acknowledged as the primary technical limitation of the current pipeline and the standard trade-off in MIMIC-IV research.

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

## Features (24h Rolling Windows — 43 total)

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
- **Training data**: MIMIC-IV ICU cohort — 93,224 stays, adults, ICU LOS >= 6h
- **Performance**: AUROC **0.895** vs NEWS2 baseline **0.614** (+0.281)
- **Calibration**: Brier score reported by `src/model/evaluate.py`
- **Fairness**: subgroup AUROC by gender and age quartile reported at evaluation

See `MODEL_CARD.md` for full model documentation.

---

## Clinician Feedback Loop

SepsisAlert includes a full active-learning loop: clinicians label patients from the dashboard, ratings improve future narratives via RAG and few-shot prompting, and labels feed back into model retraining.

### 1. Clinical alert feedback (confirm / flag)

On the Patient Detail page, two buttons appear above the risk gauge for every patient:

| Button | Meaning | Stored as |
|---|---|---|
| Confirm Sepsis | Clinician verifies this patient did develop sepsis | `confirmed_sepsis` |
| Flag as Wrong | Alert fired but clinician believes it was a false positive | `flagged_wrong` |

Labels are saved immediately to `logs/feedback.jsonl`. A clinician can always click again to update.

### 2. Narrative quality feedback

After each LLM-generated narrative, the clinician can:
- Rate it 1–5 stars
- Add a free-text correction note
- Record a voice note (transcribed locally by OpenAI Whisper — no cloud call)

Ratings are saved to `logs/narrative_feedback.jsonl` with the full SHAP vector, narrative text, and model name.

### 3. Few-shot + RAG narrative improvement

Every time a new narrative is requested, the system automatically:

- **Few-shot**: retrieves the 2 highest-rated past narratives (>= 4 stars, same model) as style anchors
- **RAG**: finds the most clinically similar past patient by cosine similarity on SHAP vectors, and uses their narrative as a content anchor

This means narrative quality improves continuously with use, without any retraining.

### 4. Feedback-driven model retraining

```bash
# See what would happen without writing any files
python retrain_with_feedback.py --dry-run

# Retrain — saves new model only if AUROC improves
python retrain_with_feedback.py

# Force save regardless of AUROC
python retrain_with_feedback.py --force
```

The script:
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
- Always creates a timestamped backup before saving

### 5. LoRA narrative fine-tuning (optional)

Once >= 5 high-rated narratives have been collected, the feedback can be exported as Alpaca-format JSONL and used to fine-tune the Ollama model via LoRA:

```bash
python finetune_narrative.py --export-only   # just export the training pairs
python finetune_narrative.py                 # export + train + load into Ollama
```

---

## Narrative Layer

- **Primary**: Ollama `mistral:7b` running **locally** — no data leaves the machine
- **Streaming**: narratives stream token-by-token to the dashboard (typewriter effect)
- **Dynamic model selector**: switch between any locally installed Ollama model from the dashboard
- **Few-shot + RAG**: past clinician-rated narratives are surfaced as context examples
- **Audio feedback**: voice notes transcribed by local Whisper (no cloud call)
- **Fallback**: deterministic SHAP-based summary if NarrativeGuard blocks the LLM output

Prompts are SBAR-structured and grounded in SHAP output only. The LLM **cannot** override the model score.

---

## REST API (FastAPI backend)

The full stack includes a FastAPI backend alongside the Streamlit dashboard. It serves the React frontend and exposes all model capabilities programmatically.

```
GET  /patients                  — list all sampled patients, sorted by risk score
GET  /patients/{stay_id}        — patient detail: risk score, SHAP top/bottom, OOD flag
POST /narrative/stream          — stream SBAR narrative for a patient
GET  /narrative/models          — list available Ollama models
POST /feedback/clinical         — save confirm / flag decision
GET  /feedback/clinical/{id}    — retrieve existing clinical feedback
POST /feedback/narrative        — save narrative star rating + correction note
POST /feedback/transcribe       — transcribe audio feedback via Whisper
GET  /stats                     — model performance metrics + ROC curve data
GET  /model/info                — algorithm, AUROC, feature count, sklearn version
GET  /audit                     — last N audit log entries (GDPR Art. 22)
```

All narrative requests pass through NarrativeGuard and AuditLogger before the response is returned.

---

## Project Structure

```
ATML_Sepsis_Alert/
├── src/
│   ├── data/
│   │   ├── cohort.py               # DuckDB cohort extraction (MIMIC-IV)
│   │   ├── features.py             # Feature engineering (24h rolling windows + trends)
│   │   ├── feedback.py             # Clinical feedback store + retraining label bridge
│   │   ├── narrative_feedback.py   # Narrative ratings, few-shot + RAG retrieval
│   │   ├── patient_buffer.py       # Streaming per-patient rolling buffer
│   │   └── streaming.py            # MIMIC stream simulator (FHIR-compatible)
│   ├── model/
│   │   ├── train.py                # Model training (HistGradientBoosting)
│   │   ├── tune.py                 # Optuna hyperparameter search
│   │   ├── evaluate.py             # AUROC, Brier, clinical thresholds, subgroup fairness
│   │   └── predict.py              # Single-patient and batch inference
│   ├── explainability/
│   │   └── shap_explainer.py       # SHAP wrapper + feature label/unit mapping
│   ├── narrative/
│   │   ├── ollama_client.py        # Ollama narrative generation + streaming
│   │   ├── prompts.py              # SBAR prompt templates + clinical thresholds
│   │   └── transcribe.py           # Local Whisper audio transcription
│   ├── safety/
│   │   └── guardrails.py           # InputGuard, NarrativeGuard, AuditLogger
│   ├── agent/
│   │   └── monitor_agent.py        # PatientMonitorAgent (ReAct loop)
│   ├── integrations/
│   │   └── fhir_adapter.py         # HL7 FHIR R4 adapter (Epic / Cerner)
│   ├── api/
│   │   ├── main.py                 # FastAPI app + lifespan loader
│   │   └── routers/
│   │       ├── patients.py         # GET /patients, GET /patients/{id}
│   │       ├── narrative.py        # POST /narrative/stream, GET /narrative/models
│   │       ├── feedback.py         # POST/GET /feedback/clinical, narrative, transcribe
│   │       └── stats.py            # GET /stats, /model/info, /audit
│   └── schemas.py                  # Pydantic input/output schemas (FHIR + clinical)
├── frontend/                       # React + Vite + Tailwind + Radix UI
│   ├── src/
│   │   ├── components/             # PatientList, PatientDetail, Stats panels
│   │   └── lib/utils.ts            # shadcn/ui cn() utility
│   └── package.json
├── tests/
│   ├── test_model.py               # Inference correctness
│   ├── test_schemas.py             # Clinical validation rules
│   ├── test_narrative.py           # Prompt structure + LLM client
│   └── test_safety.py              # Guardrails (22 tests)
├── notebooks/
│   ├── 01_eda.ipynb                # Cohort exploration
│   └── 02_feature_analysis.ipynb   # Feature importance (permutation)
├── models/                         # Model artifacts (gitignored for demo; commit real pkl)
├── data/                           # Processed data (gitignored)
├── logs/                           # Audit + feedback logs (gitignored)
├── setup_demo.py                   # One-command demo setup (synthetic data, no MIMIC needed)
├── run_pipeline.py                 # Full MIMIC-IV pipeline runner
├── retrain_with_feedback.py        # Feedback-driven model retraining
├── finetune_narrative.py           # LoRA narrative fine-tuning pipeline
├── config.yaml                     # Model + app configuration
├── MODEL_CARD.md                   # Model documentation (EU AI Act Annex IV)
├── TRANSPARENCY_LOG.md             # GenAI tool usage disclosure
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
| Narrative improvement | Few-shot + RAG (cosine SHAP similarity) | Quality improves with use, no retraining required |
| Audio feedback | OpenAI Whisper (local) | Voice correction notes, no cloud call |
| EHR integration | HL7 FHIR R4 adapter | Compatible with Epic / Oracle Health (Cerner) |
| React frontend | React + Vite + TypeScript + Tailwind + Radix UI | Modern SPA, proxies to FastAPI backend |
| Backend API | FastAPI + uvicorn | REST API serving React frontend + programmatic access |
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

# Generate synthetic demo data + reuse real model if present, else train demo (~10s)
python setup_demo.py

# Terminal 1 — FastAPI backend
uvicorn src.api.main:app --reload --port 8000
# Terminal 2 — React frontend
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

`setup_demo.py` generates 200 fully synthetic ICU patients (no real patient data) and reuses a real trained model if `models/sepsis_model.pkl` exists, otherwise trains a demo model on synthetic data.

---

## Full Pipeline (with MIMIC-IV access)

```bash
pip install -r requirements.txt

# Configure data paths
cp .env.example .env
# Edit config.yaml -> set icu_path and hosp_path to your MIMIC-IV location

# Run full pipeline
python run_pipeline.py

# Or step by step:
python -m src.data.cohort
python -m src.data.features
python -m src.model.tune        # Optuna (optional, ~30 min)
python -m src.model.train
python -m src.model.evaluate    # AUROC, Brier, subgroup fairness

# Launch React + FastAPI dashboard
uvicorn src.api.main:app --reload --port 8000
# cd frontend && npm run dev
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
