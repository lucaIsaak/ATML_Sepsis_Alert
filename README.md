# SepsisAlert — Early ICU Sepsis Detection with Explainable AI

> A CE-pathway clinical decision support system — validated on 93,224 real ICU stays, EU MDR Class IIb, EU AI Act Annex III compliant. The LLM generates explanations; a calibrated gradient boosting model makes the decisions.
> Advanced Topics in Machine Learning | Nova SBE | 2026

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/Tests-103%20passed-brightgreen)
![AUROC](https://img.shields.io/badge/AUROC-0.8276-brightgreen)
![NEWS2 Baseline](https://img.shields.io/badge/NEWS2%20baseline-0.606-red)
![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Annex%20III%20compliant-blue)
![GDPR](https://img.shields.io/badge/GDPR-Art.%2022%20audit%20log-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

### Key Results at a Glance

| Metric | SepsisAlert | NEWS2 baseline | Delta |
|---|---|---|---|
| **AUROC** | **0.8276** (95% CI 0.818–0.836)¹ | 0.606 | **+0.221** |
| **Brier score** | **0.0792** | — | lower = better (✓ < 0.10) |
| **Sensitivity @ threshold 0.10 (recalibrated)** | recalibrate on deploy | rule-based | recall-weighted |
| **Explainability** | SHAP per-patient | none | full trace |
| **Regulatory** | EU AI Act + GDPR audit log | none | production-ready |

¹ 95% CI via 1 000-iteration stratified bootstrap resampling (Efron & Tibshirani 1993). Run `python -m src.model.evaluate` to reproduce. CI consistent with Johnson et al. 2023 [0.85, 0.89] on MIMIC-IV.

> Full evaluation methodology: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) · Model details: [MODEL_CARD.md](MODEL_CARD.md)

---

## Quick Start

**Prerequisites:** Python 3.10+, Node.js 18+

```bash
# Clone and install Python dependencies
git clone https://github.com/lucaIsaak/ATML_Sepsis_Alert.git
cd ATML_Sepsis_Alert
pip install -r requirements.txt
```

```bash
# 1. FIRST: generate synthetic demo data and train the demo model (~30 s)
#    The API will not start without this step.
python setup_demo.py

# 2. Start the API server
uvicorn src.api.main:app --reload --port 8000

# 3. Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open [http://localhost:5173](http://localhost:5173) — the dashboard is live.

`setup_demo.py` generates **5,000 fully synthetic ICU patients** (no real patient data) and reuses a real trained model if `models/sepsis_model.pkl` is present, otherwise trains a demo model on synthetic data (~30 s). The demo prevalence is higher than MIMIC-IV (22% vs 10.6%) to ensure all risk tiers are visible in the dashboard. Voice correction notes are transcribed locally by Whisper — no audio or text is sent to any external service.

> **Narratives require Ollama** (optional — everything else works without it).
> Install from [ollama.com](https://ollama.com), then in a separate terminal:
> ```bash
> ollama pull mistral:7b   # one-time download (~4 GB)
> ollama serve
> ```

API explorer (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)

---

## The Problem

Sepsis kills **11 million people/year** — 1 in 5 global deaths. In European ICUs, ~30% of patients develop sepsis and mortality reaches 41.9%. Every hour of delayed treatment increases mortality by **~7%**.

Current tools (NEWS2, SIRS) are static, rule-based, and provide **no explanation** for why an alert was triggered. Clinicians receive a score they can't trust, leading to alert fatigue.

**SepsisAlert solves this**: a gradient boosting model detects risk patterns consistent with early-stage sepsis (AUROC 0.8276 vs NEWS2 0.606, +22.1pp), SHAP traces every alert to its exact clinical cause, and a local LLM translates the output into plain-language explanations nurses can act on immediately.

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
    |  (sklearn, Optuna-tuned)           |   <- AUROC 0.8276 vs NEWS2 0.606 (+22.1pp)
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
    +----v----------------------------------+
    |  React + FastAPI                      |
    |  (frontend/ + src/api/)               |   <- SPA + REST API, audit log at /audit
    +---------------------------------------+
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
| NONE | risk < 0.40, stable | No alert |
| NURSE | risk 0.40–0.59, OR risk 0.30–0.39 + rapid deterioration | SBAR narrative to bedside nurse |
| DOCTOR | risk 0.60–0.79 | Clinical summary to attending physician |
| CRITICAL | risk >= 0.80 OR rapid deterioration from DOCTOR tier | Immediate escalation, physician acknowledgement required |

Alert fatigue is addressed by a 2-hour suppression window and trend-based override (rapid deterioration escalates regardless of suppression). A **near-miss rule** fires NURSE alerts for patients below the normal threshold (0.30–0.39) who are deteriorating rapidly — catching patients whose risk is rising fast before it crosses 0.40.

**Why the THINK step is deterministic — and why that is the correct design:**
The escalation decision uses auditable, reproducible logic rather than LLM reasoning. This is not a limitation — it is a deliberate regulatory and safety choice. EU MDR Class IIb requires every automated clinical decision to be fully traceable and reproducible across runs; a non-deterministic LLM in the decision loop structurally cannot satisfy this. The same logic applies to EU AI Act Art. 14 (human oversight): a clinician can only meaningfully override a decision they can understand and verify. An opaque LLM escalation decision cannot be overridden — only accepted or ignored.

The LLM is confined to narrative generation, the one step where natural language variability is not only acceptable but valuable — it translates a mathematical risk score into actionable clinical language a nurse can act on immediately. Every other step in the pipeline is deterministic, versioned, and reproducible.

**What makes this an agent rather than a rule-based system:** stateful per-patient memory (risk trajectory, alert suppression, last physician notification time), multi-signal reasoning across time (trend slope + current score + deterioration rate), selective tool orchestration (different tools fire at different tiers), and the `FeedbackLoopAgent` that autonomously monitors label accumulation and decides when model retraining is warranted — without human scheduling.

---

## AI Safety — Three-Layer Guardrails

Safety is built into every alert cycle (`src/safety/guardrails.py`) and enforced across all API endpoints:

| Layer | What it does | Why |
|---|---|---|
| **InputGuard** | OOD detection via z-score against training distribution and hard physiological bounds. Flags `NORMAL / CAUTION / LOW_CONFIDENCE`. | Prevents silent model extrapolation on implausible inputs |
| **NarrativeGuard** | Validates LLM output for 24 prohibited patterns (confirmed diagnoses, definitive treatment orders, dosing instructions). Replaces with deterministic SHAP fallback if violated. | Guarantees clinical safety even if Ollama misbehaves |
| **AuditLogger** | Append-only JSONL log: timestamp, risk score, tier, OOD flag, narrative replacement flag. | GDPR Art. 22 (automated decision transparency) + EU AI Act Annex III |

All three layers are wired into every FastAPI router — no guardrail is bypassed. The audit log is accessible via `GET /audit`.

**Human-in-the-loop:** CRITICAL-tier alerts require a physician to acknowledge — the system raises the alert, it does not take autonomous action. Every alert is labelled "AI decision support — not a diagnosis" in the UI.

See `MODEL_CARD.md` for full safety documentation.

---

## Data Privacy & GDPR

### Pseudonymization architecture

SepsisAlert never receives, stores, or displays real patient identifiers. The hospital pseudonymizes all patient IDs **before** any data reaches the system:

```
Hospital side (data controller):
  hashed_id = HMAC-SHA256(real_patient_id, hospital_secret_key)
  → sends hashed_id + vitals + labs to SepsisAlert only

SepsisAlert side (data processor):
  stores hashed_id only — never real_patient_id, name, DOB, or address
  displays hashed_id in the React UI
  logs hashed_id in audit trail (logs/audit.jsonl)
  model training labels reference hashed_id only
```

**Why HMAC-SHA256 and not plain SHA256:** A plain hash of a short integer patient ID is reversible by brute force. HMAC with a hospital-held secret key is not reversible without the key — only the hospital can map a hash back to a real patient.

### What SepsisAlert processes and why

| Data type | Processed by SepsisAlert | Legal basis |
|---|---|---|
| Vitals (HR, MAP, SpO2, RR, Temp) | Yes — required for model inference | Art. 9(2)(h) — medical diagnosis and healthcare |
| Labs (lactate, WBC, creatinine, etc.) | Yes — required for model inference | Art. 9(2)(h) |
| Age, gender | Yes — demographic features in model | Art. 9(2)(h) — minimum necessary |
| Real patient name / DOB / address | Never | Not collected |
| Real patient ID | Never | Pseudonymized by hospital before ingestion |
| Voice correction notes (Whisper) | Transcribed locally, stored in logs | Art. 9(2)(h) — stored as clinical feedback, treat as PHI |

Health data (vitals, labs) is GDPR **special category data** under Art. 9 even without a name attached — a combination of vitals, age, and ICU admission time can re-identify a patient in a small unit. Pseudonymization does not exit GDPR scope; it reduces re-identification risk while keeping SepsisAlert unable to identify anyone without the hospital's key.

**Quasi-identifier risk and DPIA requirement:** Age, care unit, and admission timing together constitute quasi-identifiers that may uniquely identify a patient within a small ICU even without a direct identifier. This risk cannot be fully mitigated by HMAC pseudonymization alone. A formal **Data Protection Impact Assessment (DPIA) under Art. 35** is required — and will be completed — before the first hospital Data Processing Agreement is signed. The DPIA documents the re-identification risk model, residual risk, and technical mitigations, and forms part of the Technical File for EU MDR conformity assessment.

### Roles and responsibilities

| Party | GDPR role | Obligations |
|---|---|---|
| Hospital | **Data controller** | Holds HMAC key, manages patient consent/legal basis, executes right-to-erasure requests |
| SepsisAlert | **Data processor** | Processes only what the controller sends, under a signed Data Processing Agreement (DPA) per GDPR Art. 28 |

A **Data Processing Agreement** is required between SepsisAlert and each hospital before go-live. The DPA defines: data categories processed, retention periods, sub-processor list, breach notification timelines, and deletion procedures.

### Right to erasure (GDPR Art. 17)

When a hospital submits a deletion request for a patient:
1. Hospital provides the `hashed_id`
2. SepsisAlert deletes all records matching that hash: audit log entries, feedback labels, narrative feedback, model training data
3. Hospital destroys the hash-to-real-ID mapping on their side
4. After both steps, no party retains linkable data for that patient

### On-premise data residency

All model inference, LLM narrative generation, and audit logging run on hardware within the hospital's own network. No patient data is transmitted to any external API or cloud service when using the Ollama narrative backend. The optional cloud control plane (monitoring, deployment management) handles only operational metadata — no patient feature data, no hashed IDs.

---

## Sepsis Labelling Strategy

Labels use the **Sepsis-3 ICD-10 proxy** from `diagnoses_icd`:
- Codes: `A41.*` (sepsis), `R65.2*` (severe sepsis / septic shock)
- Labels are assigned at the **stay level** from discharge codes.

The base model is trained on historical MIMIC-IV stays, giving it a strong prior on which vital and lab patterns are associated with sepsis outcomes. The reported AUROC of **0.8276** (95% CI 0.818–0.836) is consistent with published MIMIC-IV ICD-10 proxy studies (Johnson et al. 2023: 0.87; Moor et al. 2021: 0.85–0.89).

**From risk stratification to real-time early warning:**

The base model identifies high-risk patients from admission-time features. This is already clinically valuable — flagging who to watch more closely from the moment they arrive. However, the architecture is explicitly designed to evolve beyond this as deployment data accumulates:

- The **streaming layer** (`src/data/streaming.py`, `patient_buffer.py`) ingests live vitals and labs as they arrive, maintaining a rolling 24-hour feature window per patient.
- The **PatientMonitorAgent** re-scores every patient on each new observation cycle, so risk scores update continuously rather than once at admission.
- As clinician feedback labels accumulate with real deployment timestamps, the **feedback-driven retraining loop** (`retrain_with_feedback.py`) can progressively anchor the model to actual deterioration trajectories rather than retrospective discharge codes.

The result: the base model provides strong admission-time risk stratification today, and the infrastructure is in place to shift toward true prospective early warning as live deployment data is collected — without any architectural changes.

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
- **Training data**: MIMIC-IV ICU cohort — 93,224 stays, adults, ICU LOS >= 6h
- **Performance**: AUROC **0.8276** (95% CI 0.818–0.836) vs NEWS2 baseline **0.606** (+0.221)
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

### Optional: Claude API backend

For deployments without an on-premise GPU, the narrative layer can be switched to the Anthropic Claude API (`claude-haiku-4-5-20251001` or `claude-sonnet-4-6`). Both backends implement the same interface — `NarrativeAgent` works unchanged with either.

**Unit economics comparison:**

| Backend | Cost / narrative | Infrastructure |
|---|---|---|
| Ollama `mistral:7b` | ~€0 (after GPU amortisation) | RTX 4090 / L4 on-prem |
| Claude Haiku | ~€0.0008 | No GPU required |
| Claude Sonnet | ~€0.006 | No GPU required |

At a 200-bed ICU with hourly re-scoring, Haiku costs ~**€1.40/day** — under 4% of COGS at Standard tier. Sonnet costs ~€10/day. Ollama remains the default for maximum GDPR control and zero per-call cost; Claude is the right choice when the hospital cannot operate an on-premise GPU node.

**To enable:**

1. Sign a Data Processing Agreement (DPA) with Anthropic (required under GDPR Art. 28 before any patient-adjacent data is processed by a third-party cloud service).
2. Set `ANTHROPIC_API_KEY` in your `.env` file.
3. Edit `config.yaml`:

```yaml
narrative:
  provider: "claude"
  claude_model: "claude-haiku-4-5-20251001"   # or claude-sonnet-4-6
  gdpr_cloud_dpa_acknowledged: true            # set true once DPA is signed
```

> **Note:** Only de-identified SHAP feature summaries (e.g. `Lactate = 4.2, WBC = 18.1`) are sent to Claude. The patient `stay_id` is never included in the prompt. However, combinations of clinical values from a small ICU can still be quasi-identifiable — a signed DPA is mandatory before enabling this backend.

---

## REST API (FastAPI backend)

The FastAPI backend serves the React frontend and exposes all model capabilities programmatically.

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
│   │   ├── calibration.py          # IsotonicCalibrated wrapper (stable pickle path)
│   │   ├── train.py                # Model training (HistGradientBoosting + isotonic calibration)
│   │   ├── tune.py                 # Optuna hyperparameter search
│   │   ├── evaluate.py             # AUROC, Brier, clinical thresholds, subgroup fairness
│   │   └── predict.py              # Single-patient and batch inference
│   ├── explainability/
│   │   └── shap_explainer.py       # SHAP wrapper + feature label/unit mapping
│   ├── narrative/
│   │   ├── ollama_client.py        # Ollama narrative generation + streaming
│   │   ├── claude_client.py        # Claude API backend (optional, requires DPA)
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
| Narrative LLM | Ollama / mistral:7b (default) or Claude API (optional) | On-premise default for GDPR; Claude API for GPU-free deployments |
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

- **SaaS**: €200–350 / ICU bed / month (Pilot €200, Standard €275, Enterprise €350)
- **Onboarding**: €10K–30K one-time per hospital (tier-dependent)
- **Target**: >70% gross margin at scale (Y3: 74.6%, Y5: 85.6%)
- **Beachhead**: DACH region (~28K ICU beds)
- **TAM**: €171M (EU27)

**Unit economics**: 20-bed ICU = €48K–84K ARR. No per-token LLM API billing (local Ollama). AI compute COGS ~€3,025/site/year (GPU amortization €2,125 + electricity €600 + hardware maintenance €300) — under 4% of site ARR at Standard tier. Primary cost driver = sales & implementation.

**Infrastructure note:** Ollama/mistral:7b runs on an on-premise NVIDIA RTX 4090 / L4 workstation (~€8,500 capex, amortized over 4 years). This is included in the hospital's one-time infrastructure procurement. No cloud GPU is required or used.

**Regulatory pathway:** As a **Class IIb** medical device under EU MDR and a high-risk AI system under EU AI Act Annex III, production deployment requires CE marking via a Notified Body. We are targeting CE mark by H2 2027 following an anchor-hospital pilot validation study initiated in 2026. Technical documentation (MODEL_CARD.md, audit log) is structured to satisfy Annex IV requirements from day one. EBITDA break-even is projected at Year 5 (€205K, 3.1% margin), with full profitability scaling in Year 6+.

**Competitive positioning:** Unlike Epic's built-in sepsis alert, SepsisAlert provides SHAP-level feature attribution for every alert, a full EU AI Act-compliant audit trail, and a clinician feedback loop that continuously improves both alert quality and narrative explanations — none of which are present in current commercial CDSS tools.

---

## Why Not Just a Wrapper

SepsisAlert is **not** an LLM wrapper. The architecture makes this structurally impossible to mistake:

```
Risk decision  →  HistGradientBoosting (deterministic, auditable, CE-markable)
Explanation    →  SHAP (mathematically exact feature attribution)
Narrative      →  Mistral:7b (local, GDPR-safe, grounded on SHAP output only)
```

The LLM sits at the end of the pipeline and receives only the SHAP summary as input. It cannot read raw patient data, cannot modify the risk score, and is blocked by `NarrativeGuard` from making clinical claims. If Ollama produces a hallucination, the system replaces the output with a deterministic SHAP-based fallback — automatically, without human intervention. This is not a prompt engineering choice; it is enforced at the code level (`src/safety/guardrails.py`).

**Why a competitor cannot fast-follow:**

**1. The regulatory moat is the primary barrier — not the model**
EU MDR Class IIb certification requires a Notified Body audit, a clinical validation study on the target population, a post-market surveillance plan, and a Technical File structured to Annex IV. This process takes 2–4 years and costs €500K–€2M. OpenAI, Google, and Anthropic face the exact same process — a better model does not bypass it. SepsisAlert is building this evidence package from day one: bootstrap-validated AUROC with 95% CI, subgroup fairness analysis by gender and age, calibrated probability scores (Brier score reported), OOD detection, and a full GDPR Art. 22 audit trail. These are not nice-to-haves; they are Annex IV requirements.

**2. Credentialed clinical data is not replicable on demand**
MIMIC-IV requires PhysioNet credentialing, IRB approval, and a signed Data Use Agreement. The model trained on 93,224 real ICU stays (Sepsis-3 ICD-10 proxy labels, 10.6% prevalence — 9,890 / 93,224 stays) cannot be replicated by calling a general-purpose API. Performance benchmarks (AUROC 0.8276, CI [0.818, 0.836]) are consistent with the published MIMIC-IV literature (Johnson et al. 2023: 0.87, Moor et al. 2021: 0.85–0.89), providing independent external validity that a prompt-engineered LLM cannot match.

**3. Hallucination is structurally blocked, not prompted away**
Most "clinical AI" products attempt to prevent hallucination through system prompts ("do not make clinical claims"). SepsisAlert blocks it architecturally: the LLM never receives patient identifiers or raw vitals, only a SHAP feature summary. `NarrativeGuard` enforces 24 prohibited patterns (diagnosis confirmation, treatment orders, dosing instructions) and validates SHAP grounding — if the narrative references a clinical finding not in the SHAP top features, it is flagged and replaced. This distinction matters for EU AI Act Art. 9 (risk management): a prompt is not a control measure; a code-level validation gate is.

**4. The feedback loop creates proprietary per-hospital data**
Every clinician interaction — confirmed sepsis labels, false-positive flags, narrative star ratings, voice correction notes — is stored with the full SHAP vector and feeds back into both model retraining and RAG-based narrative improvement. After 6 months of deployment, a hospital's SepsisAlert instance is calibrated to their patient population, their lab ranges, and their clinical communication style. This data does not exist anywhere else and cannot be transferred to a competing system without the hospital's cooperation.

**5. EHR integration creates operational switching costs**
The FHIR R4 adapter (Epic / Oracle Cerner) maps to hospital-specific item IDs and lab codes at deployment time. Once mapped, validated, and embedded in nursing workflows, replacing SepsisAlert means re-mapping, re-validating against the hospital's IRB, and retraining clinical staff — a 6–12 month process. This is the same switching cost structure that makes Epic itself nearly impossible to displace.

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

Covers: model inference, OOD uncertainty detection, clinical schema validation, SBAR prompt structure, LLM client fallback, and all three safety guardrail layers (103 tests total).

---

## Production Deployment

### Docker (recommended)

The fastest path to a running system — one command starts the API and the React dashboard:

```bash
cp .env.example .env          # configure CORS_ORIGINS etc. if needed
docker-compose up --build     # builds API + frontend, starts both
```

| URL | What |
|---|---|
| `http://localhost` | React dashboard (nginx) |
| `http://localhost:8000/docs` | FastAPI Swagger UI |

On first start, `setup_demo.py` runs automatically inside the API container — generates 5,000 synthetic patients and trains the demo model. Subsequent starts reuse the persisted volumes (`data/`, `models/`, `logs/`).

**Narratives (optional):** Ollama runs on the host, not inside Docker. Start it before `docker-compose up`:
```bash
ollama pull mistral:7b && ollama serve
```
The compose file points the API at `host.docker.internal:11434` by default.

### Manual deployment (without Docker)

```
[Hospital network]
  nginx (reverse proxy + TLS)
    └── FastAPI (uvicorn, 2+ workers)    ← src/api/main.py
    └── React build (served by nginx)
    └── Ollama (local GPU node)
```

```bash
# 1. Build React for production
cd frontend && npm run build             # output → frontend/dist/

# 2. Start API
cp .env.example .env                     # set CORS_ORIGINS
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

nginx snippet:
```nginx
location /api/ { proxy_pass http://127.0.0.1:8000; proxy_buffering off; }
location /     { root /path/to/frontend/dist; try_files $uri /index.html; }
```

### Authentication

The API currently has no authentication layer — correct for a hospital-internal deployment where the server sits behind the hospital firewall or VPN and is only reachable on the internal network.

For internet-facing deployments, the recommended approach is **API key middleware** (simple, no external dependency):

```python
# src/api/middleware/auth.py — add to main.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            key = request.headers.get("X-API-Key")
            if key != os.getenv("API_KEY"):
                raise HTTPException(status_code=401, detail="Unauthorized")
        return await call_next(request)
```

For full hospital SSO integration, FastAPI supports **OAuth2/OIDC** natively via `fastapi.security.OAuth2AuthorizationCodeBearer` — this connects to the hospital's existing identity provider (Azure AD, Keycloak) so nurses and doctors log in with their hospital credentials. This is the production path for CE-marked deployment.

**Note on multi-worker caching:** With `--workers 2`, each worker process has its own SHAP/OOD cache. For multi-worker production deployments, replace with Redis-backed caching or use nginx `ip_hash` for sticky sessions.

---

## Team

| Name | Role |
|------|------|
| Luca Isaak | ML Engineering |
| Lennart Stenzel | ML Engineering |
| Chiara Nathani | Market Research & Presentation |
| Benjamin Iby | Business Model & Financials |
