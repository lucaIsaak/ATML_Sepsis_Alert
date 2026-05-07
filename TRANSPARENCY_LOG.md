# GenAI Transparency Log — SepsisAlert

> Required disclosure per Nova SBE ATML 2026 course policy.
> Documents all use of AI tools during ideation, development, and presentation.

---

## Summary

| Tool | Primary use | Approximate contribution |
|---|---|---|
| Claude (Anthropic) | Backend code generation, debugging, architecture design | High |
| ChatGPT / GPT-4 | Literature review, business model framing | Low–Medium |
| GitHub Copilot | In-editor code completion | Low |

---

## Detailed Usage Log

### 1. Ideation and Problem Scoping

**Tool:** ChatGPT / GPT-4
**Date range:** Early project phase
**Prompts / tasks:**
- "What are the most impactful unsolved problems in ICU care that AI could address?"
- "Compare SIRS, qSOFA, and NEWS2 as sepsis screening tools — what are their limitations?"
- Summarising literature on MIMIC-IV sepsis datasets and benchmark AUROC scores

**Human contribution:** Final problem selection (sepsis early warning), decision to use MIMIC-IV, decision to focus on SHAP explainability as the differentiating feature.

---

### 2. Data Pipeline (cohort.py, features.py)

**Tool:** Claude (claude-sonnet-4-6)
**Tasks:**
- DuckDB SQL queries for cohort extraction from MIMIC-IV compressed CSVs
- Feature engineering: rolling 24h windows, mean/min/max/last/trend statistics
- Trend feature addition (`np.polyfit` slope per vital/lab)
- Pylint 10/10 compliance fixes across all src/ files

**Human contribution:** Sepsis-3 label strategy (ICD-10 A41.x proxy), feature selection (which MIMIC item IDs to include), review and testing of all generated code.

---

### 3. Model Training and Evaluation (train.py, tune.py, evaluate.py)

**Tool:** Claude (claude-sonnet-4-6)
**Tasks:**
- HistGradientBoostingClassifier training pipeline
- Optuna hyperparameter search implementation (50 trials, 5-fold CV)
- evaluate.py: Brier score, sensitivity/specificity at clinical thresholds, subgroup AUROC by gender and age
- Identification and fix of data leakage (evaluate.py previously evaluated on full dataset including training data)

**Human contribution:** Decision to use HistGradientBoosting over LightGBM/XGBoost (no native deps, handles NaN natively), interpretation of evaluation results, decision to fix data leakage.

---

### 4. AI Safety Module (src/safety/guardrails.py)

**Tool:** Claude (claude-sonnet-4-6)
**Tasks:**
- InputGuard: z-score OOD detection + physiological plausibility bounds
- NarrativeGuard: regex-based prohibited-phrase detection + safe fallback generation
- AuditLogger: append-only JSONL audit trail (GDPR / EU AI Act compliance)
- Integration into PatientMonitorAgent ReAct loop
- 22-test test suite (tests/test_safety.py)

**Human contribution:** Safety requirements specification, review of prohibited phrases list, verification that fallback narrative is clinically appropriate.

---

### 5. Agent Architecture (monitor_agent.py)

**Tool:** Claude (claude-sonnet-4-6)
**Tasks:**
- ReAct-pattern PatientMonitorAgent class
- 4-tier escalation logic (NONE / NURSE / DOCTOR / CRITICAL)
- Alert fatigue suppression (2h window, trend-based override)
- Per-patient memory (risk trajectory, last physician notification time)

**Human contribution:** Clinical validation of escalation thresholds, decision to add rapid-deterioration detection as a trigger.

---

### 6. Explainability and Narrative (shap_explainer.py, prompts.py, ollama_client.py)

**Tool:** Claude (claude-sonnet-4-6)
**Tasks:**
- SHAP wrapper for HistGradientBoosting
- SBAR and physician-summary prompt templates
- Anti-hallucination rules in system prompts
- Clinical reference threshold enrichment (enrich_shap_summary)
- OllamaClient with fallback chain (Ollama → Claude API → HuggingFace)

**Human contribution:** SBAR format selection (clinical standard), review of prompt structure with reference to ICU handover literature, decision to include explicit "never say diagnosed with" rules.

---

### 7. Presentation and Documentation

**Tool:** Claude (claude-sonnet-4-6)
**Tasks:**
- README.md: architecture diagram, data sources table, model performance section, labelling limitation documentation
- MODEL_CARD.md: full model card following Mitchell et al. (2019) + EU AI Act Annex IV format
- run_presentation_demo.py: five-scenario self-contained demo
- scripts/generate_demo_data.py: synthetic MIMIC-IV-like data (no real patient data)

**Tool:** ChatGPT / GPT-4
**Tasks:**
- Business model section (SaaS pricing, TAM, unit economics)
- Market research (DACH ICU bed count, competitor landscape)

**Human contribution:** All numbers verified against public sources (Eurostat ICU capacity, Destatis hospital statistics). Business model reviewed by team.

---

### 8. Debugging

**Tool:** Claude (claude-sonnet-4-6)
**Tasks:**
- `AttributeError: HistGradientBoostingClassifier has no attribute feature_importances_` → replaced with permutation importance in notebook
- Dashboard `KeyError: 'age' not in index` → defensive column selection
- Streaming demo firing 0 alerts → replaced with self-contained presentation demo

**Human contribution:** Identified the bugs, provided error tracebacks, validated fixes.

---

## What AI Did NOT Do

- AI did not access or read any real MIMIC-IV patient data (all data processing was designed and reviewed by team before execution).
- AI did not make architectural decisions autonomously — all suggestions were reviewed and either accepted, modified, or rejected.
- AI did not write the business plan or financial model (beyond prompt assistance).
- AI did not write the presentation slides.

---

## Reflection

The use of Claude as a coding assistant substantially accelerated implementation, particularly for boilerplate (Pydantic schemas, DuckDB queries, test suites) and for identifying best practices (model cards, audit logging, EU AI Act compliance patterns). The core domain decisions — sepsis-3 labelling, feature engineering choices, escalation thresholds, clinical prompt design — required human clinical knowledge and were made by the team.

The biggest risk of AI-assisted development was over-engineering: AI tends to add complexity. We actively pruned features and abstractions that were generated but not necessary. The final codebase reflects deliberate choices, not everything the AI suggested.
