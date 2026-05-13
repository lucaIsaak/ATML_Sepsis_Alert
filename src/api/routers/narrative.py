"""
Narrative routes.

GET  /narrative/models  — list available Ollama models
POST /narrative/stream  — stream a clinical narrative for a patient
"""

from __future__ import annotations

import asyncio
import re
import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from src.api.routers.patients import _shap_cache, _ood_cache, _risk_label
from src.explainability.shap_explainer import SHAPExplanation, format_for_narrative
from src.narrative.ollama_client import OllamaClient
from src.narrative.narrative_agent import NarrativeAgent
from src.data.narrative_feedback import load_few_shot_examples, find_similar_narratives
from src.safety.guardrails import NarrativeGuard, AuditLogger, OODResult, NarrativeResult

_audit_logger = AuditLogger(log_path="logs/audit.jsonl")
_narrative_guard = NarrativeGuard()

router = APIRouter()


class StreamRequest(BaseModel):
    stay_id: int
    model_name: str

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,99}$", v):
            raise ValueError("model_name contains invalid characters")
        return v


@router.get("/narrative/models")
async def list_models(request: Request) -> list[str]:
    """Return list of available Ollama model name strings."""
    cfg = request.app.state.cfg
    base_url = cfg["narrative"]["ollama_base_url"]
    try:
        # Run blocking requests.get in thread pool — keeps the event loop free
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: requests.get(f"{base_url}/api/tags", timeout=5)
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models]
    except requests.exceptions.RequestException:
        # Return empty list — frontend shows "Ollama not available"
        return []


@router.post("/narrative/stream")
async def stream_narrative(body: StreamRequest, request: Request):
    """Stream a clinical narrative for a patient using Ollama."""
    predictions = request.app.state.predictions
    artifact = request.app.state.artifact
    cfg = request.app.state.cfg

    stay_id = body.stay_id
    model_name = body.model_name

    row_df = predictions[predictions["stay_id"] == stay_id]
    if row_df.empty:
        raise HTTPException(status_code=404, detail=f"Patient {stay_id} not found")

    row = row_df.iloc[0]
    risk_score = float(row["risk_score"])

    # Ensure SHAP is computed (may already be cached from patients router)
    if stay_id not in _shap_cache:
        from src.api.routers.patients import _get_explainer  # noqa: PLC0415
        from src.explainability.shap_explainer import explain_patient  # noqa: PLC0415

        feature_cols = artifact["feature_cols"]
        feature_vector = row[feature_cols].values.astype(float)
        explainer = _get_explainer(artifact, predictions)
        explanation = explain_patient(
            explainer=explainer,
            feature_vector=feature_vector,
            feature_names=list(feature_cols),
            risk_score=risk_score,
            stay_id=str(stay_id),
            top_n=len(feature_cols),
        )
        _shap_cache[stay_id] = explanation.top_features

    top_features = _shap_cache[stay_id][:10]

    # Build SHAPExplanation for the narrative
    explanation = SHAPExplanation(
        stay_id=str(stay_id),
        risk_score=risk_score,
        risk_label=_risk_label(risk_score),
        top_features=top_features,
        base_value=0.0,
    )

    # Build few-shot + RAG context
    shap_vector = {f["feature"]: f["shap"] for f in top_features}
    few_shot = load_few_shot_examples(min_rating=4, max_examples=2, model_used=model_name)
    similar = find_similar_narratives(
        current_shap_vector=shap_vector,
        top_n=1,
        min_rating=4,
        model_used=model_name,
    )

    context_parts = []
    for ex in few_shot:
        context_parts.append(
            f"EXAMPLE (rated {ex['rating']}/5):\n"
            f"Input: {ex['shap_summary']}\n"
            f"Output: {ex['narrative_text']}"
        )
    for sim in similar:
        context_parts.append(
            f"[Similar patient, rated {sim['rating']}/5, "
            f"similarity {sim['similarity']:.2f}]\n{sim['narrative_text']}"
        )

    context = "\n\n".join(context_parts) if context_parts else ""

    # Override model in cfg
    cfg_copy = dict(cfg)
    cfg_copy["narrative"] = dict(cfg["narrative"])
    cfg_copy["narrative"]["ollama_model"] = model_name

    # GDPR hard gate — patient data must never leave the hospital server.
    # Non-local providers (claude, huggingface) are explicitly blocked.
    if cfg_copy["narrative"].get("provider", "ollama") != "ollama":
        raise HTTPException(
            status_code=403,
            detail=(
                "Non-local narrative providers are disabled for GDPR compliance. "
                "Patient data must not leave the hospital server. "
                "Set narrative.provider to 'ollama' in config.yaml."
            ),
        )

    client = OllamaClient(cfg_copy)
    agent  = NarrativeAgent(client)
    shap_summary = format_for_narrative(explanation)

    # Build flat feature dict from predictions row for the narrative agent
    feature_cols = request.app.state.artifact["feature_cols"]
    features_dict = {col: row.get(col) for col in feature_cols}

    # Determine escalation tier for audit log
    if risk_score >= 0.8:
        tier = "CRITICAL"
    elif risk_score >= 0.6:
        tier = "DOCTOR"
    elif risk_score >= 0.4:
        tier = "NURSE"
    else:
        tier = "NONE"

    # Alert context — tells the agent whether this is the first alert
    alert_context = "First alert for this patient in current session"

    # Build OODResult from cache (populated by patients router) or default NORMAL
    cached_ood = _ood_cache.get(stay_id, {})
    ood_result = OODResult(
        is_ood=cached_ood.get("ood_flag") != "NORMAL",
        n_outlier_features=len(cached_ood.get("outlier_features", [])),
        outlier_features=cached_ood.get("outlier_features", []),
        confidence_flag=cached_ood.get("ood_flag", "NORMAL"),
    )

    async def _generate():
        # Open the HTTP stream immediately so the frontend shows a loading state
        # rather than waiting 20-30 s for the first byte from Ollama.
        yield ""

        # 1. Collect full agent output (required for NarrativeGuard validation)
        try:
            chunks = list(agent.stream_generate(
                explanation,
                features=features_dict,
                few_shot_context=context,
                alert_context=alert_context,
            ))
        except requests.exceptions.ConnectionError:
            yield (
                "[Narrative unavailable] Ollama is not running.\n"
                "Start it with:  ollama serve\n"
                "Pull the model if needed:  ollama pull mistral:7b"
            )
            return
        except Exception as exc:
            yield f"[Narrative unavailable] Generation failed: {exc}"
            return
        full_text = "".join(chunks)

        # 2. Validate with NarrativeGuard (Layer 2 safety)
        nar_result = _narrative_guard.validate(full_text, shap_summary=shap_summary)

        # 3. Audit log (Layer 3 — GDPR Art. 22 / EU AI Act)
        _audit_logger.log_alert(
            stay_id=str(stay_id),
            risk_score=risk_score,
            tier=tier,
            top_features=[{"feature": f["feature"], "shap": f["shap"]} for f in top_features],
            ood_result=ood_result,
            narrative_result=nar_result,
        )

        # 4. Stream validated (or fallback) text in small chunks for typewriter effect.
        # asyncio.sleep gives the event loop a chance to flush each chunk to the
        # client before yielding the next — without this all chunks arrive in one burst.
        validated = nar_result.text
        chunk_size = 8
        for i in range(0, len(validated), chunk_size):
            yield validated[i:i + chunk_size]
            await asyncio.sleep(0.02)

    return StreamingResponse(
        _generate(),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"},
    )
