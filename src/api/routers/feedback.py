"""
Feedback routes.

POST /feedback/clinical            — save clinical (confirm/flag) feedback
GET  /feedback/clinical/{stay_id}  — get clinical feedback for patient
POST /feedback/narrative           — save narrative rating + correction note
POST /feedback/transcribe          — transcribe audio to text via Whisper
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel

from src.data.feedback import save_feedback, get_feedback_for_patient
from src.data.narrative_feedback import save_narrative_feedback
from src.narrative.transcribe import transcribe_audio, is_whisper_available

router = APIRouter()


class ClinicalFeedbackRequest(BaseModel):
    stay_id: int
    feedback_type: str  # "confirmed_sepsis" | "flagged_wrong"
    risk_score: float


class NarrativeFeedbackRequest(BaseModel):
    stay_id: int
    rating: int
    correction_note: str = ""
    narrative_text: str
    model_used: str


@router.post("/feedback/clinical")
async def save_clinical_feedback(body: ClinicalFeedbackRequest):
    """Save a clinical feedback label for a patient."""
    if body.feedback_type not in {"confirmed_sepsis", "flagged_wrong"}:
        raise HTTPException(
            status_code=422,
            detail="feedback_type must be 'confirmed_sepsis' or 'flagged_wrong'",
        )
    save_feedback(
        stay_id=body.stay_id,
        feedback_type=body.feedback_type,
        risk_score=body.risk_score,
    )
    return {"status": "saved"}


@router.get("/feedback/clinical/{stay_id}")
async def get_clinical_feedback(stay_id: int):
    """Return existing clinical feedback for a patient, or null."""
    result = get_feedback_for_patient(stay_id)
    if result is None:
        return None
    return {
        "feedback_type": result["feedback_type"],
        "risk_score": float(result["risk_score"]),
    }


@router.post("/feedback/narrative")
async def save_narrative_fb(body: NarrativeFeedbackRequest, request: Request):
    """Save a narrative quality rating and optional correction note."""
    # Build SHAP summary from cache if available
    from src.api.routers.patients import _shap_cache  # noqa: PLC0415
    from src.explainability.shap_explainer import format_for_narrative, SHAPExplanation  # noqa: PLC0415

    shap_vector: dict = {}
    shap_summary = ""

    if body.stay_id in _shap_cache:
        top_features = _shap_cache[body.stay_id]
        shap_vector = {f["feature"]: f["shap"] for f in top_features}

        # Build a minimal SHAPExplanation just for format_for_narrative
        predictions = request.app.state.predictions
        row_df = predictions[predictions["stay_id"] == body.stay_id]
        risk_score = float(row_df.iloc[0]["risk_score"]) if not row_df.empty else 0.0
        risk_label = "HIGH" if risk_score >= 0.6 else "MODERATE" if risk_score >= 0.4 else "LOW"

        explanation = SHAPExplanation(
            stay_id=str(body.stay_id),
            risk_score=risk_score,
            risk_label=risk_label,
            top_features=top_features[:10],
            base_value=0.0,
        )
        shap_summary = format_for_narrative(explanation)

    if not 1 <= body.rating <= 5:
        raise HTTPException(status_code=422, detail="Rating must be between 1 and 5")

    save_narrative_feedback(
        stay_id=body.stay_id,
        rating=body.rating,
        correction_note=body.correction_note,
        narrative_text=body.narrative_text,
        shap_summary=shap_summary,
        model_used=body.model_used,
        shap_vector=shap_vector,
    )
    return {"status": "saved"}


@router.get("/feedback/whisper-status")
async def get_whisper_status():
    """Return whether Whisper is installed and available for transcription."""
    available = is_whisper_available()
    return {
        "available": available,
        "message": "Whisper ready" if available else "openai-whisper not installed. Run: pip install openai-whisper",
    }


@router.post("/feedback/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file to text using Whisper."""
    if not is_whisper_available():
        raise HTTPException(
            status_code=503,
            detail="openai-whisper not installed. Run: pip install openai-whisper",
        )

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Create a file-like object with a name attribute for transcribe_audio
    class _NamedBytesIO:
        def __init__(self, data: bytes, name: str):
            import io  # noqa: PLC0415
            self._buf = io.BytesIO(data)
            self.name = name

        def read(self):
            return self._buf.read()

    audio_obj = _NamedBytesIO(content, f"audio{suffix}")
    text = transcribe_audio(audio_obj)

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)

    return {"text": text}
