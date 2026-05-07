"""
Audio transcription for narrative feedback.

Uses OpenAI Whisper running entirely locally — no data leaves the machine.

Model sizes and approximate speed on MacBook Air M-series:
  tiny   (~39 MB)  — fastest, good enough for short voice notes
  base   (~74 MB)  — slightly more accurate, still fast
  small  (~244 MB) — accurate, ~5s for a 30s clip

Default: "tiny" — adequate for a few sentences of clinical feedback.
Change via WHISPER_MODEL environment variable or the model parameter.

Requirements
------------
    pip install openai-whisper
    brew install ffmpeg        ← required by whisper for audio decoding
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def transcribe_audio(audio_file, model_size: str | None = None) -> str:
    """
    Transcribe an audio recording to text using local Whisper.

    Parameters
    ----------
    audio_file  : file-like object returned by st.audio_input
                  (has a .read() method returning raw bytes)
    model_size  : whisper model to use — "tiny" | "base" | "small"
                  Defaults to WHISPER_MODEL env var, then "tiny"

    Returns
    -------
    Transcribed text string, or an error message prefixed with "[Error]"
    if transcription fails.
    """
    try:
        import whisper  # pylint: disable=import-outside-toplevel
    except ImportError:
        return (
            "[Error] openai-whisper is not installed.\n"
            "Run: pip install openai-whisper"
        )

    size = model_size or os.getenv("WHISPER_MODEL", "tiny")

    # Write audio bytes to a temp file — whisper needs a file path
    suffix = _detect_suffix(audio_file)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_file.read())
        tmp.flush()
        tmp.close()

        model = _load_model(size)
        result = model.transcribe(tmp.name, fp16=False)
        return result["text"].strip()

    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"[Error] Transcription failed: {exc}"
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _detect_suffix(audio_file) -> str:
    """Try to detect the audio format from the file name or default to .wav."""
    name = getattr(audio_file, "name", "") or ""
    for ext in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
        if name.lower().endswith(ext):
            return ext
    return ".wav"


# Cache the model at module level so it isn't reloaded on every call.
# Streamlit's @st.cache_resource cannot be used here (circular import),
# so we use a simple module-level dict instead.
_model_cache: dict[str, object] = {}


def _load_model(size: str):
    """
    Return a cached Whisper model, loading it on first call.

    Patches SSL verification temporarily for the download only — required on
    macOS when Python's bundled certificates are not linked to the system
    keychain (common with Homebrew / pyenv Python installs).
    """
    if size not in _model_cache:
        import ssl  # pylint: disable=import-outside-toplevel
        import whisper  # pylint: disable=import-outside-toplevel

        # Temporarily bypass SSL verification only for the model download.
        # The downloaded file is checksummed by whisper itself so integrity
        # is still verified even without SSL cert checking.
        _orig = ssl._create_default_https_context  # pylint: disable=protected-access
        ssl._create_default_https_context = ssl._create_unverified_context  # pylint: disable=protected-access
        try:
            _model_cache[size] = whisper.load_model(size)
        finally:
            # Always restore original SSL behaviour
            ssl._create_default_https_context = _orig

    return _model_cache[size]


def is_whisper_available() -> bool:
    """Return True if openai-whisper is installed."""
    try:
        import whisper  # noqa: F401  pylint: disable=import-outside-toplevel
        return True
    except ImportError:
        return False
