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


TRANSCRIPTION_TIMEOUT = 60   # seconds — fail fast rather than hang forever


def transcribe_audio(audio_file, model_size: str | None = None) -> str:
    """
    Transcribe an audio recording to text using local Whisper.

    Runs in a background thread with a hard timeout so the dashboard
    never hangs indefinitely.

    Parameters
    ----------
    audio_file  : file-like object returned by st.audio_input
    model_size  : "tiny" | "base" | "small"  (default: "tiny")

    Returns
    -------
    Transcribed text, or "[Error] ..." message on failure.
    """
    try:
        import whisper  # pylint: disable=import-outside-toplevel
    except ImportError:
        return "[Error] openai-whisper not installed — run: pip install openai-whisper"

    # Check ffmpeg is available — whisper hangs silently without it
    if not _ffmpeg_available():
        return (
            "[Error] ffmpeg not found — required for audio decoding.\n"
            "Install with: brew install ffmpeg"
        )

    size = model_size or os.getenv("WHISPER_MODEL", "tiny")
    suffix = _detect_suffix(audio_file)
    audio_bytes = audio_file.read()

    def _run() -> str:
        # Write original audio to temp file
        tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp_in.write(audio_bytes)
            tmp_in.flush()
            tmp_in.close()
            tmp_wav.close()

            # Force-convert to 16kHz mono WAV via ffmpeg.
            # This normalises whatever format the browser produced
            # (WebM, OGG, MP4) into a format whisper handles reliably.
            import subprocess  # pylint: disable=import-outside-toplevel
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", tmp_in.name,
                    "-ar", "16000",   # 16 kHz sample rate (whisper default)
                    "-ac", "1",       # mono
                    "-f", "wav",
                    tmp_wav.name,
                ],
                capture_output=True,
                check=True,
            )

            model = _load_model(size)
            result = model.transcribe(tmp_wav.name, fp16=False)
            return result["text"].strip()
        finally:
            Path(tmp_in.name).unlink(missing_ok=True)
            Path(tmp_wav.name).unlink(missing_ok=True)

    # Run with timeout so dashboard never hangs forever
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout  # pylint: disable=import-outside-toplevel
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=TRANSCRIPTION_TIMEOUT)
        except FuturesTimeout:
            return (
                f"[Error] Transcription timed out after {TRANSCRIPTION_TIMEOUT}s. "
                "Try a shorter recording, or switch to the text field."
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return f"[Error] Transcription failed: {exc}"


def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    import shutil  # pylint: disable=import-outside-toplevel
    return shutil.which("ffmpeg") is not None


def _detect_suffix(audio_file) -> str:
    """Try to detect the audio format from the file name or default to .wav."""
    name = getattr(audio_file, "name", "") or ""
    for ext in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
        if name.lower().endswith(ext):
            return ext
    return ".wav"


# Cache the model at module level so it isn't reloaded on every call.
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
