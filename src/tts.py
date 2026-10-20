# src/tts.py – Text‑to‑Speech using the modern google‑genai SDK
"""Utility functions to synthesize Indic language audio with Gemini TTS.

The legacy ``google‑generativeai`` package does not support audio generation via
``response_mime_type``.  The newer ``google‑genai`` SDK (installed as
``google‑genai``) provides a clean way to request audio output by setting the
``response_modalities`` to ``[\"AUDIO\"]`` and supplying a ``SpeechConfig`` that
specifies a pre‑built voice.

This module hides those details behind a simple ``generate_audio`` function that
matches the existing ``src.driver.run_full_pipeline`` call signature.
"""

import io
import re
import struct
import time
import wave
from pathlib import Path

import google.genai as genai  # modern SDK for TTS
from google.genai import types
from config import GEMINI_API_KEY

# ---------------------------------------------------------------------------
# Available voices (all official Gemini pre‑built voices)
# ---------------------------------------------------------------------------
AVAILABLE_VOICES = [
    "Aoede",
    "Charon",
    "Fenrir",
    "Kore",
    "Puck",
    "Orbit",
    "Zephyr",
    "Leda",
    "Orus",
    "Schedar",
]

# Default voice per language
VOICE_MAP = {
    "telugu":    "Charon",
    "hindi":     "Kore",
    "bengali":   "Fenrir",
    "tamil":     "Aoede",
    "marathi":   "Puck",
    "gujarati":  "Kore",
    "punjabi":   "Aoede",
    "kannada":   "Fenrir",
    "malayalam": "Charon",
}
DEFAULT_VOICE = "Charon"


def _choose_voice(language: str, override: str | None = None) -> str:
    """Return the Gemini voice name for *language*.
    If *override* is provided and is a valid voice name, it is used instead.
    Falls back to ``DEFAULT_VOICE`` if the language is unknown.
    """
    if override and override in AVAILABLE_VOICES:
        return override
    return VOICE_MAP.get(language.strip().lower(), DEFAULT_VOICE)


# ---------------------------------------------------------------------------
# Convert raw PCM bytes (L16 24 kHz mono) to a proper WAV file in memory.
# ---------------------------------------------------------------------------
def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV container and return the WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Low‑level chunk request – uses ``google‑genai``.
# ---------------------------------------------------------------------------
def _tts_chunk(genai_client, text_chunk: str, voice_name: str, idx: int, total: int) -> bytes:
    """Generate WAV audio for a single *text_chunk* using a GenAI client.

    Parameters
    ----------
    genai_client : ``genai.Client`` – the GenAI client instance.
    text_chunk   : str – portion of the transcript to synthesize.
    voice_name   : str – pre‑built Gemini voice identifier.
    idx, total   : int – for error reporting only.

    Returns
    -------
    bytes – a valid WAV file (with header).
    """
    gen_config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )
    try:
        response = genai_client.models.generate_content(
            model="gemini-3.8-flash-tts",
            contents=text_chunk,
            config=gen_config,
        )
        for part in response.parts:
            if getattr(part, "inline_data", None):
                raw = part.inline_data.data          # already bytes (SDK decodes base64)
                mime = part.inline_data.mime_type    # e.g. "audio/L16;rate=24000" or "audio/wav"

                # If the API returns raw PCM (L16), wrap it in a WAV container.
                if "wav" in mime.lower():
                    return raw  # already a valid WAV
                else:
                    # Parse sample rate from mime if present, e.g. "audio/L16;rate=24000"
                    rate = 24000
                    if "rate=" in mime:
                        try:
                            rate = int(mime.split("rate=")[1].split(";")[0])
                        except (ValueError, IndexError):
                            pass
                    return _pcm_to_wav(raw, sample_rate=rate)

        raise RuntimeError("TTS response contained no audio data")
    except Exception as exc:
        raise RuntimeError(
            f"TTS chunk {idx}/{total} failed (voice={voice_name}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Split long transcripts into safe‑size chunks (≈ 12 k token limit).
# ---------------------------------------------------------------------------
MAX_CHUNK_SIZE = 11_000  # safe margin


def _split_into_chunks(text: str, limit: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split *text* at sentence boundaries so each chunk is ≤ *limit* characters.
    Falls back to paragraph splitting for exceptionally long sentences.
    """
    sentences = re.split(r"(?<=[।.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= limit:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            if len(sent) > limit:
                for part in sent.split("\n\n"):
                    if part.strip():
                        chunks.append(part.strip())
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Merge WAV chunks into a single WAV file.
# ---------------------------------------------------------------------------
def _merge_wav_chunks(wav_chunks: list[bytes]) -> bytes:
    """Concatenate multiple WAV files into one, reusing the first file's header."""
    if not wav_chunks:
        raise ValueError("No WAV chunks to merge")
    if len(wav_chunks) == 1:
        return wav_chunks[0]

    # Collect PCM data from every chunk using the wave module.
    pcm_parts: list[bytes] = []
    params = None
    for chunk_bytes in wav_chunks:
        buf = io.BytesIO(chunk_bytes)
        try:
            with wave.open(buf, "rb") as wf:
                if params is None:
                    params = wf.getparams()
                pcm_parts.append(wf.readframes(wf.getnframes()))
        except Exception:
            # If wave parsing fails, append raw bytes (best effort).
            pcm_parts.append(chunk_bytes)

    if params is None:
        return b"".join(wav_chunks)

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf:
        wf.setparams(params)
        for pcm in pcm_parts:
            wf.writeframes(pcm)
    return out_buf.getvalue()


# ---------------------------------------------------------------------------
# Public API – generate a WAV file for the full transcript.
# ---------------------------------------------------------------------------
def generate_audio(
    text: str,
    language: str,
    output_path: str,
    voice_override: str | None = None,
    progress_callback=None,
) -> None:
    """Generate a WAV file for *text* in *language*.

    Parameters
    ----------
    text              : str – the Indic transcript to synthesize.
    language          : str – target Indic language key (e.g. "telugu").
    output_path       : str – path where the WAV file will be written.
    voice_override    : str | None – if provided, use this voice instead of the default.
    progress_callback : callable | None – called with (current_chunk, total_chunks).
    """
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    voice = _choose_voice(language, override=voice_override)

    chunks = _split_into_chunks(text)
    total = len(chunks)
    wav_parts: list[bytes] = []
    for i, chunk in enumerate(chunks, start=1):
        wav = _tts_chunk(genai_client, chunk, voice, i, total)
        wav_parts.append(wav)
        time.sleep(0.2)  # rate‑limit friendly
        if progress_callback:
            progress_callback(i, total)

    final_wav = _merge_wav_chunks(wav_parts)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(final_wav)
    print(f"Audio saved to {out_file} ({len(final_wav) / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Helper for UI previews – returns raw bytes without writing to disk.
# ---------------------------------------------------------------------------
def synthesize_to_bytes(text: str, language: str, voice_override: str | None = None) -> bytes:
    """Synthesize *text* in *language* and return WAV bytes (no file written)."""
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    voice = _choose_voice(language, override=voice_override)
    return _tts_chunk(genai_client, text, voice, 1, 1)
