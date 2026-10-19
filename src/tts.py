# src/tts.py – Text‑to‑Speech using the modern google‑genai SDK
"""Utility functions to synthesize Indic language audio with Gemini TTS.

The legacy ``google‑generativeai`` package does not support audio generation via
``response_mime_type``.  The newer ``google‑genai`` SDK (installed as
``google‑genai``) provides a clean way to request audio output by setting the
``response_modalities`` to ``["AUDIO"]`` and supplying a ``SpeechConfig`` that
specifies a pre‑built voice.

This module hides those details behind a simple ``generate_audio`` function that
matches the existing ``src.driver.run_full_pipeline`` call signature.
"""

import base64
import re
import time
from pathlib import Path

import google.genai as genai  # modern SDK for TTS
from config import GEMINI_API_KEY

# ---------------------------------------------------------------------------
# Voice mapping – choose a voice that works well for each Indic script.
# ---------------------------------------------------------------------------
VOICE_MAP = {
    "telugu": "Charon",   # male voice, good Telugu pronunciation
    "hindi": "Kore",      # female voice, clear Hindi articulation
    "bengali": "Fenrir",  # neutral voice works for Bengali script
    "tamil": "Aoede",
    "marathi": "Puck",
    "gujarati": "Kore",
    "punjabi": "Aoede",
    "kannada": "Fenrir",
    "malayalam": "Charon",
}
DEFAULT_VOICE = "Charon"

def _choose_voice(language: str) -> str:
    """Return the Gemini voice name for *language*.
    Falls back to ``DEFAULT_VOICE`` if the language is unknown.
    """
    return VOICE_MAP.get(language.strip().lower(), DEFAULT_VOICE)

# ---------------------------------------------------------------------------
# Low‑level chunk request – uses ``google‑genai``.
# ---------------------------------------------------------------------------
def _tts_chunk(genai_client, text_chunk: str, voice_name: str, idx: int, total: int) -> bytes:
    """Generate audio for a single *text_chunk* using a GenAI client.

    Parameters
    ----------
    genai_client: ``genai.Client`` – the GenAI client instance.
    text_chunk: str – portion of the transcript to synthesize.
    voice_name: str – pre‑built Gemini voice identifier.
    idx, total: int – for error reporting only.
    """
    system_prompt = f"You are a speech synthesis model. Use the voice '{voice_name}'."
    # Build a generation config that requests audio and supplies the voice.
    gen_config = genai.types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=genai.types.SpeechConfig(
            voice_config=genai.types.VoiceConfig(
                prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )
    try:
        response = genai_client.models.generate_content(
            model="gemini-3.8-flash-tts",
            contents=f"{system_prompt}\n{text_chunk}",
            config=gen_config,
        )
        # Extract WAV bytes from the inline_data payload.
        for part in response.parts:
            if getattr(part, "inline_data", None) and part.inline_data.mime_type == "audio/wav":
                return base64.b64decode(part.inline_data.data)
        raise RuntimeError("TTS chunk did not contain audio data")
    except Exception as exc:  # pragma: no‑cover
        raise RuntimeError(
            f"TTS chunk {idx}/{total} failed (voice={voice_name}): {exc}"
        ) from exc

# ---------------------------------------------------------------------------
# Split long transcripts into safe‑size chunks (≈12 k token limit).
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
# Merge raw WAV chunks into a single file, handling possible missing headers.
# ---------------------------------------------------------------------------
def _merge_wav_chunks(wav_chunks: list[bytes]) -> bytes:
    if not wav_chunks:
        raise ValueError("No WAV chunks to merge")

    def _data_offset(chunk: bytes) -> int:
        pos = chunk.find(b"data")
        if pos == -1:
            raise ValueError("Invalid WAV chunk: missing 'data' marker")
        return pos + 8

    try:
        header = wav_chunks[0][: _data_offset(wav_chunks[0])]
        pcm_parts = []
        for c in wav_chunks:
            try:
                off = _data_offset(c)
                pcm_parts.append(c[off:])
            except ValueError:
                pcm_parts.append(c)
        return header + b"".join(pcm_parts)
    except ValueError:
        return b"".join(wav_chunks)

# ---------------------------------------------------------------------------
# Public API – generate a WAV file for the full transcript.
# ---------------------------------------------------------------------------
def generate_audio(text: str, language: str, output_path: str, progress_callback=None) -> None:
    """Generate a WAV file for *text* in *language*.

    ``progress_callback`` is called with ``(current_chunk, total_chunks)`` after
    each chunk is processed; the Streamlit UI uses this to update a progress bar.
    """
    # Initialize GenAI client with API key (no configure needed)
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    voice = _choose_voice(language)

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
def synthesize_to_bytes(text: str, language: str) -> bytes:
    # Initialize GenAI client for on‑the‑fly synthesis (no configure needed)
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    voice = _choose_voice(language)
    return _tts_chunk(genai_client, text, voice, 1, 1)
