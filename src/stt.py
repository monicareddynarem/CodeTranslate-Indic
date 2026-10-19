import google.generativeai as genai
from config import GEMINI_API_KEY
import time
import tempfile
import os

# ---------------------------------------------------------------------------
# Speech‑to‑Text helper using Gemini Speech model
# ---------------------------------------------------------------------------

def speech_to_text(audio_bytes: bytes) -> str:
    """Convert an audio file (wav/mp3) to English transcript.

    The function uploads the audio to Gemini, calls a speech‑to‑text model,
    and returns the recognised text.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    # Use the Gemini Speech model (as of SDK 2.x the model name is
    # "gemini-1.5-flash" with audio support). Adjust if a newer model
    # becomes available.
    model = genai.GenerativeModel("gemini-1.5-flash")
    # Write audio bytes to a temporary file – Gemini needs a file path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        # Upload the file to Gemini (required for large audio)
        uploaded = genai.upload_file(path=tmp_path)
        # Request transcription – the model returns a `content` object with .text
        response = model.generate_content([uploaded], generation_config={"response_mime_type": "text/plain"})
        return response.text.strip()
    except Exception as e:
        # Fallback: return an empty string and log the error
        print(f"[STT] Error during transcription: {e}")
        return ""
    finally:
        # Clean up temporary file
        try:
            os.remove(tmp_path)
        except OSError:
            pass
