import google.generativeai as genai
from config import GEMINI_API_KEY
import time
import tempfile
import os

# ---------------------------------------------------------------------------
# Speech‑to‑Text helper using Gemini Speech model
# ---------------------------------------------------------------------------

def speech_to_text(audio_bytes: bytes, mime_type: str = "audio/mp3") -> str:
    """Convert an audio file (wav/mp3) to English transcript.

    The function uploads the audio to Gemini, calls a speech‑to‑text model,
    and returns the recognised text.

    Parameters
    ----------
    audio_bytes : bytes – raw audio data.
    mime_type   : str   – MIME type of the audio (e.g. "audio/mp3" or "audio/wav").
    """
    genai.configure(api_key=GEMINI_API_KEY)
    # Use the Gemini Flash model which supports audio input.
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Determine the correct file extension from the mime type.
    ext = ".mp3" if "mp3" in mime_type else ".wav"

    # Write audio bytes to a temporary file – Gemini needs a file path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        # Upload the file to Gemini with the correct mime type.
        uploaded = genai.upload_file(path=tmp_path, mime_type=mime_type)

        # Wait until the file is fully processed by the API.
        while uploaded.state.name == "PROCESSING":
            time.sleep(1)
            uploaded = genai.get_file(uploaded.name)

        # Request transcription with an explicit prompt.
        response = model.generate_content(
            ["Please transcribe the following audio accurately.", uploaded],
            generation_config={"response_mime_type": "text/plain"},
        )
        return response.text.strip()
    except Exception as e:
        # Log the error and return empty so the UI shows the friendly message.
        print(f"[STT] Error during transcription: {e}")
        return ""
    finally:
        # Clean up temporary file.
        try:
            os.remove(tmp_path)
        except OSError:
            pass

