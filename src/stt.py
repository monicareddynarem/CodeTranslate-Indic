'''Utility module for handling English transcript input.

The original implementation performed audio‑to‑text conversion using Gemini
speech models. The workflow has been simplified – the UI now provides the
English transcript directly, so this module only provides a thin pass‑through
helper to keep the rest of the codebase unchanged.
'''

def speech_to_text(transcript: str) -> str:
    """Return the supplied English transcript unchanged.

    Parameters
    ----------
    transcript: str
        The English lecture transcript (already transcribed).
    """
    # Strip leading/trailing whitespace and return.
    return transcript.strip()
