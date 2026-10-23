import time
import google.generativeai as genai
from config import GEMINI_API_KEY

# ---------------------------------------------------------------------------
# Script name for each supported Indic language (used in the prompt)
# ---------------------------------------------------------------------------
_SCRIPT_MAP = {
    "telugu":    "Telugu script (Unicode)",
    "hindi":     "Devanagari script",
    "bengali":   "Bengali script (Unicode)",
    "tamil":     "Tamil script (Unicode)",
    "marathi":   "Devanagari script",
    "gujarati":  "Gujarati script (Unicode)",
    "punjabi":   "Gurmukhi script",
    "kannada":   "Kannada script (Unicode)",
    "malayalam": "Malayalam script (Unicode)",
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt(target_language: str, transliterate: bool) -> str:
    """Build the Gemini system prompt.

    transliterate=False → technical terms kept in English.
    transliterate=True  → technical terms written PURELY in the target script
                          (no English word alongside).
    """
    script_name = _SCRIPT_MAP.get(target_language.strip().lower(), f"{target_language} native script")

    base = f"""You are a specialized translator for Computer Science (CS) lecture transcripts.
Your ONLY task is to translate the non-technical English text into {target_language}.
"""

    if transliterate:
        base += f"""
TRANSLITERATION MODE — FULL SCRIPT OUTPUT:
Every technical CS term that appears in the output MUST be written ONLY in {script_name}.
Do NOT keep the English word alongside. Replace the English term completely with its
phonetic transliteration in {script_name}.

Examples for {target_language} ({script_name}):
  - "memory"             → write ONLY the {script_name} phonetic form
  - "Binary Search Tree" → write ONLY the {script_name} phonetic form
  - "Stack"              → write ONLY the {script_name} phonetic form
  - "Algorithm"          → write ONLY the {script_name} phonetic form

This rule applies to EVERY technical term, including those in the MUST-PRESERVE and
CONTEXT-DEPENDENT lists. Do NOT output any English CS term — the entire output must
be in {target_language} / {script_name}.
"""
    else:
        base += """
Technical CS terms MUST remain in English at all times.
"""

    preserve_rule = (
        f"Write each term purely in {script_name} (phonetic transliteration)."
        if transliterate
        else "Every term in the MUST-PRESERVE list MUST appear EXACTLY as written in English."
    )
    cs_domain_rule = (
        f"Transliterate these into {script_name}."
        if transliterate
        else "Computer Science, Data Structures, Operating Systems, Database Management Systems,"
             " Computer Networks, Algorithms, Compiler Design, Sorting Algorithms, Graph Algorithms,"
             " Data Types must stay in English."
    )
    ambiguous_action = f"transliterate it into {script_name}" if transliterate else "keep it in English"
    unlisted_action  = f"transliterate it into {script_name}" if transliterate else "keep it in English"

    base += f"""
RULE 1 — MUST-PRESERVE LIST
{preserve_rule}

RULE 2 — CS FIELD AND DOMAIN NAMES
{cs_domain_rule}

RULE 3 — CONTEXT-DEPENDENT TERMS
If a term is in the CONTEXT-DEPENDENT list and is used as a CS concept, {ambiguous_action};
otherwise translate it naturally to {target_language}.

RULE 4 — UNLISTED TECHNICAL TERMS
If a term looks technical but is not listed, {unlisted_action}.

RULE 5 — NATURAL, NATIVE-SPEAKER STYLE
Translate all prose into natural, idiomatic {{target_language}} as a fluent native speaker
would say it in an educational lecture — NOT as a word-for-word or connector-for-connector
mapping from English.

Specifically:
* Do NOT repeat the same connecting word or discourse filler (e.g. "so", "therefore",
  "కాబట్టి", "तो", "সুতরাং", "எனவே", "ਇਸ ਲਈ", "તો", "ಆದ್ದರಿಂದ", "അതിനാല്")
  at the start of every sentence. English lecture speech overuses "So" as a discourse filler;
  it does NOT need to be translated — drop it, restructure the sentence to start directly
  with the subject, or vary the connector naturally across sentences.
* Vary sentence openings: mix direct subject starts, different connectors, and light
  restructuring so the output sounds like natural speech, not a mechanical translation.
* Prefer everyday conversational vocabulary over formal or literary equivalents when the
  original English is informal or conversational.
* Aim for the rhythm and flow a real teacher would use when explaining the topic live.

RULE 6 — FORMATTING
Preserve paragraph breaks; output ONLY the translated text.
"""
    return base


def _build_user_prompt(text: str, preserve_str: str, context_str: str) -> str:
    return f"""---
Original Text:
{text}
---
Terms that MUST stay unchanged (or be fully transliterated in transliteration mode):
{preserve_str or 'None'}
---
Terms that are CONTEXT-DEPENDENT:
{context_str or 'None'}
---
Translate the text according to the rules above.
"""


# ---------------------------------------------------------------------------
# Core Gemini call with retry / fallback
# ---------------------------------------------------------------------------

def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini with retry logic. Falls back to a lighter model on repeated failure."""
    genai.configure(api_key=GEMINI_API_KEY)
    model_name = "gemini-3.8-flash"
    backoffs = [5, 10, 20]
    for i, wait in enumerate(backoffs, 1):
        try:
            client = genai.GenerativeModel(model_name, system_instruction=system_prompt)
            response = client.generate_content(user_prompt, generation_config={"temperature": 0.0})
            return response.text.strip()
        except Exception:
            if i == len(backoffs):
                fallback = genai.GenerativeModel("gemini-3.5-flash-lite", system_instruction=system_prompt)
                response = fallback.generate_content(user_prompt, generation_config={"temperature": 0.0})
                return response.text.strip()
            time.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate(text: str, preserve_str: str, context_str: str,
              target_language: str, transliterate: bool = False) -> str:
    """Translate *text* and return a single version.

    transliterate=False → English terms kept.
    transliterate=True  → terms fully replaced by target-script phonetics.
    """
    system_prompt = build_system_prompt(target_language, transliterate)
    user_prompt   = _build_user_prompt(text, preserve_str, context_str)
    return _call_gemini(system_prompt, user_prompt)


def translate_both(text: str, preserve_str: str, context_str: str,
                   target_language: str) -> tuple[str, str]:
    """Translate *text* twice and return ``(normal, transliterated)``.

    normal        → technical terms kept in English.
    transliterated → technical terms written purely in the target script.

    Both API calls are made sequentially so the model is never asked to
    do two conflicting things in the same prompt.
    """
    user_prompt = _build_user_prompt(text, preserve_str, context_str)

    # Version 1: keep English terms
    normal = _call_gemini(
        build_system_prompt(target_language, transliterate=False),
        user_prompt,
    )

    # Version 2: full target-script phonetic transliteration
    transliterated = _call_gemini(
        build_system_prompt(target_language, transliterate=True),
        user_prompt,
    )

    return normal, transliterated


# ---------------------------------------------------------------------------
# Simple yes/no helper used by main.py
# ---------------------------------------------------------------------------

def ask_yes_no(prompt: str) -> bool:
    ans = input(prompt).strip().lower()
    return ans.startswith('y')
