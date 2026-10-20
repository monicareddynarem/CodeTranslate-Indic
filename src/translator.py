import os
import re
import time
from pathlib import Path
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


def build_system_prompt(target_language: str, transliterate: bool) -> str:
    script_name = _SCRIPT_MAP.get(target_language.strip().lower(), f"{target_language} native script")

    base = f"""You are a specialized translator for Computer Science (CS) lecture transcripts.
Your ONLY task is to translate the non-technical English text into {target_language}.
Technical CS terms MUST remain in English at all times.
"""
    if transliterate:
        base += f"""
Additionally, for EVERY technical CS term that appears in the output, write it in the following
format — English term first, followed immediately by its phonetic transliteration in {script_name}
inside parentheses:

    English_Term ({script_name} transliteration)

Examples for {target_language}:
  - memory ({script_name} → write how "memory" sounds in {script_name})
  - Binary Search Tree ({script_name} → write how "Binary Search Tree" sounds in {script_name})
  - Stack ({script_name} → write how "Stack" sounds in {script_name})

This format MUST be applied consistently for EVERY language including {target_language}.
Do NOT skip this transliteration for any technical term.
"""
    base += f"""
RULE 1 — MUST-PRESERVE LIST (absolutely non-negotiable)
Every term in the MUST-PRESERVE list below MUST appear in the output EXACTLY as written in English
(with its {script_name} transliteration in parentheses if transliteration is enabled).

RULE 2 — CS FIELD AND DOMAIN NAMES (never translate)
Computer Science, Data Structures, Operating Systems, Database Management Systems, Computer Networks, Algorithms, Compiler Design, Sorting Algorithms, Graph Algorithms, Data Types must stay in English.

RULE 3 — CONTEXT-DEPENDENT TERMS (use context to decide)
If a term is in the CONTEXT-DEPENDENT list and is used as a CS concept, keep it in English; otherwise translate it.

RULE 4 — UNLISTED TECHNICAL TERMS (when in doubt, keep English)
If a term looks technical but is not listed, keep it in English.

RULE 5 — GENERAL TEXT
Translate all other prose into natural, fluent {target_language}.

RULE 6 — FORMATTING
Preserve paragraph breaks; output ONLY the translated text.
"""
    return base


# ---------------------------------------------------------------------------
# Translation function
# ---------------------------------------------------------------------------

def translate(text: str, preserve_str: str, context_str: str, target_language: str, transliterate: bool = False) -> str:
    """Translate `text` using Gemini.

    Args:
        text: Original English transcript (code blocks already stripped).
        preserve_str: Comma‑separated list of terms that must stay unchanged.
        context_str: Comma‑separated list of ambiguous terms.
        target_language: Human‑readable language name (e.g., "telugu").
        transliterate: If True, ask the model to transliterate technical terms.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    model_name = "gemini-3.8-flash"

    system_prompt = build_system_prompt(target_language, transliterate)
    # Build the user prompt with the term lists
    user_prompt = f"""
---
Original Text:
{text}
---
Terms that MUST stay unchanged:
{preserve_str or 'None'}
---
Terms that are CONTEXT‑DEPENDENT:
{context_str or 'None'}
---
Translate the text according to the rules above.
"""
    # Retry logic (same as previous implementation)
    backoffs = [5, 10, 20]
    for i, wait in enumerate(backoffs, 1):
        try:
            client = genai.GenerativeModel(model_name, system_instruction=system_prompt)
            response = client.generate_content(user_prompt, generation_config={"temperature": 0.0})
            return response.text.strip()
        except Exception as e:
            if i == len(backoffs):
                # fallback to lighter model
                fallback = genai.GenerativeModel("gemini-3.5-flash-lite", system_instruction=system_prompt)
                response = fallback.generate_content(user_prompt, generation_config={"temperature": 0.0})
                return response.text.strip()
            time.sleep(wait)
    return ""

# ---------------------------------------------------------------------------
# Simple yes/no helper used by main.py
# ---------------------------------------------------------------------------

def ask_yes_no(prompt: str) -> bool:
    ans = input(prompt).strip().lower()
    return ans.startswith('y')

# ---------------------------------------------------------------------------
# TTS integration (unchanged, imported from src/tts later if needed)
# ---------------------------------------------------------------------------
