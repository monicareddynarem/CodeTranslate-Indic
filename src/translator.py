"""
translator.py
--------------
Handles translation of CS lecture text from English to any target language
using the Gemini API, with NER-informed term preservation.
"""

import os
import time
from google import genai


def build_system_prompt(target_language):
    """
    Builds the system prompt for the given target language.
    The CS term preservation rules are language-agnostic.
    """
    return f"""You are a specialized translator for Computer Science (CS) lecture transcripts.
Your ONLY task is to translate the non-technical English text into {target_language}.
Technical CS terms MUST remain in English at all times.

═══════════════════════════════════════════════════
RULE 1 — MUST-PRESERVE LIST (absolutely non-negotiable)
═══════════════════════════════════════════════════
Every term in the MUST-PRESERVE list below MUST appear in the output EXACTLY
as written in English. These are confirmed CS technical terms.
Do NOT translate, transliterate, or paraphrase them — not even partially.

WRONG: "Binary Search Tree" → "ikili arama ağacı" / "二叉搜索树" / "द्विआधारी खोज वृक्ष"
RIGHT: "Binary Search Tree" stays as "Binary Search Tree" in ALL languages.

═══════════════════════════════════════════════════
RULE 2 — CS FIELD AND DOMAIN NAMES (never translate)
═══════════════════════════════════════════════════
The following categories of terms must ALWAYS stay in English regardless of
whether they appear in the lists or not:
- CS subject names: "Computer Science", "Data Structures", "Operating Systems",
  "Database Management Systems", "Computer Networks", "Algorithms", "Compiler Design"
- CS concept categories: "Sorting Algorithms", "Graph Algorithms", "Data Types"
- Standard CS vocabulary used in an academic/educational context

WRONG: "Operating Systems" → "sistemas operativos" / "ऑपरेटिंग सिस्टम" (transliterated)
WRONG: "Data Structures" → "структуры данных" / "veri yapıları"
RIGHT: "Operating Systems", "Data Structures", "Computer Science" stay in English.

═══════════════════════════════════════════════════
RULE 3 — CONTEXT-DEPENDENT TERMS (use context to decide)
═══════════════════════════════════════════════════
For terms in the CONTEXT-DEPENDENT list:
- If used as a CS/technical term in this context → keep it in English
- If used as a plain everyday word → translate it to {target_language}

Example: "tree"
  CS context: "traverse the tree using DFS" → keep "tree" in English
  Everyday:   "a monkey sat on a tree"      → translate "tree" to {target_language}

═══════════════════════════════════════════════════
RULE 4 — UNLISTED TECHNICAL TERMS (when in doubt, keep English)
═══════════════════════════════════════════════════
If a term looks technical (it names a CS concept, algorithm, data structure,
system, or methodology) but is NOT in either list, keep it in English.
It is better to leave a technical term in English than to mistranslate it.

═══════════════════════════════════════════════════
RULE 5 — GENERAL TEXT
═══════════════════════════════════════════════════
All non-technical sentence structure, connectives, and explanations should be
translated into natural, fluent {target_language}.
The output should sound like a {target_language}-medium CS professor explaining
concepts — naturally code-switching English CS terms into {target_language} sentences.

═══════════════════════════════════════════════════
RULE 6 — FORMATTING
═══════════════════════════════════════════════════
- Preserve paragraph breaks (blank lines between paragraphs).
- Do NOT add explanations, footnotes, or commentary.
- Output ONLY the translated text, nothing else.
"""



def build_translation_prompt(original_text, preserve_str, context_str):
    """
    Builds the per-request user prompt that includes the NER-detected terms
    and the text to translate.
    """
    return f"""MUST-PRESERVE TERMS (always keep in English):
{preserve_str}

CONTEXT-DEPENDENT TERMS (decide based on context):
{context_str}

---

TEXT TO TRANSLATE:

{original_text}"""


def translate(original_text, preserve_str, context_str, target_language):
    """
    Translates the given English CS lecture text to the specified target language
    using Gemini, preserving technical terms based on NER classification.

    Requires GEMINI_API_KEY environment variable to be set.

    Args:
        original_text:   the English lecture text
        preserve_str:    formatted string of must-preserve terms
        context_str:     formatted string of context-dependent terms
        target_language: name of the target language (e.g. "Telugu", "Hindi")

    Returns:
        translated_text: the translated transcript
    """
    # Try config.py first, then fall back to the environment variable
    api_key = None
    try:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from config import GEMINI_API_KEY as _cfg_key
        if _cfg_key and _cfg_key != "your_api_key_here":
            api_key = _cfg_key
    except ImportError:
        pass

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "No API key found.\n"
            "Option 1 (recommended): Open config.py and paste your key there.\n"
            "Option 2: Set the environment variable GEMINI_API_KEY.\n"
            "Get your free key from https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    system_prompt = build_system_prompt(target_language)
    user_prompt = build_translation_prompt(original_text, preserve_str, context_str)

    # Retry settings
    PRIMARY_MODEL  = "gemini-3.8-flash"
    FALLBACK_MODEL = "gemini-3.5-flash-lite"
    MAX_RETRIES    = 3          # attempts on the primary model
    WAIT_SECONDS   = [5, 10, 20]  # wait before each retry

    last_error = None

    for attempt in range(MAX_RETRIES + 1):  # +1 for the fallback attempt
        model = PRIMARY_MODEL if attempt < MAX_RETRIES else FALLBACK_MODEL

        if attempt == 0:
            print(f"  Sending to Gemini (translating to {target_language})...")
        elif attempt < MAX_RETRIES:
            wait = WAIT_SECONDS[attempt - 1]
            print(f"  Server busy — retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
        else:
            print(f"  Primary model unavailable — trying fallback model ({FALLBACK_MODEL})...")

        try:
            interaction = client.interactions.create(
                model=model,
                system_instruction=system_prompt,
                input=user_prompt,
                store=False,
            )
            translated_text = interaction.output_text

            if not translated_text:
                raise RuntimeError("Gemini returned an empty response.")

            if attempt > 0:
                print(f"  Success on attempt {attempt + 1}.")
            return translated_text

        except Exception as e:
            last_error = e
            err_str = str(e)
            # Only retry on transient server errors (503, 429, 500)
            if any(code in err_str for code in ["503", "429", "500", "service_unavailable", "overloaded"]):
                if attempt < MAX_RETRIES:
                    continue  # will wait and retry
                else:
                    # Fallback also failed
                    raise RuntimeError(
                        f"Translation failed after {MAX_RETRIES} retries and fallback.\n"
                        f"The Gemini API is experiencing high demand right now.\n"
                        f"Please wait a few minutes and try again.\n"
                        f"Last error: {err_str}"
                    ) from e
            else:
                # Non-transient error (bad key, quota, etc.) — don't retry
                raise

