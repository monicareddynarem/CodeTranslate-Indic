import argparse
import re
import sys
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "input" / "lecture.txt"
OUTPUT_DIR = BASE_DIR / "output"
DICT_DIR = BASE_DIR / "dictionaries"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import pipeline components
from src.lexicon import generate_lexicon
from src.pipeline import setup_ner_pipeline
from src.classifier import classify_entities, format_term_lists
from src.translator import translate

# ---------------------------------------------------------------------------
# Argument parsing – efficient Indic language selector and transliteration flag
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CodeTranslate‑Indic pipeline")
parser.add_argument(
    "--lang",
    type=str,
    required=False,
    help="Target Indic language (e.g., telugu, hindi, bengali).",
)
parser.add_argument(
    "--transliterate",
    action="store_true",
    help="Transliterate technical terms into the target script instead of keeping them in English.",
)
args = parser.parse_args()

# Supported Indic languages (name → human‑readable identifier used in prompts)
INDIC_LANGS = {
    "telugu": "Telugu",
    "tamil": "Tamil",
    "hindi": "Hindi",
    "bengali": "Bengali",
    "marathi": "Marathi",
    "gujarati": "Gujarati",
    "punjabi": "Punjabi",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
}

if args.lang:
    key = args.lang.strip().lower()
    # allow exact or prefix match (case‑insensitive)
    matches = [k for k in INDIC_LANGS if k.startswith(key) or key.startswith(k)]
    if not matches:
        print(f"Unsupported language '{args.lang}'. Supported: {', '.join(INDIC_LANGS)}")
        sys.exit(1)
    selected_lang = matches[0]
else:
    # Interactive fallback menu
    print("Select target Indic language:")
    for i, name in enumerate(INDIC_LANGS.keys(), 1):
        print(f"{i}. {name.title()}")
    try:
        choice = int(input("Enter number: ").strip())
        selected_lang = list(INDIC_LANGS.keys())[choice - 1]
    except Exception:
        print("Invalid selection.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Preserve code blocks (fenced ```...``` and inline back‑ticks) during translation
# ---------------------------------------------------------------------------
raw_text = INPUT_FILE.read_text(encoding="utf-8")
code_blocks = {}

def _store_block(match):
    placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
    code_blocks[placeholder] = match.group(0)
    return placeholder

# Replace fenced code blocks first, then inline back‑ticks
text_no_fenced = re.sub(r"(?s)```.*?```", _store_block, raw_text)
clean_text = re.sub(r"`[^`]+`", _store_block, text_no_fenced)

# ---------------------------------------------------------------------------
# NER pipeline setup  (ontology now returns 3 values)
# ---------------------------------------------------------------------------
patterns, ambiguous_ids, entry_map = generate_lexicon(str(DICT_DIR))
nlp = setup_ner_pipeline(patterns)

doc = nlp(clean_text)
must_preserve, context_dependent = classify_entities(doc, ambiguous_ids, entry_map)

# Convert entity lists to enriched strings for the LLM prompt
preserve_str, context_str = format_term_lists(must_preserve, context_dependent)

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
translated = translate(
    text=clean_text,
    preserve_str=preserve_str,
    context_str=context_str,
    target_language=INDIC_LANGS[selected_lang],
    transliterate=args.transliterate,
)

# Re‑insert the original code blocks
for placeholder, block in code_blocks.items():
    translated = translated.replace(placeholder, block)

# ---------------------------------------------------------------------------
# Write output file
# ---------------------------------------------------------------------------
out_path = OUTPUT_DIR / f"translated_{selected_lang}.txt"
out_path.write_text(translated, encoding="utf-8")
print(f"✅ Translation saved to {out_path}")
