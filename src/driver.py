import argparse
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH for direct script execution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Project paths (same layout as before)
BASE_DIR = ROOT
INPUT_FILE = BASE_DIR / "input" / "lecture.txt"
OUTPUT_DIR = BASE_DIR / "output"
DICT_DIR = BASE_DIR / "dictionaries"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import pipeline components
from src.lexicon import generate_lexicon
from src.pipeline import setup_ner_pipeline
from src.classifier import classify_entities, format_term_lists
from src.translator import translate
from src.tts import generate_audio

def run_full_pipeline(
    english_text: str,
    target_language: str,
    transliterate: bool = False,
    progress_callback=None,
) -> tuple[str, str]:
    """Run the complete translation + TTS pipeline.

    Returns:
        translated_text (str): The Indic transcript.
        audio_path (str): Path to the generated WAV file.
    """
    # ---------- NER ----------
    patterns, ambiguous_terms = generate_lexicon(str(DICT_DIR))
    nlp = setup_ner_pipeline(patterns)
    doc = nlp(english_text)
    must_preserve, context_dependent = classify_entities(doc, ambiguous_terms)
    preserve_str, context_str = format_term_lists(must_preserve, context_dependent)

    # ---------- Translation ----------
    translated = translate(
        text=english_text,
        preserve_str=preserve_str,
        context_str=context_str,
        target_language=target_language,
        transliterate=transliterate,
    )

    # ---------- TTS ----------
    audio_path = str(OUTPUT_DIR / f"lecture_{target_language.lower()}.wav")
    generate_audio(translated, target_language, audio_path, progress_callback=progress_callback)

    return translated, audio_path

if __name__ == "__main__":
    # Simple CLI for debugging – not used by Streamlit
    parser = argparse.ArgumentParser(description="Run pipeline on a given text file")
    parser.add_argument("text_file", type=str, help="Path to English transcript file")
    parser.add_argument("--lang", type=str, default="telugu", help="Target Indic language")
    parser.add_argument("--transliterate", action="store_true", help="Transliterate technical terms")
    args = parser.parse_args()
    txt = Path(args.text_file).read_text(encoding="utf-8")
    tr, audio = run_full_pipeline(txt, args.lang, args.transliterate)
    out_txt = OUTPUT_DIR / f"translated_{args.lang}.txt"
    out_txt.write_text(tr, encoding="utf-8")
    print(f"✔️  Translation saved to {out_txt}\n✔️  Audio saved to {audio}")
