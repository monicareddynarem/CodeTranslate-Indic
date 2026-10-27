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
DICT_DIR   = BASE_DIR / "dictionaries"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import pipeline components
from src.lexicon     import generate_lexicon
from src.pipeline    import setup_ner_pipeline
from src.classifier  import classify_entities, format_term_lists
from src.translator  import translate
from src.tts         import generate_audio


def run_full_pipeline(
    english_text: str,
    target_language: str,
    transliterate: bool = False,
    voice_override: str | None = None,
    progress_callback=None,
) -> tuple[str, str]:
    """Run the complete translation + TTS pipeline.

    Args:
        english_text:    English lecture transcript.
        target_language: Target Indic language key (e.g. 'hindi').
        transliterate:   If True, CS terms are written in the target script;
                         if False (default), they stay in English.
        voice_override:  Optional TTS voice name override.
        progress_callback: Optional callable(current, total) for TTS progress.

    Returns:
        translated_text (str): Translated transcript.
        audio_path      (str): Path to the generated WAV file.
    """
    # ---------- NER  (ontology now returns 3 values) ----------
    patterns, ambiguous_ids, entry_map = generate_lexicon(str(DICT_DIR))
    nlp = setup_ner_pipeline(patterns)
    doc = nlp(english_text)
    must_preserve, context_dependent = classify_entities(doc, ambiguous_ids, entry_map)
    preserve_str, context_str = format_term_lists(must_preserve, context_dependent)

    # ---------- Single translation based on chosen mode ----------
    translated_text = translate(
        text=english_text,
        preserve_str=preserve_str,
        context_str=context_str,
        target_language=target_language,
        transliterate=transliterate,
    )

    # ---------- TTS ----------
    audio_path = str(OUTPUT_DIR / f"lecture_{target_language.lower()}.wav")
    generate_audio(
        translated_text,
        target_language,
        audio_path,
        voice_override=voice_override,
        progress_callback=progress_callback,
    )

    return translated_text, audio_path


if __name__ == "__main__":
    # Simple CLI for debugging – not used by Streamlit
    parser = argparse.ArgumentParser(description="Run pipeline on a given text file")
    parser.add_argument("text_file", type=str, help="Path to English transcript file")
    parser.add_argument("--lang", type=str, default="telugu", help="Target Indic language")
    args = parser.parse_args()

    txt  = Path(args.text_file).read_text(encoding="utf-8")
    translated, audio = run_full_pipeline(txt, args.lang)

    out_file = OUTPUT_DIR / f"translated_{args.lang}.txt"
    out_file.write_text(translated, encoding="utf-8")
    print(f"Translation → {out_file}")
    print(f"Audio       → {audio}")
