"""
CS Lecture Translator — English to Any Regional Language
=========================================================
Translates CS lecture transcripts while preserving technical terminology.

Pipeline:
  1. Select target language interactively
  2. Load CS term dictionaries (with ambiguity annotations)
  3. Run NER to detect CS terms in the input text
  4. Classify terms into must-preserve vs context-dependent
  5. Send to Gemini with structured instructions for translation
  6. Save output transcript

Usage:
  1. Place your lecture transcript in  input/lecture.txt
  2. Set your API key:  set GEMINI_API_KEY=your_key_here
  3. Run:  python main.py
"""

import os
from src.lexicon import generate_lexicon
from src.pipeline import setup_ner_pipeline
from src.classifier import classify_entities, format_term_lists
from src.translator import translate
from src.tts import generate_audio, select_voice


# --- Configuration ---
DICT_DIR   = "dictionaries"
INPUT_FILE = os.path.join("input", "lecture.txt")
OUTPUT_DIR = "output"
NER_REPORT = os.path.join(OUTPUT_DIR, "ner_report.txt")

# Suggested languages shown to the user (they can also type any other language)
SUGGESTED_LANGUAGES = [
    "Telugu",
    "Hindi",
    "Tamil",
    "Kannada",
    "Malayalam",
    "Bengali",
    "Marathi",
    "Gujarati",
    "Punjabi",
    "Odia",
]


def select_language():
    """
    Interactively prompts the user to choose a target language.
    Returns the chosen language name as a string.
    """
    print("\n" + "=" * 60)
    print("  CS LECTURE TRANSLATOR")
    print("=" * 60)
    print("\nAvailable languages (or type any other language name):\n")
    for i, lang in enumerate(SUGGESTED_LANGUAGES, 1):
        print(f"  [{i:2d}]  {lang}")
    print()

    while True:
        choice = input("Enter a number or type a language name: ").strip()

        if not choice:
            print("  Please enter a number or language name.")
            continue

        # If user entered a number, map to the suggested list
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(SUGGESTED_LANGUAGES):
                return SUGGESTED_LANGUAGES[idx]
            else:
                print(f"  Please enter a number between 1 and {len(SUGGESTED_LANGUAGES)}.")
                continue

        # Otherwise treat the raw input as a language name
        # Capitalise the first letter for consistency
        return choice.strip().title()


def get_output_filename(target_language):
    """Returns the output file path based on the selected language."""
    safe_name = target_language.lower().replace(" ", "_")
    return os.path.join(OUTPUT_DIR, f"translated_{safe_name}.txt")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------
    # Step 0: Select target language
    # --------------------------------------------------
    target_language = select_language()
    output_file = get_output_filename(target_language)
    print(f"\n  Target language: {target_language}")
    print(f"  Output will be saved to: {output_file}\n")

    # --------------------------------------------------
    # Step 1: Build the CS lexicon from dictionary files
    # --------------------------------------------------
    print(f"[Step 1/5] Building lexicon from '{DICT_DIR}'...")
    cs_patterns, ambiguous_terms = generate_lexicon(DICT_DIR)

    if not cs_patterns:
        print(f"  ERROR: No patterns found! Add terms to the .txt files in '{DICT_DIR}'.")
        return

    print(f"  Loaded {len(cs_patterns)} patterns ({len(ambiguous_terms)} marked as ambiguous)")

    # --------------------------------------------------
    # Step 2: Read the input lecture transcript
    # --------------------------------------------------
    print(f"\n[Step 2/5] Reading input from '{INPUT_FILE}'...")

    if not os.path.exists(INPUT_FILE):
        print(f"  ERROR: '{INPUT_FILE}' not found!")
        print(f"  Please paste your lecture transcript into that file and run again.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lecture_text = f.read().strip()

    if not lecture_text:
        print(f"  ERROR: '{INPUT_FILE}' is empty!")
        return

    # Normalise smart quotes
    lecture_text = lecture_text.replace("\u2018", "'").replace("\u2019", "'")
    print(f"  Read {len(lecture_text)} characters ({lecture_text.count(chr(10)) + 1} lines)")

    # --------------------------------------------------
    # Step 3: Run NER to detect CS terms
    # --------------------------------------------------
    print(f"\n[Step 3/5] Running NER pipeline...")
    nlp = setup_ner_pipeline(cs_patterns)
    doc = nlp(lecture_text)
    print(f"  Detected {len(doc.ents)} entity mentions")

    # --------------------------------------------------
    # Step 4: Classify entities
    # --------------------------------------------------
    print(f"\n[Step 4/5] Classifying detected terms...")
    must_preserve, context_dependent = classify_entities(doc, ambiguous_terms)
    preserve_str, context_str = format_term_lists(must_preserve, context_dependent)

    print(f"  Must-preserve (unambiguous): {len(must_preserve)} unique terms")
    print(f"  Context-dependent (ambiguous): {len(context_dependent)} unique terms")

    # Save NER report for inspection
    with open(NER_REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("NER CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Target language: {target_language}\n\n")
        f.write("MUST-PRESERVE TERMS (always kept in English):\n")
        f.write(preserve_str + "\n\n")
        f.write("CONTEXT-DEPENDENT TERMS (LLM decides from context):\n")
        f.write(context_str + "\n\n")
        f.write("ALL DETECTED ENTITY MENTIONS:\n")
        for ent in doc.ents:
            std = ent.ent_id_ if ent.ent_id_ else ent.text
            f.write(f"  [{ent.label_}] \"{ent.text}\" → {std}\n")
    print(f"  NER report saved to '{NER_REPORT}'")

    # --------------------------------------------------
    # Step 5: Translate using Gemini
    # --------------------------------------------------
    print(f"\n[Step 5/5] Translating to {target_language}...")

    try:
        translated = translate(lecture_text, preserve_str, context_str, target_language)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return
    except Exception as e:
        print(f"  ERROR during translation: {e}")
        return

    # Save the translated transcript
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(translated)

    print(f"  Translation saved to '{output_file}'")

    # --------------------------------------------------
    # Done — ask about audio
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRANSLATION COMPLETE!")
    print("=" * 60)
    print(f"\n  Language:    {target_language}")
    print(f"  Input:       {INPUT_FILE}")
    print(f"  NER Report:  {NER_REPORT}")
    print(f"  Output:      {output_file}")

    # Optional TTS step
    print()
    want_audio = input("Generate audio from the translated transcript? [y/N]: ").strip().lower()
    if want_audio == "y":
        voice_name = select_voice()
        safe_name  = target_language.lower().replace(" ", "_")
        audio_file = os.path.join(OUTPUT_DIR, f"lecture_{safe_name}.wav")

        print(f"\n[Bonus] Generating speech audio...")
        try:
            generate_audio(translated, audio_file, voice_name)
            print(f"\n  Audio saved to '{audio_file}'")
            print(f"  Open the WAV file with any media player (Windows Media Player,")
            print(f"  VLC, etc.) to listen to the translated lecture.")
        except Exception as e:
            print(f"  ERROR during audio generation: {e}")
    else:
        print("\nSkipping audio. Run again and press 'y' to generate it anytime.")

    print(f"\nTo translate different content, edit '{INPUT_FILE}' and run again.")



if __name__ == "__main__":
    main()