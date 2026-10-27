import streamlit as st
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH so imports work when running via streamlit
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Local modules
from src.driver import run_full_pipeline
from src.tts    import AVAILABLE_VOICES, VOICE_MAP
from src.slides import (
    extract_slide_texts,
    build_translation_input,
    split_translation_output,
    build_translated_pptx,
)
from src.translator import translate

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CodeTranslate‑Indic Demo", layout="wide")
st.title("📚 CodeTranslate‑Indic Demo")
st.markdown(
    """
    **Localized translation of Computer‑Science lectures** –
    upload an **English transcript** *or* **PowerPoint slides (.pptx)**, select a target
    Indic language, and get translated output with CS technical terms preserved or
    transliterated into the target script.
    """
)

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
mode = st.radio(
    "Input type",
    ["📄 Transcript (.txt)", "📊 Slides (.pptx)"],
    horizontal=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Input: text file (English transcript) – shown only in transcript mode
# ---------------------------------------------------------------------------
if mode == "📄 Transcript (.txt)":
    txt_file = st.file_uploader("Upload English transcript (.txt)", type=["txt"])
    pptx_file = None
else:
    txt_file  = None
    pptx_file = st.file_uploader("Upload PowerPoint slides (.pptx)", type=["pptx"])

if mode == "📄 Transcript (.txt)" and not txt_file:
    st.info("Please upload an English transcript (.txt) to continue.")
    st.stop()

if mode == "📊 Slides (.pptx)" and not pptx_file:
    st.info("Please upload a PowerPoint file (.pptx) to continue.")
    st.stop()

# ---------------------------------------------------------------------------
# Language + translation mode + voice selection
# ---------------------------------------------------------------------------
INDIC_LANGS = {
    "telugu":    "Telugu",
    "tamil":     "Tamil",
    "hindi":     "Hindi",
    "bengali":   "Bengali",
    "marathi":   "Marathi",
    "gujarati":  "Gujarati",
    "punjabi":   "Punjabi",
    "kannada":   "Kannada",
    "malayalam": "Malayalam",
}

col_lang, col_tmode, col_voice = st.columns(3)

with col_lang:
    lang_name = st.selectbox("Target Indic language", list(INDIC_LANGS.values()))
    lang_key  = next(k for k, v in INDIC_LANGS.items() if v.lower() == lang_name.lower())

with col_tmode:
    translation_mode = st.selectbox(
        "Translation mode",
        [
            "🔤 English terms preserved",
            "🅰️ Fully transliterated into target script",
        ],
        help=(
            "English terms preserved → CS terms stay in English, prose is translated.\n"
            "Fully transliterated → every CS term is also written in the target script."
        ),
    )
    transliterate = translation_mode.startswith("🅰️")

# Voice selector only shown in transcript mode (TTS not generated for slides)
with col_voice:
    if mode == "📄 Transcript (.txt)":
        default_voice = VOICE_MAP.get(lang_key, "Charon")
        default_idx   = AVAILABLE_VOICES.index(default_voice) if default_voice in AVAILABLE_VOICES else 0
        selected_voice = st.selectbox(
            "TTS Voice",
            AVAILABLE_VOICES,
            index=default_idx,
            help="Charon & Fenrir – male  |  Aoede, Kore & Leda – female  |  others – neutral",
        )
    else:
        selected_voice = VOICE_MAP.get(lang_key, "Charon")
        st.info("🔇 Audio generation is not available for slides mode.")

st.divider()

# ===========================================================================
# TRANSCRIPT MODE
# ===========================================================================
if mode == "📄 Transcript (.txt)":

    if st.button("Translate & Generate Audio", key="btn_transcript"):
        with st.spinner("Processing… (translating + audio synthesis)"):
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)

            def _update_progress(current: int, total: int) -> None:
                progress_bar.progress(int(current / total * 100))
                progress_placeholder.text(f"Generating audio chunk {current}/{total}…")

            # 1️⃣ Get English text from uploaded transcript
            english_text = txt_file.read().decode("utf-8")

            # 2️⃣ Run pipeline – single translation mode chosen by user
            translated_text, audio_path = run_full_pipeline(
                english_text=english_text,
                target_language=lang_key,
                transliterate=transliterate,
                voice_override=selected_voice,
                progress_callback=_update_progress,
            )

            progress_placeholder.empty()
            progress_bar.empty()

            # 3️⃣ Persist results
            st.session_state["result_transcript"] = {
                "english_text":    english_text,
                "translated_text": translated_text,
                "audio_path":      audio_path,
                "lang_name":       lang_name,
                "lang_key":        lang_key,
                "voice":           selected_voice,
                "mode_label":      translation_mode,
            }

    # ----- Display transcript results -----
    if "result_transcript" in st.session_state:
        r = st.session_state["result_transcript"]

        st.success("✅ Translation and audio generation completed!")

        tab_eng, tab_translated = st.tabs([
            "🇬🇧 English (original)",
            f"{r['mode_label']} – {r['lang_name']}",
        ])

        with tab_eng:
            st.text_area(
                "English transcript",
                value=r["english_text"],
                height=350,
                label_visibility="collapsed",
            )

        with tab_translated:
            st.text_area(
                "Translated transcript",
                value=r["translated_text"],
                height=350,
                label_visibility="collapsed",
            )

        # ----- Audio -----
        st.subheader(f"🔊 Synthesized audio – {r['lang_name']} · Voice: {r['voice']}")
        audio_file_path = Path(r["audio_path"])
        if audio_file_path.is_file():
            audio_bytes_out = audio_file_path.read_bytes()
            st.audio(audio_bytes_out, format="audio/wav")
            st.download_button(
                label="⬇️ Download audio",
                data=audio_bytes_out,
                file_name=f"lecture_{r['lang_key']}_{r['voice']}.wav",
                mime="audio/wav",
            )
        else:
            st.warning("Audio file not found. Please run the pipeline again.")


# ===========================================================================
# SLIDES MODE
# ===========================================================================
else:
    if st.button("Translate Slides", key="btn_slides"):
        pptx_bytes = pptx_file.read()

        with st.spinner("Step 1 / 3 – Extracting text from slides…"):
            slide_texts = extract_slide_texts(pptx_bytes)
            total_boxes = sum(
                1 for slide in slide_texts for box in slide if not box["is_code"]
            )
            st.info(
                f"Found **{len(slide_texts)} slides** with "
                f"**{total_boxes} translatable text boxes**."
            )

        with st.spinner(f"Step 2 / 3 – Translating ({translation_mode})…"):
            translation_input = build_translation_input(slide_texts)

            from src.lexicon    import generate_lexicon
            from src.pipeline   import setup_ner_pipeline
            from src.classifier import classify_entities, format_term_lists
            from pathlib import Path as _Path

            DICT_DIR = _Path(__file__).resolve().parent / "dictionaries"
            patterns, ambiguous_ids, entry_map = generate_lexicon(str(DICT_DIR))
            nlp = setup_ner_pipeline(patterns)
            doc = nlp(translation_input)
            must_preserve, context_dependent = classify_entities(doc, ambiguous_ids, entry_map)
            preserve_str, context_str = format_term_lists(must_preserve, context_dependent)

            # Single translation call based on user's chosen mode
            translated_str = translate(
                text=translation_input,
                preserve_str=preserve_str,
                context_str=context_str,
                target_language=lang_key,
                transliterate=transliterate,
            )

        with st.spinner("Step 3 / 3 – Building translated PPTX…"):
            replacements = split_translation_output(translated_str, slide_texts)
            pptx_out_bytes = build_translated_pptx(pptx_bytes, slide_texts, replacements)

        # Persist so download button survives re-runs
        st.session_state["result_slides"] = {
            "lang_name":       lang_name,
            "lang_key":        lang_key,
            "pptx_out_bytes":  pptx_out_bytes,
            "num_slides":      len(slide_texts),
            "num_boxes":       total_boxes,
            "mode_label":      translation_mode,
            "transliterate":   transliterate,
        }

    # ----- Display slides results -----
    if "result_slides" in st.session_state:
        r = st.session_state["result_slides"]

        st.success(
            f"✅ Slides translated successfully! "
            f"({r['num_slides']} slides, {r['num_boxes']} text boxes)"
        )

        st.markdown(
            f"**Mode:** {r['mode_label']}  \n"
            "The **layout, fonts, colors, and images** are identical to the original — "
            "only the text has been replaced."
        )

        fname_suffix = "transliterated" if r["transliterate"] else "normal"
        st.download_button(
            label=f"⬇️ Download translated PPTX ({r['lang_name']})",
            data=r["pptx_out_bytes"],
            file_name=f"slides_{r['lang_key']}_{fname_suffix}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True,
        )
