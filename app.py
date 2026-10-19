import streamlit as st
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH so imports work when running via streamlit
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Local modules
from src.stt import speech_to_text
from src.driver import run_full_pipeline

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CodeTranslate‑Indic Demo", layout="wide")
st.title("📚 CodeTranslate‑Indic Demo")
st.markdown(
    """
    **Localized translation of Computer‑Science lectures** –
    upload an **English audio file** *or* an **English transcript**, select a target Indic language, and get:
    1. the original English transcript,
    2. the translated Indic transcript (with technical terms preserved or transliterated),
    3. a synthetic audio version of the Indic transcript.
    """
)

# ---------------------------------------------------------------------------
# Input: audio OR text file
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload English audio (wav/mp3)", type=["wav", "mp3"])
with col2:
    txt_file = st.file_uploader("Upload English transcript (txt)", type=["txt"])

if not (audio_file or txt_file):
    st.info("Please upload an audio file **or** a transcript to continue.")
    st.stop()

# ---------------------------------------------------------------------------
# Language selection & options
# ---------------------------------------------------------------------------
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

lang_name = st.selectbox("Target Indic language", list(INDIC_LANGS.values()))
# Find the key that matches the selected name (case‑insensitive)
lang_key = next(k for k, v in INDIC_LANGS.items() if v.lower() == lang_name.lower())

transliterate = st.checkbox("Transliterate technical terms into the target script", value=False)

if st.button("Translate & Generate Audio"):
    # UI feedback while processing
    with st.spinner("Processing…"):
        # Progress bar placeholder
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)

        def _update_progress(current: int, total: int) -> None:
            # Update Streamlit progress bar (0‑100)
            progress = int(current / total * 100)
            progress_bar.progress(progress)
            progress_placeholder.text(f"Processing chunk {current}/{total}…")

        # 1️⃣ Get English text
        if txt_file:
            english_text = txt_file.read().decode("utf-8")
        else:
            # audio → STT
            audio_bytes = audio_file.read()
            english_text = speech_to_text(audio_bytes)
            if not english_text:
                st.error("Speech‑to‑text failed – please try a different audio file.")
                st.stop()

        # 2️⃣ Run the full pipeline (NER → translation → TTS) with progress callback
        indic_text, audio_path = run_full_pipeline(
            english_text=english_text,
            target_language=lang_key,
            transliterate=transliterate,
            progress_callback=_update_progress,
        )
        # Clear progress UI after completion
        progress_placeholder.empty()
        progress_bar.empty()

    # -------------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------------
    st.success("✅ Translation and audio generation completed!")
    col_e, col_i = st.columns(2)
    with col_e:
        st.subheader("English transcript")
        st.text_area("eng", value=english_text, height=300)
    with col_i:
        st.subheader(f"{lang_name} transcript")
        st.text_area("ind", value=indic_text, height=300)

    # Audio playback and download
    st.subheader("Synthesized audio (Indic)")
    audio_bytes = Path(audio_path).read_bytes()
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        label="Download audio",
        data=audio_bytes,
        file_name=f"lecture_{lang_key}.wav",
        mime="audio/wav",
    )
