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

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CodeTranslate‑Indic Demo", layout="wide")
st.title("📚 CodeTranslate‑Indic Demo")
st.markdown(
    """
    **Localized translation of Computer‑Science lectures** –
    upload an **English transcript**, select a target Indic language, and get:
    1. the original English transcript,
    2. a translated transcript **with English technical terms preserved**,
    3. a translated transcript **with all terms written in the target script**,
    4. a synthetic audio version of the translation.
    """
)

# ---------------------------------------------------------------------------
# Input: text file (English transcript)
# ---------------------------------------------------------------------------
txt_file = st.file_uploader("Upload English transcript (txt)", type=["txt"])

if not txt_file:
    st.info("Please upload an English transcript (.txt) to continue.")
    st.stop()

# ---------------------------------------------------------------------------
# Language + voice selection
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

col_lang, col_voice = st.columns(2)
with col_lang:
    lang_name = st.selectbox("Target Indic language", list(INDIC_LANGS.values()))
    lang_key  = next(k for k, v in INDIC_LANGS.items() if v.lower() == lang_name.lower())

with col_voice:
    default_voice = VOICE_MAP.get(lang_key, "Charon")
    default_idx   = AVAILABLE_VOICES.index(default_voice) if default_voice in AVAILABLE_VOICES else 0
    selected_voice = st.selectbox(
        "TTS Voice",
        AVAILABLE_VOICES,
        index=default_idx,
        help="Charon & Fenrir – male  |  Aoede, Kore & Leda – female  |  others – neutral",
    )

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
if st.button("Translate & Generate Audio"):
    with st.spinner("Processing… (this runs two translations + audio synthesis)"):
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)

        def _update_progress(current: int, total: int) -> None:
            progress_bar.progress(int(current / total * 100))
            progress_placeholder.text(f"Generating audio chunk {current}/{total}…")

        # 1️⃣ Get English text from uploaded transcript
        english_text = txt_file.read().decode("utf-8")

        # 2️⃣ Run pipeline → produces both transcript versions + audio in one go
        normal_text, transliterated_text, audio_path = run_full_pipeline(
            english_text=english_text,
            target_language=lang_key,
            voice_override=selected_voice,
            progress_callback=_update_progress,
        )

        progress_placeholder.empty()
        progress_bar.empty()

        # 3️⃣ Persist results so they survive Streamlit page re-runs
        st.session_state["result"] = {
            "english_text":       english_text,
            "normal_text":        normal_text,
            "transliterated_text": transliterated_text,
            "audio_path":         audio_path,
            "lang_name":          lang_name,
            "lang_key":           lang_key,
            "voice":              selected_voice,
        }

# ---------------------------------------------------------------------------
# Display results – outside the button block so they survive re-runs
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    r = st.session_state["result"]

    st.success("✅ Translation and audio generation completed!")

    # ----- Transcripts in three tabs -----
    tab_eng, tab_normal, tab_trans = st.tabs([
        "🇬🇧 English (original)",
        f"🔤 {r['lang_name']} – English terms preserved",
        f"🅰️ {r['lang_name']} – Fully in {r['lang_name']} script",
    ])

    with tab_eng:
        st.text_area(
            "English transcript",
            value=r["english_text"],
            height=350,
            label_visibility="collapsed",
        )

    with tab_normal:
        st.caption("Technical CS terms are kept in English; surrounding prose is in " + r["lang_name"] + ".")
        st.text_area(
            "Normal translation",
            value=r["normal_text"],
            height=350,
            label_visibility="collapsed",
        )

    with tab_trans:
        st.caption(
            "All technical CS terms have been phonetically transliterated into the "
            + r["lang_name"] + " script — no English words appear."
        )
        st.text_area(
            "Transliterated translation",
            value=r["transliterated_text"],
            height=350,
            label_visibility="collapsed",
        )

    # ----- Audio (generated from the normal-text version) -----
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
