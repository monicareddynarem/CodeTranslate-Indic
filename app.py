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
from src.tts import AVAILABLE_VOICES, VOICE_MAP

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
    lang_key = next(k for k, v in INDIC_LANGS.items() if v.lower() == lang_name.lower())

with col_voice:
    # Show default voice for the selected language, but allow override
    default_voice = VOICE_MAP.get(lang_key, "Charon")
    default_idx = AVAILABLE_VOICES.index(default_voice) if default_voice in AVAILABLE_VOICES else 0
    selected_voice = st.selectbox(
        "TTS Voice",
        AVAILABLE_VOICES,
        index=default_idx,
        help="Charon & Fenrir are male; Aoede, Kore & Leda are female; others are neutral.",
    )

transliterate = st.checkbox("Transliterate technical terms into the target script", value=False)

if st.button("Translate & Generate Audio"):
    with st.spinner("Processing…"):
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)

        def _update_progress(current: int, total: int) -> None:
            progress = int(current / total * 100)
            progress_bar.progress(progress)
            progress_placeholder.text(f"Generating audio chunk {current}/{total}…")

        # 1️⃣ Get English text
        if txt_file:
            english_text = txt_file.read().decode("utf-8")
        else:
            audio_bytes_input = audio_file.read()
            mime_type = "audio/mpeg" if audio_file.name.lower().endswith(".mp3") else "audio/wav"
            english_text = speech_to_text(audio_bytes_input, mime_type=mime_type)
            if not english_text:
                st.error("Speech‑to‑text failed – please try a different audio file.")
                st.stop()

        # 2️⃣ Run the full pipeline (NER → translation → TTS)
        indic_text, audio_path = run_full_pipeline(
            english_text=english_text,
            target_language=lang_key,
            transliterate=transliterate,
            voice_override=selected_voice,
            progress_callback=_update_progress,
        )

        progress_placeholder.empty()
        progress_bar.empty()

        # 3️⃣ Save results in session_state so they survive Streamlit re-runs
        st.session_state["result"] = {
            "english_text": english_text,
            "indic_text":   indic_text,
            "audio_path":   audio_path,
            "lang_name":    lang_name,
            "lang_key":     lang_key,
            "voice":        selected_voice,
        }

# ---------------------------------------------------------------------------
# Display results – outside the button block so they persist on page re-runs
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    r = st.session_state["result"]

    st.success("✅ Translation and audio generation completed!")

    col_e, col_i = st.columns(2)
    with col_e:
        st.subheader("English transcript")
        st.text_area("eng", value=r["english_text"], height=300, label_visibility="collapsed")
    with col_i:
        st.subheader(f"{r['lang_name']} transcript")
        st.text_area("ind", value=r["indic_text"], height=300, label_visibility="collapsed")

    # Audio playback and download
    st.subheader(f"Synthesized audio – {r['lang_name']} · Voice: {r['voice']}")
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
