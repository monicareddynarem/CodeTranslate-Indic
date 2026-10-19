"""
tts.py
-------
Converts the translated text transcript to speech using the Gemini TTS API.
Produces a single WAV file that reads out the translated lecture.

Handles long transcripts by splitting into sentence-level chunks
(the TTS model has a 16,384 token input limit).
"""

import base64
import os
import re
import time
from google import genai


# Voices that work well for lecture-style speech (calm, clear, educational).
# The user can choose one when running.
AVAILABLE_VOICES = {
    "1": ("Charon",  "deep, authoritative — good for lectures"),
    "2": ("Kore",    "natural, warm — good for explanations"),
    "3": ("Fenrir",  "clear, steady — good for technical content"),
    "4": ("Aoede",   "smooth, expressive"),
    "5": ("Puck",    "bright, energetic"),
}

# Approx character limit per chunk to stay well under the token limit.
# ~12,000 chars ≈ ~3,000 tokens, safe buffer below the 16,384 limit.
CHUNK_CHAR_LIMIT = 12_000


def select_voice():
    """Prompts the user to choose a TTS voice."""
    print("\nAvailable voices for speech:")
    for num, (name, desc) in AVAILABLE_VOICES.items():
        print(f"  [{num}]  {name:10s} — {desc}")
    print()

    while True:
        choice = input("Choose a voice number (or press Enter for default 'Charon'): ").strip()
        if choice == "":
            return "Charon"
        if choice in AVAILABLE_VOICES:
            return AVAILABLE_VOICES[choice][0]
        print(f"  Please enter a number between 1 and {len(AVAILABLE_VOICES)}.")


def split_into_chunks(text, limit=CHUNK_CHAR_LIMIT):
    """
    Splits text into chunks at sentence boundaries so each chunk
    stays within the TTS model's token limit.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[।.!?])\s+', text.strip())

    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= limit:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence is itself too long, split at paragraph breaks
            if len(sentence) > limit:
                parts = sentence.split("\n\n")
                for part in parts:
                    if part.strip():
                        chunks.append(part.strip())
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def _get_api_key():
    """Reads the API key from config.py or the environment variable."""
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
            "No API key found. Open config.py and paste your key there."
        )
    return api_key


def _tts_chunk(client, text_chunk, voice_name, chunk_index, total_chunks):
    """
    Calls the Gemini TTS API for a single text chunk.
    Returns raw WAV bytes.
    """
    MAX_RETRIES = 3
    WAIT_SECONDS = [5, 10, 20]

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            wait = WAIT_SECONDS[attempt - 1]
            print(f"    Retrying chunk {chunk_index}/{total_chunks} in {wait}s...")
            time.sleep(wait)

        try:
            interaction = client.interactions.create(
                model="gemini-3.8-flash-tts",
                input=[{
                    "type": "user_input",
                    "content": [{
                        "type": "text",
                        "text": text_chunk,
                        "annotations": [{
                            "type": "speech_metadata",
                            "style": "clear and educational, like a university professor",
                        }],
                    }],
                }],
                response_format={"type": "audio"},
                generation_config={
                    "speech_config": [
                        {"voice": voice_name},
                    ]
                },
                store=False,
            )

            audio = interaction.output_audio
            if not audio or not audio.data:
                raise RuntimeError("TTS returned empty audio.")

            return base64.b64decode(audio.data)

        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in ["503", "429", "500", "service_unavailable", "overloaded"]):
                if attempt < MAX_RETRIES - 1:
                    continue
                raise RuntimeError(
                    f"TTS API unavailable after {MAX_RETRIES} retries. "
                    f"Please try again later.\nLast error: {err_str}"
                ) from e
            raise  # non-transient error, fail immediately


def _merge_wav_chunks(wav_chunks):
    """
    Merges multiple WAV byte-strings into one by concatenating PCM data.
    All chunks from Gemini TTS share the same format (24000 Hz, 16-bit, mono).
    """
    import struct

    def read_wav_header(data):
        """Returns (header_size, data_size) from a WAV byte string."""
        # Find the 'data' sub-chunk
        data_pos = data.find(b'data')
        if data_pos == -1:
            raise ValueError("Invalid WAV: no 'data' chunk found.")
        data_size = struct.unpack_from('<I', data, data_pos + 4)[0]
        return data_pos + 8, data_size

    pcm_parts = []
    for chunk in wav_chunks:
        header_size, _ = read_wav_header(chunk)
        pcm_parts.append(chunk[header_size:])

    combined_pcm = b"".join(pcm_parts)

    # Rebuild a single WAV header
    sample_rate    = 24000
    num_channels   = 1
    bits_per_sample = 16
    byte_rate  = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size  = len(combined_pcm)

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,              # sub-chunk size
        1,               # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size,
    )

    return header + combined_pcm


def generate_audio(translated_text, output_wav_path, voice_name):
    """
    Converts the translated transcript to a WAV audio file.

    Args:
        translated_text: the full translated text
        output_wav_path: path to save the output .wav file
        voice_name:      Gemini TTS voice name (e.g. "Charon")
    """
    api_key = _get_api_key()
    client = genai.Client(api_key=api_key)

    chunks = split_into_chunks(translated_text)
    total = len(chunks)

    if total == 1:
        print(f"  Generating audio (1 chunk, voice: {voice_name})...")
    else:
        print(f"  Generating audio in {total} chunks (voice: {voice_name})...")

    wav_chunks = []
    for i, chunk in enumerate(chunks, 1):
        if total > 1:
            print(f"  Processing chunk {i}/{total}...")
        wav_data = _tts_chunk(client, chunk, voice_name, i, total)
        wav_chunks.append(wav_data)

    print("  Merging audio chunks..." if total > 1 else "  Saving audio...")
    final_wav = _merge_wav_chunks(wav_chunks)

    with open(output_wav_path, "wb") as f:
        f.write(final_wav)

    size_kb = len(final_wav) / 1024
    print(f"  Audio saved to '{output_wav_path}' ({size_kb:.0f} KB)")
