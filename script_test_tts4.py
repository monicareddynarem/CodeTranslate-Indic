import os, base64
import google.generativeai as genai
from config import GEMINI_API_KEY
from src.tts import _tts_chunk, _choose_voice

genai.configure(api_key=GEMINI_API_KEY)
client = genai.GenerativeModel('gemini-3.8-flash-tts')
voice = _choose_voice('telugu')
text = 'Hello world, this is a test of Gemini TTS.'
try:
    wav = _tts_chunk(client, text, voice, 1, 1)
    print('Audio bytes length:', len(wav))
except Exception as e:
    print('Error during TTS:', e)
