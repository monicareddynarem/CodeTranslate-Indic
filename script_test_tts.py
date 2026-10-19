import os, base64
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.8-flash-tts')
cfg = genai.GenerationConfig(response_mime_type='audio/wav')
resp = model.generate_content([
    "You are a speech synthesis model. Use the voice 'Charon'.",
    "Hello world"
], generation_config=cfg)
print('Audio bytes length:', len(base64.b64decode(resp.output_audio.data)))
