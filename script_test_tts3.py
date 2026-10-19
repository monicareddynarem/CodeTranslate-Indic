import os, base64
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.8-flash-tts')
system_prompt = "You are a speech synthesis model. Use the voice 'Charon'."
response = model.generate_content([system_prompt, 'Hello world'])
# Extract audio data from inline_data parts
audio_bytes = None
for candidate in response.candidates:
    for part in candidate.content.parts:
        if getattr(part, 'inline_data', None) and part.inline_data.mime_type == 'audio/wav':
            audio_bytes = base64.b64decode(part.inline_data.data)
            break
    if audio_bytes:
        break
if audio_bytes:
    print('Got audio bytes length:', len(audio_bytes))
else:
    print('Audio data not found in response')
