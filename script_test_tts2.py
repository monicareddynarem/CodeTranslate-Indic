import os, base64
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.8-flash-tts')
system_prompt = "You are a speech synthesis model. Use the voice 'Charon'."
resp = model.generate_content([system_prompt, 'Hello world'])
print('Got response, type:', type(resp))
# output_audio may be present in response.candidates[0].content.parts ... but per SDK maybe resp.output_audio
if hasattr(resp, 'output_audio'):
    data = base64.b64decode(resp.output_audio.data)
    print('Audio bytes length:', len(data))
else:
    # fallback parse
    print('No output_audio attribute')
