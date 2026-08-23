import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from gradio_client import Client, handle_file

app = FastAPI()

# HuggingFace Client
hf_client = Client("mrfakename/E2-F5-TTS")

REF_AUDIO = "my_voice.mp3"
REF_TEXT = "Welcome back to the channel. Today, we are diving deep into one of the most fascinating topics in science and technology. Make sure to hit the subscribe button, and let's explore what the future holds."

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/generate-audio")
async def generate_audio(request: Request):
    data = await request.json()
    script_text = data.get("text", "")
    
    # api_name এবং কি-ওয়ার্ড আর্গুমেন্ট ছাড়া সরাসরি পজিশনাল কল
    result = hf_client.predict(
        handle_file(REF_AUDIO),  # 1. ref_audio
        REF_TEXT,                # 2. ref_text
        script_text,             # 3. gen_text
        True,                    # 4. remove_silence
        0.15,                    # 5. cross_fade_duration
        32,                      # 6. nfe_step
        1.0                      # 7. speed
    )
    return FileResponse(result[0], media_type="audio/wav", filename="output.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
