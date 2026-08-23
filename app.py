import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from gradio_client import Client, handle_file

app = FastAPI()

# HuggingFace Client Connection
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
    
    # নতুন এবং সঠিক প্যারামিটার নেম অনুযায়ী সাজানো কোড
    result = hf_client.predict(
        ref_audio_orig=handle_file(REF_AUDIO),
        ref_text_input=REF_TEXT,
        gen_text_input=script_text,
        remove_silence=True,
        cross_fade_duration=0.15,
        nfe_step=32,
        speed=1.0,
        api_name="/infer"
    )
    return FileResponse(result[0], media_type="audio/wav", filename="output.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
