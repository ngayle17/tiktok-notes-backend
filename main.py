from fastapi import FastAPI, Request
from utils import download_audio, transcribe_audio, summarize_text

app = FastAPI()

@app.get("/")
def root():
    return {"message": "TikTok Notes Backend is Live"}

@app.get("/summarize")
def summarize_from_url(url: str):
    try:
        # Optional: log what’s being processed
        # print(f"Received URL: {url}")
        
        audio_path = download_audio(url)
        transcript = transcribe_audio(audio_path)
        summary = summarize_text(transcript)

        return {
            "transcript": transcript,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}
