from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils import download_audio, transcribe_audio, summarize_text

app = FastAPI()


# Health check route
@app.get("/")
def root():
    return {"message": "TikTok Notes Backend is Live"}


# GET version for browser/manual testing
@app.get("/summarize")
def summarize_from_url(url: str):
    try:
        audio_path = download_audio(url)
        transcript = transcribe_audio(audio_path)
        summary = summarize_text(transcript)

        return {
            "transcript": transcript,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}


# Schema for POST request body
class SummarizeRequest(BaseModel):
    url: str


# POST version for mobile app/API integration
@app.post("/summarize")
def summarize_post(data: SummarizeRequest):
    try:
        audio_path = download_audio(data.url)
        transcript = transcribe_audio(audio_path)
        summary = summarize_text(transcript)

        return {
            "transcript": transcript,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}
