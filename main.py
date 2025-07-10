from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils import download_audio, transcribe_audio, summarize_text

app = FastAPI()

class VideoURL(BaseModel):
    url: str

@app.post("/process")
async def process_video(data: VideoURL):
    audio_path = download_audio(data.url)
    transcript = transcribe_audio(audio_path)
    summary = summarize_text(transcript)
    return {
        "transcript": transcript,
        "summary": summary
    }
