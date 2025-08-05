from fastapi import FastAPI, Request
from utils import download_audio, transcribe_audio, summarize_text

app = FastAPI()

@app.get("/summarize")
def summarize_from_url(url: str):
    audio_path = download_audio(url)
    transcript = transcribe_audio(audio_path)
    summary = summarize_text(transcript)
    return {"summary": summary}
