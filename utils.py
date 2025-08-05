import os
from huggingface_hub import login
import requests
import yt_dlp
import uuid
import os

# Step 1: Download TikTok Audio
def download_audio(url):
    audio_id = str(uuid.uuid4())
    output_path = f"audio/{audio_id}.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

# Step 2: Transcribe via Hugging Face Whisper
def transcribe_audio(audio_path):
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
    headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

    with open(audio_path, "rb") as f:
        response = requests.post(API_URL, headers=headers, data=f)

    if response.status_code == 200:
        return response.json().get("text", "No transcript found.")
    else:
        return f"Error: {response.status_code} - {response.text}"

# Step 3: Summarize via Hugging Face BART
def summarize_text(text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}
    payload = {"inputs": text}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0].get("summary_text", "No summary generated.")
    else:
        return f"Error: {response.status_code} - {response.text}"
