import os
import requests
import uuid

# Step 1: Download TikTok Audio via Third-Party API
def download_audio(tiktok_url):
    audio_id = str(uuid.uuid4())
    output_path = f"audio/{audio_id}.mp3"

    # Replace this with your actual third-party downloader URL (if required)
    downloader_api = "https://tiktokdownloader.p.rapidapi.com/download"
    headers = {
        "X-RapidAPI-Key": os.environ["RAPIDAPI_KEY"],  # Set this in your environment on Render
        "X-RapidAPI-Host": "tiktokdownloader.p.rapidapi.com"
    }
    params = {"url": tiktok_url}

    response = requests.get(downloader_api, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        audio_url = data.get("music") or data.get("music_url")
        if not audio_url:
            raise Exception("Audio URL not found in downloader response.")
        
        # Download the actual audio file
        audio_data = requests.get(audio_url)
        with open(output_path, "wb") as f:
            f.write(audio_data.content)
        return output_path
    else:
        raise Exception(f"Downloader API failed: {response.status_code} - {response.text}")

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
