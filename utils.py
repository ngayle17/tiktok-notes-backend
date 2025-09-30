import os
import requests
import uuid

# Step 1: Download TikTok Audio via Third-Party API
def download_audio(tiktok_url):
    audio_id = str(uuid.uuid4())
    output_path = f"audio/{audio_id}.mp3"

    # Use TikWM API to get download links
    api_url = "https://api.tikwm.com/"
    params = {"url": tiktok_url}
    response = requests.get(api_url, params=params)

    if response.status_code != 200:
        raise Exception("Failed to get download link from TikWM.")

    data = response.json()
    if not data.get("data") or not data["data"].get("music"):
        raise Exception("TikWM did not return a valid music link.")

    music_url = data["data"]["music"]

    # Download the MP3 audio file
    audio_response = requests.get(music_url)
    if audio_response.status_code != 200:
        raise Exception("Failed to download audio file.")

    os.makedirs("audio", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_response.content)

    return output_path
