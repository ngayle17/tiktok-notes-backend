from transformers import pipeline
import yt_dlp
import whisper
import os
import uuid

# Initialize Whisper and Summarizer
model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def download_audio(url):
    filename = f"audio/{uuid.uuid4()}.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename.replace('.mp3', '.webm'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return filename

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

def summarize_text(text):
    # Hugging Face models have a max input length; we'll chunk if needed
    if len(text) < 1000:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    else:
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summary_parts = []
        for chunk in chunks:
            result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summary_parts.append(result)
        summary = " ".join(summary_parts)
    return summary
