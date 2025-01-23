import subprocess
import os
import whisper
import torch

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
 
model = whisper.load_model("base", device=device)

def download_audio(youtube_url, output_path="audio.mp3"):
    """Download the audio from a YouTube video."""
    try:
        command = [
            "yt-dlp",
            "--format", "bestaudio",
            "--extract-audio",
            "--audio-format", "mp3",
            "--output", output_path,
            youtube_url,
        ]
        subprocess.run(command, check=True)
        print(f"Audio downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribe the audio using Whisper."""
    try:
        print("Extracting transcript")
        result = model.transcribe(audio_path)
        return result  # Contains text and timestamps
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def save_transcript_with_timestamps(transcript, output_file="transcript.txt"):
    """Save the transcript with timestamps to a file."""
    try:
        with open(output_file, "w") as file:
            for segment in transcript['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']
                file.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")
        print(f"Transcript saved to {output_file}")
    except Exception as e:
        print(f"Error saving transcript: {e}")

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=DmgGGUYn2c8"
    audio_file = download_audio(youtube_url)
    
    if audio_file:
 
        transcript = transcribe_audio(audio_file)
        
        if transcript:
 
            save_transcript_with_timestamps(transcript)
 
        os.remove(audio_file)
