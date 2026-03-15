from pathlib import Path
from app.services.speech_to_text import SpeechToText

audio_path = Path("data/processed/call_processed.wav")

stt = SpeechToText()

segments = stt.transcribe(audio_path)

for s in segments:
    print(s)