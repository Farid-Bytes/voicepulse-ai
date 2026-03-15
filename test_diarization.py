from pathlib import Path
from app.services.diarization import SpeakerSegmenter
import soundfile as sf

audio_path = Path("data/processed/call_processed.wav")

segmenter = SpeakerSegmenter()
segments = segmenter.segment(audio_path)

for seg in segments:
    print(seg)

audio, sr = sf.read(audio_path)
print("Duration:", len(audio) / sr, "seconds")