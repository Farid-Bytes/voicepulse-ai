import soundfile as sf
# import numpy as np
from pathlib import Path
from typing import List, Dict

class SegmentExtractor:
    SEGMENTS_DIR = Path("data/segments")

    def __init__(self):
        self.SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    def extract(self, audio_path: Path, segments: List[Dict]) -> List[Dict]:
        """
        Slices the audio file based on segment timestamps.
        Adds the audio array directly to the segment dict (for fast processing).
        Saves each segment as a separate .wav file.
        """
        print(f"Extracting segments from: {audio_path}")
        
        # 1. Load the full audio file
        audio, sr = sf.read(str(audio_path))

        # 2. Iterate and slice
        for i, seg in enumerate(segments):
            start_time = seg["start"]
            end_time = seg["end"]
            
            # Convert time (seconds) to samples (indices)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Slice the numpy array
            audio_chunk = audio[start_sample : min(end_sample, len(audio))]

            # --- OPTIMIZATION ---
            # Store audio in memory for the next step (Emotion Detection)
            # This prevents the Emotion Detector from reading from disk again.
            seg["audio_array"] = audio_chunk
            seg["sample_rate"] = sr
            # --------------------

            # Create filename: segment_001_SPEAKER_00.wav
            filename = f"segment_{i:03d}_{seg['speaker']}.wav"
            output_path = self.SEGMENTS_DIR / filename

            # Save the chunk (Optional: keeps files for debugging, but takes time)
            sf.write(str(output_path), audio_chunk, sr)

            # Add the path to our data dictionary
            seg["segment_audio_path"] = str(output_path)

        print(f"Extracted {len(segments)} segments to {self.SEGMENTS_DIR}")
        return segments