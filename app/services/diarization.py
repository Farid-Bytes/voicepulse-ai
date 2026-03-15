import os
from pathlib import Path
from dotenv import load_dotenv

import soundfile as sf
import torch
import numpy as np

from pyannote.audio import Pipeline

load_dotenv()


class SpeakerSegmenter:

    def __init__(self):
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN not set")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token
        )

    def segment(self, audio_path: Path):
        # Load the processed MONO file
        audio, sample_rate = sf.read(str(audio_path))

        # Ensure it's treated as mono (shape: [samples])
        if len(audio.shape) > 1:
             # Fallback just in case
            audio = np.mean(audio, axis=1)

        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        audio_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        # --- CRITICAL FIX ---
        # We add 'segmentation_onset=0.3' (default is usually higher).
        # This forces the model to detect speaker changes faster.
        # We also force exactly 2 speakers.
        diarization = self.pipeline(
            audio_dict,
            min_speakers=2,
            max_speakers=2,
            segmentation_onset=0.3
        )

        # Handle pyannote version differences
        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization

        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": speaker
            })

        return segments