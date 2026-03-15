import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple


class AudioPreprocessor:
    TARGET_SAMPLE_RATE = 16000
    PROCESSED_DIR = Path("data/processed")

    def __init__(self):
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def preprocess(self, input_path: Path) -> Path:
        """
        Standard Mono Processing:
        - Converts to Mono (mixes channels).
        - Resamples to 16kHz.
        - Normalizes volume.
        
        WARNING: For stereo call recordings, this mixes speakers together.
        Use 'preprocess_stereo_split' instead for stereo files.
        """
        
        # Load audio (force mono + target sr)
        audio, sr = librosa.load(
            input_path,
            sr=self.TARGET_SAMPLE_RATE,
            mono=True
        )

        # Safe normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Output path
        output_path = self.PROCESSED_DIR / f"{input_path.stem}_processed.wav"

        # Save
        sf.write(output_path, audio, self.TARGET_SAMPLE_RATE)

        return output_path

    def preprocess_stereo_split(self, input_path: Path) -> Tuple[Path, Path]:
        """
        Stereo Split Processing:
        - Detects stereo audio.
        - Splits Left and Right channels into separate mono files.
        - Returns tuple: (left_channel_path, right_channel_path).
        
        Best for call center recordings where Agent/Customer are on separate channels.
        """

        # Load audio without mixing (mono=False)
        audio, sr = librosa.load(input_path, sr=self.TARGET_SAMPLE_RATE, mono=False)

        # Check if actually stereo
        if audio.ndim == 1:
            print("Warning: Input file is Mono. Use .preprocess() instead. Returning duplicate paths.")
            path = self.preprocess(input_path)
            return path, path

        # audio shape is (2, samples) -> [Left, Right]
        left_channel = audio[0]
        right_channel = audio[1]

        # Helper to normalize
        def _normalize(x):
            m = np.max(np.abs(x))
            return x / m if m > 0 else x

        left_channel = _normalize(left_channel)
        right_channel = _normalize(right_channel)

        # Save Left Channel
        left_path = self.PROCESSED_DIR / f"{input_path.stem}_left.wav"
        sf.write(left_path, left_channel, self.TARGET_SAMPLE_RATE)

        # Save Right Channel
        right_path = self.PROCESSED_DIR / f"{input_path.stem}_right.wav"
        sf.write(right_path, right_channel, self.TARGET_SAMPLE_RATE)

        print(f"Stereo file split successfully:\n  Left: {left_path}\n  Right: {right_path}")

        return left_path, right_path