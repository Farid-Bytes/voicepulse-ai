from pathlib import Path
import whisper


class WhisperTranscriber:

    def __init__(self, model_size: str = "base"):
        """
        Load Whisper model once
        """
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: Path):
        """
        Convert speech to text with timestamps
        """

        result = self.model.transcribe(str(audio_path), word_timestamps=True)

        segments = []

        for seg in result["segments"]:
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            })

        return segments