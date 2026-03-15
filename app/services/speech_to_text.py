from pathlib import Path
import whisper


class SpeechToText:

    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: Path):

        result = self.model.transcribe(str(audio_path))

        segments = []

        for seg in result["segments"]:
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            })

        return segments