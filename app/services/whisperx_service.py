import os
from pathlib import Path
from dotenv import load_dotenv
import whisperx
from whisperx.diarize import DiarizationPipeline

load_dotenv()

class WhisperXService:
    def __init__(self, model_size="base"):
        self.device = "cpu"  
        self.compute_type = "int8" 
        self.model_size = model_size
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

    def process(self, audio_path: Path):
        """
        Runs the full pipeline: Transcribe -> Align -> Diarize
        """
        print(f"Loading audio: {audio_path}")
        
        # 1. Load Audio
        audio = whisperx.load_audio(str(audio_path))

        # 2. Load Model
        print("Loading Whisper model...")
        model = whisperx.load_model(
            self.model_size, 
            self.device, 
            compute_type=self.compute_type
        )

        # 3. Transcribe
        print("Transcribing...")
        result = model.transcribe(audio, batch_size=8)

        # 4. Align
        print("Aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=self.device
        )
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            self.device
        )

        # 5. Diarization
        print("Running Diarization...")
        
        diarize_model = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1",
            token=self.hf_token, 
            device=self.device
        )
        
        # FIX: Allow 3 speakers.
        # If the model is confused, it will put the 'confused agent' in SPEAKER_02.
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=3)

        # 6. Assign Speakers
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 7. Format Output & Merge Logic
        final_segments = []
        for seg in result["segments"]:
            speaker = seg.get("speaker", "UNKNOWN")
            
            # --- THE HEALING LOGIC ---
            # If the model detected a 3rd speaker, we assume it is a "split" version of the Agent.
            # We merge SPEAKER_02 into SPEAKER_01.
            # NOTE: If SPEAKER_02 is actually the Customer in your result, change this to "SPEAKER_00".
            if speaker == "SPEAKER_02":
                speaker = "SPEAKER_01"
            
            final_segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "speaker": speaker,
                "text": seg["text"].strip()
            })

        return final_segments