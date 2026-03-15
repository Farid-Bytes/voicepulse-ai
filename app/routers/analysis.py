from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from app.services.audio_preprocessing import AudioPreprocessor
from app.services.diarization import SpeakerSegmenter
# Add these imports
from app.services.transcription import WhisperTranscriber 
from app.services.speaker_text_align import SpeakerTextAligner

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):

    # 1. Save uploaded file
    upload_path = UPLOAD_DIR / file.filename
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Preprocess
    preprocessor = AudioPreprocessor()
    processed_path = preprocessor.preprocess(upload_path)

    # 3. Diarization (Who spoke when?)
    segmenter = SpeakerSegmenter()
    diarization_segments = segmenter.segment(processed_path)

    # --- NEW CODE STARTS HERE --

    # 4. Transcription (What was said?)
    transcriber = WhisperTranscriber() # or SpeechToText()
    transcription_segments = transcriber.transcribe(processed_path)

    # 5. Alignment (Combine Who + What)
    aligned_segments = SpeakerTextAligner.align(
        diarization_segments,
        transcription_segments
    )

    # --- NEW CODE ENDS HERE --

    return {
        "status": "processed",
        "processed_file": str(processed_path),
        "segments": aligned_segments # Returns text + speaker now
    }