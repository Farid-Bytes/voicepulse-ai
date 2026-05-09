from fastapi import APIRouter, Request, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, RedirectResponse # Add RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil

# Import Security
from app.auth import get_current_user # <--- IMPORT THIS
from app.models import User           # <--- IMPORT THIS

# Import all your services
from app.services.whisperx_service import WhisperXService
from app.services.audio_preprocessing import AudioPreprocessor
from app.services.segment_extractor import SegmentExtractor
from app.services.emotion_detector import EmotionDetector
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.data_organizer import DataOrganizer
from app.services.metrics_calculator import MetricsCalculator
from app.services.report_generator import ReportGenerator

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(get_current_user)):
    # CHECK LOGIN
    if not user:
        return RedirectResponse(url="/login")
        
    # Render empty dashboard
    return templates.TemplateResponse("dashboard.html", {"request": request, "report": None, "user": user})

@router.post("/dashboard/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...), user: User = Depends(get_current_user)):
    
    # 1. CHECK LOGIN (PUT THIS FIRST)
    if not user:
        return RedirectResponse(url="/login")

    # 2. Save uploaded file (Original code starts here)
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Initialize Services
    preprocessor = AudioPreprocessor()
    whisper_service = WhisperXService(model_size="base")
    extractor = SegmentExtractor()
    emotion_detector = EmotionDetector()
    sentiment_analyzer = SentimentAnalyzer()
    report_generator = ReportGenerator()

    # 4. Run Pipeline
    # Preprocess
    processed_path = preprocessor.preprocess(file_path)
    
    # Transcribe & Diarize
    segments = whisper_service.process(processed_path)
    
    # Extract
    segments = extractor.extract(processed_path, segments)
    
    # Emotion
    segments = emotion_detector.detect(segments)
    
    # Sentiment
    segments = sentiment_analyzer.analyze(segments)
    
    # Organize
    organized_data = DataOrganizer.organize(segments)
    
    # Metrics
    metrics = MetricsCalculator.calculate(organized_data)
    
    # LLM Report
    llm_report = report_generator.generate(organized_data, metrics)

    # 5. Prepare Data for Template
    # Combine agent/customer segments into one list and sort by time
    all_segments = organized_data["agent"]["segments"] + organized_data["customer"]["segments"]
    all_segments.sort(key=lambda x: x["start"])

    report_data = {
        "metrics": metrics,
        "transcript": all_segments,
        "llm_report": llm_report
    }

    # 6. Render Template with Data (Pass user variable to template)
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "report": report_data,
        "user": user # <--- Pass user so we can show "Logged in as..."
    })