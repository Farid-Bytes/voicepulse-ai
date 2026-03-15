from pathlib import Path
# import json
from app.services.whisperx_service import WhisperXService
from app.services.segment_extractor import SegmentExtractor
from app.services.emotion_detector import EmotionDetector
from app.services.sentiment_analyzer import SentimentAnalyzer 
from app.services.data_organizer import DataOrganizer
from app.services.metrics_calculator import MetricsCalculator
from app.services.report_generator import ReportGenerator 

# 1. Define Path
audio_path = Path("data/processed/call_processed.wav")

# 2. Transcribe & Diarize
service = WhisperXService(model_size="base")
print("Running WhisperX...")
segments = service.process(audio_path)

# 3. Extract Segments
extractor = SegmentExtractor()
print("Extracting audio segments...")
segments_with_paths = extractor.extract(audio_path, segments)

# 4. Emotion Detection
detector = EmotionDetector()
segments_final = detector.detect(segments_with_paths)

# 5. Sentiment Analysis (NEW)
sentiment = SentimentAnalyzer()
segments_final = sentiment.analyze(segments_final)

# 6. Organize Data (Stage 8)
print("\nOrganizing data by speaker...")
organized_data = DataOrganizer.organize(segments_final)

# 7. Calculate Metrics (Stage 9)
print("\nCalculating interaction metrics...")
metrics = MetricsCalculator.calculate(organized_data)

# 8. Generate LLM Report (Stage 10)
reporter = ReportGenerator()
final_report = reporter.generate(organized_data, metrics)

# # 7. Print Result
# print("\n--- AGENT DATA ---")
# agent = organized_data["agent"]
# print(f"Speaker ID: {agent['speaker_id']}")
# print(f"Total Talk Time: {agent['total_talk_time']:.2f}s")
# print(f"Full Text Preview: {agent['full_text'][:100]}...")
# print(f"Sample Emotions: {agent['emotions'][:5]}")

# print("\n--- CUSTOMER DATA ---")
# customer = organized_data["customer"]
# print(f"Speaker ID: {customer['speaker_id']}")
# print(f"Total Talk Time: {customer['total_talk_time']:.2f}s")
# print(f"Full Text Preview: {customer['full_text'][:100]}...")
# print(f"Sample Emotions: {customer['emotions'][:5]}")

# # 8. Print Final Report
# print("\n--- FINAL INTERACTION REPORT ---")
# print(json.dumps(metrics, indent=4))

# 9. Print Final Report
print("\n" + "="*40)
print("VOICEPULSE AI - FINAL REPORT")
print("="*40)
print(final_report)