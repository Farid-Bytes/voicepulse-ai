from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import torch
import soundfile as sf
import numpy as np
from typing import List, Dict

class EmotionDetector:
    def __init__(self):
        print("Loading Emotion Detection model...")
        model_name = "superb/wav2vec2-base-superb-er"
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        
        self.id2label = self.model.config.id2label
        self.label_map = {
            "neu": "Neutral", "hap": "Happy", 
            "ang": "Angry", "sad": "Sad"
        }
        
        # Optimization: Merge segments up to 10 seconds for better context & speed
        self.max_chunk_duration = 10.0 

    def detect(self, segments: List[Dict]) -> List[Dict]:
        print("Analyzing emotions (Smart Merging Mode)...")

        # 1. Group segments into batches
        batches = []
        current_batch = {"segments": [], "audio": [], "duration": 0.0, "speaker": None}

        for seg in segments:
            audio = seg.get("audio_array")
            if audio is None and seg.get("segment_audio_path"):
                try:
                    audio, _ = sf.read(seg["segment_audio_path"])
                except Exception:
                    seg["emotion"] = "Error"
                    continue

            if audio is None or len(audio) == 0:
                seg["emotion"] = "No Audio"
                continue

            duration = len(audio) / 16000 # 16kHz sample rate

            # Logic: Merge if same speaker AND total duration < max
            if (seg["speaker"] == current_batch["speaker"] and 
                current_batch["duration"] + duration < self.max_chunk_duration):
                
                # Add to existing batch
                current_batch["segments"].append(seg)
                current_batch["audio"].append(audio)
                current_batch["duration"] += duration
                
            else:
                # Save previous batch if it exists
                if current_batch["segments"]:
                    batches.append(current_batch)
                
                # Start new batch
                current_batch = {
                    "segments": [seg],
                    "audio": [audio],
                    "duration": duration,
                    "speaker": seg["speaker"]
                }

        # Add the last batch
        if current_batch["segments"]:
            batches.append(current_batch)

        # 2. Process Batches
        print(f"Reduced {len(segments)} segments to {len(batches)} analysis batches.")

        for batch in batches:
            # Concatenate audio
            merged_audio = np.concatenate(batch["audio"])

            # Prepare inputs
            inputs = self.feature_extractor(
                merged_audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )

            # Run Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Get Prediction
            predicted_id = torch.argmax(logits, dim=-1).item()
            raw_label = self.id2label[predicted_id]
            final_label = self.label_map.get(raw_label, raw_label)
            score = round(torch.softmax(logits, dim=-1)[0][predicted_id].item(), 2)

            # Assign the SAME emotion to all segments in this batch
            for seg in batch["segments"]:
                seg["emotion"] = final_label
                seg["emotion_score"] = score

        return segments