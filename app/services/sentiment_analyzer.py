from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, segments: List[Dict]) -> List[Dict]:
        print("Analyzing text sentiment...")
        
        for seg in segments:
            text = seg.get("text", "")
            
            if text:
                # Get scores
                scores = self.analyzer.polarity_scores(text)
                
                # Determine label based on 'compound' score
                compound = scores['compound']
                
                if compound >= 0.05:
                    label = "Positive"
                elif compound <= -0.05:
                    label = "Negative"
                else:
                    label = "Neutral"
                
                seg["sentiment"] = label
                seg["sentiment_score"] = round(compound, 2) # -1 to 1
            else:
                seg["sentiment"] = "N/A"
                seg["sentiment_score"] = 0.0

        return segments