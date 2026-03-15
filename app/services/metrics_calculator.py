from typing import Dict, List
from collections import Counter

class MetricsCalculator:

    @staticmethod
    def calculate(organized_data: Dict) -> Dict:
        """
        Calculates aggregate metrics for the call.
        """
        agent = organized_data["agent"]
        customer = organized_data["customer"]

        # 1. Calculate Talk Ratio
        total_time = agent["total_talk_time"] + customer["total_talk_time"]
        agent_talk_ratio = 0.0
        if total_time > 0:
            agent_talk_ratio = round((agent["total_talk_time"] / total_time) * 100, 2)

        # 2. Calculate Average Sentiment
        def get_avg_sentiment(scores: List[float]) -> str:
            if not scores:
                return "Neutral"
            avg = sum(scores) / len(scores)
            if avg >= 0.05: 
                return "Positive"
            if avg <= -0.05: 
                return "Negative"
            return "Neutral"

        agent_avg_sentiment = get_avg_sentiment(agent["sentiment_scores"])
        customer_avg_sentiment = get_avg_sentiment(customer["sentiment_scores"])

        # 3. Calculate Dominant Emotion
        def get_dominant(emotions: List[str]) -> str:
            if not emotions:
                return "Neutral"
            # Count frequency
            counts = Counter(emotions)
            return counts.most_common(1)[0][0]

        agent_dominant_emotion = get_dominant(agent["emotions"])
        customer_dominant_emotion = get_dominant(customer["emotions"])

        # 4. Interaction Score (Simple heuristic)
        # Good interaction = Positive Sentiment + Agent talks slightly more
        interaction_score = 50 # Base
        if agent_avg_sentiment == "Positive":
            interaction_score += 10
        if customer_avg_sentiment == "Positive":
            interaction_score += 20
        if agent_talk_ratio > 60:
            interaction_score -= 5 # Agent talking too much
        
        return {
            "talk_ratio": {
                "agent": f"{agent_talk_ratio}%",
                "customer": f"{100 - agent_talk_ratio}%"
            },
            "average_sentiment": {
                "agent": agent_avg_sentiment,
                "customer": customer_avg_sentiment
            },
            "dominant_emotion": {
                "agent": agent_dominant_emotion,
                "customer": customer_dominant_emotion
            },
            "interaction_score": interaction_score, # Out of 100
            "total_duration_seconds": round(total_time, 2)
        }