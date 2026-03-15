import os
from groq import Groq
from typing import Dict

class ReportGenerator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # FIX: Updated to the current supported model
        self.model = "llama-3.1-8b-instant" 

    def generate(self, organized_data: Dict, metrics: Dict) -> str:
        print("Generating LLM Report...")

        # 1. Prepare the Transcript
        transcript_text = ""
        
        # Combine agent and customer segments for context
        all_segments = organized_data["agent"]["segments"] + organized_data["customer"]["segments"]
        # Sort by start time
        all_segments.sort(key=lambda x: x["start"])

        for seg in all_segments:
            role = "Agent" if seg["speaker"] == organized_data["agent"]["speaker_id"] else "Customer"
            transcript_text += f"{role}: {seg['text']}\n"

        # 2. Create the Prompt
        prompt = f"""
        You are an expert Call Center Quality Assurance Manager.
        Analyze the following call data and generate a concise performance report.

        ### CALL METRICS ###
        - Talk Ratio: Agent {metrics['talk_ratio']['agent']}, Customer {metrics['talk_ratio']['customer']}
        - Average Sentiment: Agent ({metrics['average_sentiment']['agent']}), Customer ({metrics['average_sentiment']['customer']})
        - Dominant Emotion: Agent ({metrics['dominant_emotion']['agent']}), Customer ({metrics['dominant_emotion']['customer']})
        - Interaction Score: {metrics['interaction_score']}/100

        ### TRANSCRIPT ###
        {transcript_text}

        ### INSTRUCTIONS ###
        1. Write a brief "Call Summary" (2-3 sentences).
        2. Evaluate "Agent Performance" (Tone, Professionalism, Clarity).
        3. Evaluate "Customer Sentiment" (Were they engaged? Happy?).
        4. Provide 2-3 "Key Recommendations" for the agent to improve.
        
        Keep the response professional and formatted in Markdown.
        """

        # 3. Call Groq API
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful call center QA assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=800
            )
            
            return chat_completion.choices[0].message.content

        except Exception as e:
            return f"Error generating report: {str(e)}"