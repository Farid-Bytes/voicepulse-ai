from typing import List, Dict

class DataOrganizer:
    
    @staticmethod
    def organize(segments: List[Dict]) -> Dict:
        """
        Organizes flat segments into structured Agent/Customer objects.
        Rule: The first speaker is the Agent.
        """
        if not segments:
            return {"agent": {}, "customer": {}}

        # 1. Identify Roles
        # Assume first speaker is the Agent
        first_speaker = segments[0]["speaker"]
        
        # Assign IDs
        agent_id = first_speaker
        customer_id = "SPEAKER_00" if agent_id == "SPEAKER_01" else "SPEAKER_01"
        
        # Handle UNKNOWN case - assign to nobody or ignore
        # We will filter them out for now to keep data clean

        print(f"Identified Agent: {agent_id}, Customer: {customer_id}")

        # 2. Initialize Structures
        agent_data = {
            "role": "Agent",
            "speaker_id": agent_id,
            "full_text": [],
            "segments": [],
            "total_talk_time": 0.0,
            "sentiment_scores": [],
            "emotions": []
        }

        customer_data = {
            "role": "Customer",
            "speaker_id": customer_id,
            "full_text": [],
            "segments": [],
            "total_talk_time": 0.0,
            "sentiment_scores": [],
            "emotions": []
        }

        # 3. Loop and Assign
        for seg in segments:
            speaker = seg["speaker"]
            
            # Skip Unknown speakers
            if speaker == "UNKNOWN":
                continue

            # Calculate duration
            duration = seg["end"] - seg["start"]

            # Assign to Agent
            if speaker == agent_id:
                agent_data["segments"].append(seg)
                agent_data["full_text"].append(seg["text"])
                agent_data["total_talk_time"] += duration
                if "sentiment_score" in seg:
                    agent_data["sentiment_scores"].append(seg["sentiment_score"])
                if "emotion" in seg:
                    agent_data["emotions"].append(seg["emotion"])
            
            # Assign to Customer
            elif speaker == customer_id:
                customer_data["segments"].append(seg)
                customer_data["full_text"].append(seg["text"])
                customer_data["total_talk_time"] += duration
                if "sentiment_score" in seg:
                    customer_data["sentiment_scores"].append(seg["sentiment_score"])
                if "emotion" in seg:
                    customer_data["emotions"].append(seg["emotion"])

        # 4. Finalize Text
        agent_data["full_text"] = " ".join(agent_data["full_text"])
        customer_data["full_text"] = " ".join(customer_data["full_text"])

        return {
            "agent": agent_data,
            "customer": customer_data
        }