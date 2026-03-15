from typing import List, Dict

class SpeakerTextAligner:

    @staticmethod
    def align(
        diarization_segments: List[Dict],
        transcription_segments: List[Dict]
    ) -> List[Dict]:

        aligned = []

        for t in transcription_segments:
            t_start = t["start"]
            t_end = t["end"]
            text = t["text"]
            
            best_speaker = "UNKNOWN"
            max_overlap = 0.0

            for d in diarization_segments:
                d_start = d["start"]
                d_end = d["end"]
                speaker = d["speaker"]

                # Calculate overlap duration
                # overlap_start is the LATER of the two starts
                overlap_start = max(t_start, d_start)
                # overlap_end is the EARLIER of the two ends
                overlap_end = min(t_end, d_end)

                # Calculate duration
                overlap_duration = overlap_end - overlap_start

                # If this segment has more overlap than the previous best, update it
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker

            aligned.append({
                "speaker": best_speaker,
                "start": t_start,
                "end": t_end,
                "text": text
            })

        return aligned