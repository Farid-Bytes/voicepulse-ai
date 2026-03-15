# VoicePulse AI 🎙️

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-orange?style=flat-square)

VoicePulse AI is an intelligent Call Center Analytics platform that transforms raw audio recordings into actionable insights. It leverages state-of-the-art AI models to transcribe conversations, identify speakers, detect emotions, and generate automated performance reports.

## 🚀 Key Features

- **Audio Preprocessing**: Converts stereo/noisy audio to standardized mono 16kHz WAV format.
- **Speaker Diarization**: Identifies "Who spoke when" using Pyannote Audio.
- **Speech-to-Text**: High-accuracy transcription with WhisperX.
- **Emotion Detection**: Detects acoustic emotions (Happy, Sad, Angry, Neutral) from voice tone.
- **Sentiment Analysis**: Analyzes text sentiment (Positive, Negative) using VADER.
- **LLM Reporting**: Generates professional performance reviews and coaching tips using LLMs (Groq).
- **Interactive Dashboard**: A beautiful web interface to upload files and visualize results.

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Backend | FastAPI (Python) |
| AI/ML | WhisperX, Pyannote Audio, HuggingFace Transformers, PyTorch |
| LLM | Groq (Llama 3.1) |
| Frontend | Jinja2 Templates, Bootstrap 5 |
| Audio Processing | Librosa, Soundfile |

## 📦 Installation

### Prerequisites

- Python 3.11+
- Poetry (Dependency Management)
- FFmpeg (Required for audio processing)

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/voicepulse-ai.git
   cd voicepulse-ai
   ```

2. Install dependencies
   ```bash
   poetry install
   ```

3. Set up Environment Variables
   
   Create a `.env` file in the root directory:
   ```env
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Run the Application
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

5. Access the Dashboard
   
   Open your browser and navigate to: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## 📖 Usage

1. Navigate to the Dashboard.
2. Upload an audio file (MP3 or WAV).
3. Click "Analyze".
4. View the interaction score, metrics, transcript, and AI-generated performance report.

## ⚠️ Note on Performance

Processing time depends on hardware. On a CPU (Intel Core i5, 16GB RAM), a 2-minute audio file takes approximately 5 minutes to process fully. For production use, a GPU-enabled server is recommended to reduce this to seconds.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is MIT licensed.
