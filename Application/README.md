# Neurodiversity-Inspired Coaching Tool (Voice Input)

A Flask web application that provides personalized coaching feedback through voice input analysis using advanced NLP and audio processing techniques.

## Features

- **Voice Input Processing**: Upload audio files for transcription and analysis
- **Natural Language Processing**: Advanced text analysis using spaCy and TextBlob
- **Intent Classification**: Machine learning-based classification of actionable vs non-actionable advice
- **Personalized Feedback**: Tailored coaching recommendations based on analysis
- **Audio Enhancement**: Noise reduction and audio preprocessing capabilities

## Prerequisites

### System Requirements
- Python 3.8 or higher
- FFmpeg (required for audio processing)

### FFmpeg Installation

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH environment variable

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd compreheND/Application
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_md
```

## Usage

### 1. Start the Application
```bash
# Method 1: Using Flask CLI
flask --app app run

# Method 2: Direct Python execution
python app.py
```

### 2. Access the Application
Open your web browser and navigate to: `http://127.0.0.1:5000`

### 3. Using the Application
1. Upload an audio file (WAV, MP3, M4A, etc.)
2. Wait for processing (transcription and analysis)
3. View personalized coaching feedback

## Project Structure

```
compreheND/Application/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── uploads/              # Directory for uploaded audio files
└── venv/                 # Virtual environment (created during setup)
```

## Dependencies

### Core Dependencies
- **Flask**: Web framework
- **spaCy**: Natural language processing
- **scikit-learn**: Machine learning utilities
- **numpy**: Numerical computing
- **textblob**: Text processing and sentiment analysis

### Audio Processing
- **faster-whisper**: Speech-to-text transcription
- **librosa**: Audio analysis
- **noisereduce**: Audio noise reduction
- **soundfile**: Audio file I/O
- **pydub**: Audio manipulation

### Machine Learning
- **lightgbm**: Gradient boosting framework
- **joblib**: Model persistence

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed and added to PATH
2. **spaCy model not found**: Run `python -m spacy download en_core_web_md`
3. **Audio processing errors**: Check that audio files are in supported formats
4. **M4A file errors**: 
   - Ensure FFmpeg is properly installed
   - Try converting M4A to WAV first using: `ffmpeg -i input.m4a output.wav`
   - Use the test script: `python test_audio.py your_file.m4a`
5. **Memory issues**: The application uses CPU for Whisper model; consider using smaller models for low-end systems

### Performance Optimization

- For low-end systems, the application uses the "tiny.en" Whisper model
- Audio files are processed in chunks to manage memory usage
- Consider using GPU acceleration if available (modify WhisperModel device parameter)

## Development

### Adding New Features
1. The application is modular and well-commented
2. Core functions are separated for easy modification
3. Machine learning models can be retrained with new data

### Customization
- Modify `GOLDEN_ADVICE_LIBRARY` for different coaching domains
- Adjust `ANNOTATED_DATA` to improve intent classification
- Customize personalization logic in `apply_personalization()`

## License

This project is part of an Extended Research Project (ERP) at the University of Manchester.

## Support

For issues or questions, please refer to the project documentation or contact the development team. 