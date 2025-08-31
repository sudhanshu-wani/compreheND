# Neurodiversity-Inspired Coaching Tool (Voice Input)

A Flask web application that provides personalized coaching feedback through voice input analysis using advanced NLP and audio processing techniques.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [FFmpeg Installation](#ffmpeg-installation)
- [Installation](#installation)
  - [1. Install Anaconda or Miniconda](#1-install-anaconda-or-miniconda)
  - [2. Create and activate a new Conda environment](#2-create-and-activate-a-new-conda-environment)
  - [3. Install pip and LightGBM via conda-forge](#3-install-pip-and-lightgbm-via-conda-forge)
  - [4. Clone the repository and navigate into it](#4-clone-the-repository-and-navigate-into-it)
  - [5. Install other Python dependencies via pip](#5-install-other-python-dependencies-via-pip)
  - [6. Download spaCy language model](#6-download-spacy-language-model)
  - [7. (Optional) Install FFmpeg via conda-forge for extended audio format support](#7-optional-install-ffmpeg-via-conda-forge-for-extended-audio-format-support)
- [Usage](#usage)
  - [1. Start the application](#1-start-the-application)
  - [2. Access the application](#2-access-the-application)
  - [3. Using the application](#3-using-the-application)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
  - [Core](#core)
  - [Audio Processing](#audio-processing)
  - [Machine Learning](#machine-learning)
- [Troubleshooting](#troubleshooting)
  - [LightGBM Import Error on macOS/Linux/Windows](#lightgbm-import-error-on-macoslinuxwindows)
  - [FFmpeg Not Found Warning](#ffmpeg-not-found-warning)
  - [spaCy Model Not Found](#spacy-model-not-found)
  - [Audio Format or Processing Errors](#audio-format-or-processing-errors)
  - [Memory Issues](#memory-issues)
- [Performance Optimization](#performance-optimization)
- [Development](#development)
  - [Adding Features](#adding-features)
  - [Customization](#customization)
- [Privacy & Data Security](#privacy--data-security)
- [License](#license)
- [Support](#support)

---

## Features

- **Voice Input Processing**: Upload audio files for transcription and analysis
- **Natural Language Processing**: Advanced text analysis using spaCy and TextBlob
- **Intent Classification**: Machine learning-based classification of actionable vs non-actionable advice
- **Personalized Feedback**: Tailored coaching recommendations based on analysis
- **Audio Enhancement**: Noise reduction and audio preprocessing capabilities
- **Privacy-First Design**: Audio files are automatically deleted immediately after processing for enhanced privacy

## Prerequisites

### System Requirements
- Python 3.8 or higher
- FFmpeg (optional but recommended for broader audio format support)

### FFmpeg Installation

**Note:** FFmpeg is optional but highly recommended. The application includes fallback audio processing that works with common formats (WAV, MP3, FLAC) without FFmpeg, but installing it enables support for additional formats like M4A, AAC, OGG, and OPUS.

**Windows:**
1. (Recommended) Install via winget:
   ```powershell
   winget install --id=Gyan.FFmpeg -e
   ```
   After installation, close and reopen the terminal so PATH updates.
2. (Manual alternative)
   - Download a static build from `https://www.gyan.dev/ffmpeg/builds/`
   - Extract to a folder (e.g., `C:\ffmpeg`)
   - Add `C:\ffmpeg\bin` to your system PATH environment variable
   - Open a new terminal

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

To minimize dependency issues (especially on macOS and Linux), the recommended setup uses Conda:

### 1. Install Anaconda or Miniconda

- Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

### 2. Create and activate a new Conda environment

```bash
conda create --name compreheND_env python=3.10 -y
conda activate compreheND_env
```

### 3. Install pip and LightGBM via conda-forge

This ensures proper handling of native dependencies like OpenMP for LightGBM.

```bash
conda install -c conda-forge pip lightgbm
```

### 4. Clone the repository and navigate into it

```bash
git clone https://github.com/sudhanshu-wani/compreheND.git
cd compreheND/Application
```

### 5. Install other Python dependencies via pip

```bash
pip install -r requirements.txt
```

> **Note:** If `lightgbm` is in `requirements.txt`, remove it to avoid conflicts with the conda-installed version.

### 6. Download spaCy language model

```bash
python -m spacy download en_core_web_md
```

### 7. (Optional) Install FFmpeg via conda-forge for extended audio format support

```bash
conda install -c conda-forge ffmpeg
```

FFmpeg is optional if only common audio formats (WAV, MP3, FLAC) are used.

---

## Usage

### 1. Start the application

```bash
# Using Flask CLI
flask --app app run

# Or direct Python execution
python app.py
```

### 2. Access the application

Open your browser at: `http://127.0.0.1:5000`

### 3. Using the application

1. Personalize feedback via available preference options
2. Upload an audio file (WAV, MP3, M4A, etc.)
3. Wait for transcription and analysis
4. View personalized coaching feedback

Trial audio samples in `compreheND/samples/` help you test.

---

## Project Structure

```
compreheND/
├── README.md                      # Project documentation (this file)
├── samples/                       # Trial audio notes for testing & replication 
└── Application/
    ├── app.py                    # Main Flask application
    ├── requirements.txt          # Python dependencies
    ├── test_audio.py             # Utility to test audio loading/conversion
    └── intent_classifier.pkl     # Trained classifier (auto-created on first run)
   
```

---

## Dependencies

### Core

- Flask: Web framework
- spaCy: Natural language processing
- scikit-learn: Machine learning utilities
- numpy: Numerical computing
- textblob: Text processing & sentiment analysis

### Audio Processing

- faster-whisper: Speech-to-text transcription
- librosa: Audio analysis
- noisereduce: Noise reduction
- soundfile: Audio I/O
- pydub: Audio manipulation

### Machine Learning

- lightgbm: Gradient boosting framework (conda-forge recommended)
- joblib: Model persistence

---

## Troubleshooting

### LightGBM Import Error on macOS/Linux/Windows

```
OSError: dlopen(...lib_lightgbm.dylib): Library not loaded: @rpath/libomp.dylib
```

Occurs because OpenMP runtime is missing.

**Fix:** Install LightGBM via conda-forge which bundles OpenMP properly:

```bash
conda install -c conda-forge lightgbm
```

If you installed LightGBM through pip earlier, uninstall it first:

```bash
pip uninstall lightgbm
```

---

### FFmpeg Not Found Warning

Pydub needs FFmpeg to support certain audio formats.

**Fix:** Install FFmpeg optionally via conda:

```bash
conda install -c conda-forge ffmpeg
```

If FFmpeg is installed but not detected, explicitly set its path in code:

```python
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")
```

---

### spaCy Model Not Found

Ensure spaCy model is installed in active environment:

```bash
python -m spacy download en_core_web_md
```

Verify installation:

```bash
python -c "import spacy; spacy.load('en_core_web_md'); print('OK')"
```

---

### Audio Format or Processing Errors

Check audio formats; you might need FFmpeg to convert unsupported ones:

```bash
ffmpeg -i input.m4a output.wav
```

---

### Memory Issues

Use smaller Whisper models for machines with limited resources.

---

## Performance Optimization

- Use "tiny.en" Whisper model for low-end environments
- Audio processing in chunks to reduce memory usage
- Enable GPU acceleration if available by adjusting model config

---

## Development

### Adding Features

- Modular, well-commented code
- Core functions separated for easy updates
- Retrain ML models with new data as needed

### Customization

- Modify `GOLDEN_ADVICE_LIBRARY` for other coaching domains
- Adjust `ANNOTATED_DATA` for intent classification improvements
- Update personalization logic in `apply_personalization()`

---

## Privacy & Data Security

- Audio files auto-deleted immediately after processing
- No permanent audio storage
- Local processing only, no external transmission of audio
- Complies with GDPR and privacy-by-design principles

---

## License

Part of an Extended Research Project (ERP) at the University of Manchester for MSc Data Science.

---

## Support

Developed & tested on Windows 10 and macOS using Conda environments.

Contact: wanisudhanshu@gmail.com

---
