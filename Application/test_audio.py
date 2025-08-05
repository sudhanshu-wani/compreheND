#!/usr/bin/env python3
"""
Test script to debug M4A file processing issues.
Run this script with your M4A file to identify the problem.
"""

import os
import sys
import librosa
import soundfile as sf
from pydub import AudioSegment

def test_audio_file(file_path):
    """Test different methods of loading an audio file."""
    print(f"Testing audio file: {file_path}")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📁 File size: {file_size} bytes")
    
    # Test 1: Librosa
    print("\n🔍 Test 1: Loading with librosa...")
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        print(f"✅ Librosa success! Shape: {audio.shape}, Sample rate: {sr}")
        return True
    except Exception as e:
        print(f"❌ Librosa failed: {e}")
    
    # Test 2: Pydub
    print("\n🔍 Test 2: Loading with pydub...")
    try:
        audio_segment = AudioSegment.from_file(file_path)
        print(f"✅ Pydub success! Duration: {len(audio_segment)}ms, Channels: {audio_segment.channels}")
        
        # Try to export as WAV
        temp_wav = file_path + "_test.wav"
        audio_segment.export(temp_wav, format="wav")
        print(f"✅ Exported to WAV: {temp_wav}")
        
        # Try loading the WAV with librosa
        try:
            audio, sr = librosa.load(temp_wav, sr=16000, mono=True)
            print(f"✅ WAV loaded with librosa! Shape: {audio.shape}")
            os.remove(temp_wav)  # Clean up
            return True
        except Exception as e:
            print(f"❌ WAV loading failed: {e}")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        
    except Exception as e:
        print(f"❌ Pydub failed: {e}")
    
    # Test 3: Soundfile
    print("\n🔍 Test 3: Loading with soundfile...")
    try:
        audio, sr = sf.read(file_path)
        print(f"✅ Soundfile success! Shape: {audio.shape}, Sample rate: {sr}")
        return True
    except Exception as e:
        print(f"❌ Soundfile failed: {e}")
    
    print("\n❌ All audio loading methods failed!")
    return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_audio.py <path_to_audio_file>")
        print("Example: python test_audio.py test.m4a")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = test_audio_file(file_path)
    
    if success:
        print("\n✅ Audio file can be processed!")
    else:
        print("\n❌ Audio file cannot be processed. Check FFmpeg installation.")
        print("\nTroubleshooting tips:")
        print("1. Make sure FFmpeg is installed and in your PATH")
        print("2. Try converting the file to WAV format first")
        print("3. Check if the file is corrupted") 