"""
AudioSync Studio - Complete Demo
Demonstrates the full pipeline: Audio → Voice Synthesis → Lip-Sync Video
"""

import os
import sys
from pathlib import Path

import torch
# Patch torch.load to default weights_only=False for legacy models (Bark, Wav2Lip)
_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
         kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from audio_processor import AudioProcessor
from voice_synthesis import VoiceSynthesizer
import numpy as np

def run_demo():
    print("\n" + "="*60)
    print("  🎬 AudioSync Studio - Demo Run")
    print("="*60 + "\n")
    
    # Initialize processors
    print("1️⃣  Initializing Audio Processor...")
    audio_processor = AudioProcessor(sample_rate=16000)
    
    print("2️⃣  Initializing Voice Synthesizer...")
    # Use pyttsx3 for cleaner audio to ensure lipsync works
    synthesizer = VoiceSynthesizer(engine="pyttsx3")
    
    # Generate speech
    print("\n3️⃣  Generating Speech...")
    demo_text = "Hello! Welcome to AudioSync Studio. I am a demo of AI-powered lip synchronization."
    
    output_audio = os.path.join('data', 'sample_data', 'demo_speech.wav')
    audio_array = synthesizer.generate_speech(demo_text, output_audio, voice_quality="high")
    
    if audio_array is not None:
        print(f"✓ Speech generated successfully!")
        print(f"   Saved to: {output_audio}")
        
        # Load and analyze
        print("\n4️⃣  Analyzing Generated Audio...")
        audio = audio_processor.load_audio(output_audio)
        features = audio_processor.extract_features(audio)
        duration = audio_processor.get_duration(audio)
        
        print(f"✓ Audio Analysis Complete")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample Rate: 16000 Hz")
        print(f"   Feature Shape: {features.shape}")
        
        print("\n✅ Demo Complete!")
        print("\nNext: Use this audio for Wav2Lip video generation")
    else:
        print("✗ Failed to generate speech")

if __name__ == "__main__":
    run_demo()
