import librosa
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_audio(self, audio_path):
        """Load audio file and resample to standard rate"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                print(f"✓ Resampled from {sr} to {self.sample_rate} Hz")
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            print(f"✓ Audio loaded: {len(audio)} samples, duration: {len(audio) / self.sample_rate:.2f}s")
            
            return audio
        
        except Exception as e:
            print(f"✗ Error loading audio: {e}")
            return None
    
    def extract_features(self, audio):
        """Extract mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=80,
            fmin=0,
            fmax=8000,
            n_fft=2048,
            hop_length=512
        )
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = np.expand_dims(mel_spec, axis=0)  # Add channel dimension
        
        print(f"✓ Mel-spectrogram shape: {mel_spec.shape}")
        return mel_spec
    
    def save_audio(self, audio, output_path):
        """Save audio to file"""
        try:
            sf.write(output_path, audio, self.sample_rate)
            print(f"✓ Audio saved to {output_path}")
        except Exception as e:
            print(f"✗ Error saving audio: {e}")
    
    def get_duration(self, audio):
        """Get audio duration in seconds"""
        return len(audio) / self.sample_rate
