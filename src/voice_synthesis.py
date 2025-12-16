"""
AudioSync Studio - Advanced Voice Synthesis
Uses multiple TTS engines for high-quality, natural-sounding speech
"""

import torch
import numpy as np
from pathlib import Path

class VoiceSynthesizer:
    def __init__(self, engine="bark"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.synthesizer_type = "glow-tts"
        self.preferred_engine = engine
        self.setup_tts()
    
    def setup_tts(self):
        """Setup best available TTS engine"""
        # Method 1: Check Preference or Try Bark
        if self.preferred_engine == "bark":
            try:
                # Method 1: Try Bark with improved settings
                from bark import SAMPLE_RATE, generate_audio, preload_models
                self.generate_audio = generate_audio
                self.SAMPLE_RATE = SAMPLE_RATE
                self.synthesizer_type = "bark"
                print("✓ Bark TTS loaded (best quality)")
                print("  Preloading models...")
                preload_models()
                print("  ✓ Models preloaded")
                return
            except ImportError as e:
                print(f"Note: Bark not available ({e})")
        
        try:
            # Method 2: Try pyttsx3 as fallback (fast, local)
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Slower = more natural
            self.engine.setProperty('volume', 0.9)
            self.synthesizer_type = "pyttsx3"
            print("✓ pyttsx3 TTS loaded (local, fast)")
            return
        except ImportError as e:
            print(f"Note: pyttsx3 not available ({e})")
            
        print("✗ No TTS engine available")
        self.generate_audio = None
        
        print("✗ No TTS engine available")
        self.generate_audio = None
    
    def generate_speech(self, text, output_path=None, voice_quality="high", gender=None):
        """
        Generate speech from text with quality options
        voice_quality: "low" (fast), "medium" (balanced), "high" (best)
        gender: "male" or "female" (for pyttsx3)
        """
        if self.synthesizer_type == "bark":
            return self._generate_bark(text, output_path, voice_quality)
        elif self.synthesizer_type == "pyttsx3":
            return self._generate_pyttsx3(text, output_path, gender)
        else:
            print("✗ No TTS engine available")
            return None
    
    def _generate_bark(self, text, output_path=None, voice_quality="high"):
        """Generate speech using Bark with quality settings"""
        try:
            from bark import generate_audio
            
            # Bark quality parameters
            quality_settings = {
                "low": {"text_temp": 0.8, "waveform_temp": 0.6},
                "medium": {"text_temp": 0.7, "waveform_temp": 0.5},
                "high": {"text_temp": 0.6, "waveform_temp": 0.4}
            }
            
            settings = quality_settings.get(voice_quality, quality_settings["high"])
            
            print(f"  Generating speech (quality: {voice_quality})...")
            print(f"  Text: '{text[:60]}...'")
            
            # Generate with quality settings
            audio_array = generate_audio(
                text,
                text_temp=settings["text_temp"],
                waveform_temp=settings["waveform_temp"]
            )
            
            audio_array = np.array(audio_array)
            print(f"✓ Generated audio shape: {audio_array.shape}")
            print(f"  Duration: {len(audio_array) / self.SAMPLE_RATE:.2f}s")
            
            # Trim Leading Silence automatically
            audio_array = self._trim_silence(audio_array)
            print(f"  ✓ Trimmed Duration: {len(audio_array) / self.SAMPLE_RATE:.2f}s")
            
            if output_path:
                import scipy.io.wavfile as wavfile
                # Normalize audio
                audio_normalized = np.clip(audio_array, -1, 1)
                audio_int16 = (audio_normalized * 32767).astype(np.int16)
                wavfile.write(output_path, self.SAMPLE_RATE, audio_int16)
                print(f"✓ Saved to {output_path}")
            
            return audio_array
        
        except Exception as e:
            print(f"✗ Bark generation error: {e}")
            return None
            
    def _trim_silence(self, audio, threshold=0.05):
        """Trim silence from beginning of audio array"""
        try:
            energy = np.abs(audio)
            # Find first index where energy > threshold
            start_idx = np.argmax(energy > threshold)
            
            # If start_idx is 0 but energy[0] is small, check if it found nothing (all False)
            if start_idx == 0 and energy[0] < threshold:
                # All silent?
                return audio
                
            return audio[start_idx:]
        except Exception:
            return audio
    
    def _generate_pyttsx3(self, text, output_path=None, gender=None):
        """Generate speech using pyttsx3"""
        try:
            # Set Voice based on gender
            if gender:
                voices = self.engine.getProperty('voices')
                target = gender.lower()
                found = False
                for v in voices:
                    name = v.name.lower()
                    if target == 'female' and ('zira' in name or 'female' in name):
                        self.engine.setProperty('voice', v.id)
                        print(f"  ✓ Switched voice to: {v.name}")
                        found = True
                        break
                    if target == 'male' and ('david' in name or 'male' in name):
                        self.engine.setProperty('voice', v.id)
                        print(f"  ✓ Switched voice to: {v.name}")
                        found = True
                        break
                if not found:
                    print(f"  ⚠️ Requested gender '{gender}' not found, using default.")

            if output_path:
                self.engine.save_to_file(text, output_path)
                self.engine.runAndWait()
                print(f"✓ Saved to {output_path}")
                
                # Load back as numpy array
                import soundfile as sf
                audio, sr = sf.read(output_path)
                print(f"✓ Generated audio: {audio.shape}")
                return audio
            else:
                print("✗ pyttsx3 requires output path")
                return None
        
        except Exception as e:
            print(f"✗ pyttsx3 error: {e}")
            return None
