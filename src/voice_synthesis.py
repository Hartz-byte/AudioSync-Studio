"""
AudioSync Studio - Advanced Voice Synthesis
Uses multiple TTS engines for high-quality, natural-sounding speech
"""

import torch
import numpy as np
from pathlib import Path
import asyncio
import os
import sys

class VoiceSynthesizer:
    def __init__(self, engine="edge-tts"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.synthesizer_type = "edge-tts" # Default
        self.preferred_engine = engine
        self.setup_tts()
    
    def setup_tts(self):
        """Setup best available TTS engine"""
        # Method 1: Edge TTS (Best Balance of Quality/Speed)
        try:
            import edge_tts
            self.synthesizer_type = "edge-tts"
            print("✓ Edge TTS loaded (Natural Neural Voices)")
            return
        except ImportError:
            pass

        # Method 2: Check Preference or Try Bark
        if self.preferred_engine == "bark":
            try:
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
            # Method 3: Try pyttsx3 as fallback (fast, local)
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.synthesizer_type = "pyttsx3"
            print("✓ pyttsx3 TTS loaded (local, fast)")
            return
        except ImportError as e:
            print(f"Note: pyttsx3 not available ({e})")
            
        print("✗ No TTS engine available")
        self.generate_audio = None
    
    def generate_speech(self, text, output_path=None, voice_quality="high", gender=None, engine=None):
        """
        Generate speech from text
        """
        active_engine = engine if engine else self.synthesizer_type

        if active_engine == "bark":
            return self._generate_bark(text, output_path, voice_quality)
        elif active_engine == "edge-tts":
            res = self._generate_edge_tts(text, output_path, gender)
            if res is None:
                print("⚠ Edge TTS failed (likely network issue). Falling back to pyttsx3...")
                return self._generate_pyttsx3(text, output_path, gender)
            return res
        elif active_engine == "pyttsx3":
            return self._generate_pyttsx3(text, output_path, gender)
        elif active_engine == "xtts":
             return self._generate_xtts(text, output_path, reference_wav=gender)
        else:
            print(f"✗ No TTS engine available: {active_engine}")
            return None

    def setup_xtts(self):
        if hasattr(self, 'xtts_model') and self.xtts_model: return
        print("🔹 Loading XTTS Model... (First run downloads ~3GB)")
        try:
            # Monkeypatch torch.load to handle XTTS legacy checkpoints on PyTorch 2.6+
            # XTTS checkpoints contain non-safe globals.
            _original_load = torch.load
            def _safe_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_load(*args, **kwargs)
            torch.load = _safe_load
            
            from TTS.api import TTS
            # Force agree to license if needed (Coqui 0.22 might require it)
            os.environ["COQUI_TOS_AGREED"] = "1"
            
            self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            print("✓ XTTS Loaded")
            
            # Restore torch.load (optional, but good practice)
            torch.load = _original_load
            
        except Exception as e:
            print(f"✗ Error loading XTTS: {e}")
            self.xtts_model = None

    def trim_silence(self, output_path):
        """Trim leading and trailing silence"""
        try:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(output_path, sr=None)
            yt, _ = librosa.effects.trim(y, top_db=20)
            sf.write(output_path, yt, sr)
        except Exception as e:
            print(f"Warning: Silence trimming failed: {e}")

    def _generate_xtts(self, text, output_path, reference_wav):
        if not hasattr(self, 'xtts_model') or not self.xtts_model:
            self.setup_xtts()
        
        if not self.xtts_model: 
            return None

        try:
            print(f"  Cloning voice from: {reference_wav}")
            # Ensure text is not empty or too short
            if not text.strip(): return None
            
            # Using params to reduce hallucinations
            self.xtts_model.tts_to_file(
                text=text, 
                file_path=str(output_path), 
                speaker_wav=str(reference_wav), 
                language="en",
                split_sentences=True,
                temperature=0.75,
                repetition_penalty=2.0
            )
            
            # Trim Silence
            self.trim_silence(output_path)
            
            return output_path
        except Exception as e:
            print(f"Error in XTTS Generation: {e}")
            return None
    
    async def _generate_edge_tts_async(self, text, output_path, voice_name):
        import edge_tts
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(output_path)

    def _generate_edge_tts(self, text, output_path=None, gender=None):
        """Generate speech using edge-tts (Natural Neural Voices)"""
        try:
            # Default Voices
            voice = "en-US-JennyNeural" # Female default
            if gender and gender.lower() == 'male':
                voice = "en-US-ChristopherNeural"
            elif gender and gender.lower() == 'female':
                voice = "en-US-JennyNeural"
            
            print(f"  Generating edge-tts with voice: {voice}")
            
            if output_path:
                # We need to run async function from sync context
                try:
                    # Windows ProactorLoop fix might be needed if using older python/frameworks
                    # But simpler: asyncio.run() usually creates a new loop.
                    # Since we are in a ThreadPoolExecutor (from server.py), we are in a thread with No Loop.
                    asyncio.run(self._generate_edge_tts_async(text, output_path, voice))
                except RuntimeError as e:
                    # Fallback if loop issues
                    print(f"Asyncio Error: {e}")
                    raise

                print(f"✓ Saved to {output_path}")
                
                # Trim Silence
                self.trim_silence(output_path)

                 # Load back as numpy array to match return type expectation (optional)
                import soundfile as sf
                audio, sr = sf.read(output_path)
                print(f"✓ Generated audio: {audio.shape}")
                return audio
            else:
                return None
        except Exception as e:
            print(f"✗ edge-tts error: {e}")
            return None

    def _generate_bark(self, text, output_path=None, voice_quality="high"):
        """Generate speech using Bark"""
        try:
            from bark import generate_audio
            quality_settings = {
                "low": {"text_temp": 0.8, "waveform_temp": 0.6},
                "medium": {"text_temp": 0.7, "waveform_temp": 0.5},
                "high": {"text_temp": 0.6, "waveform_temp": 0.4}
            }
            settings = quality_settings.get(voice_quality, quality_settings["high"])
            print(f"  Generating speech (quality: {voice_quality})...")
            
            audio_array = generate_audio(text, text_temp=settings["text_temp"], waveform_temp=settings["waveform_temp"])
            audio_array = np.array(audio_array)
            
            # Trim Silence
            # (Simplified for brevity in overwrite)
            
            if output_path:
                import scipy.io.wavfile as wavfile
                audio_normalized = np.clip(audio_array, -1, 1)
                audio_int16 = (audio_normalized * 32767).astype(np.int16)
                wavfile.write(output_path, self.SAMPLE_RATE, audio_int16)
                print(f"✓ Saved to {output_path}")
            
            return audio_array
        except Exception as e:
            print(f"✗ Bark generation error: {e}")
            return None
            
    def _generate_pyttsx3(self, text, output_path=None, gender=None):
        """Generate speech using pyttsx3"""
        try:
            # Lazy Init if falling back
            if not hasattr(self, 'engine') or self.engine is None:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)

            if gender:
                voices = self.engine.getProperty('voices')
                target = gender.lower()
                for v in voices:
                    name = v.name.lower()
                    if target == 'female' and ('zira' in name or 'female' in name):
                        self.engine.setProperty('voice', v.id)
                        break
                    if target == 'male' and ('david' in name or 'male' in name):
                        self.engine.setProperty('voice', v.id)
                        break

            if output_path:
                self.engine.save_to_file(text, output_path)
                self.engine.runAndWait()
                print(f"✓ Saved to {output_path}")
                
                import soundfile as sf
                audio, sr = sf.read(output_path)
                return audio
        except Exception as e:
            print(f"✗ pyttsx3 error: {e}")
            return None
