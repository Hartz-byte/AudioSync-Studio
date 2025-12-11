
import numpy as np
import librosa
import os

def analyze():
    path = r"D:\AudioSync-Studio\data\sample_data\demo_speech.wav"
    if not os.path.exists(path):
        print("File not found.")
        return

    y, sr = librosa.load(path, sr=None)
    print(f"Loaded {len(y)} samples at {sr}Hz")
    print(f"Min: {np.min(y)}, Max: {np.max(y)}")
    print(f"Mean Abs: {np.mean(np.abs(y))}")
    
    if np.max(np.abs(y)) < 0.01:
        print("WARNING: Audio is effectively silent!")
    else:
        print("Audio seems to have content.")

    # Check mel levels
    import sys
    sys.path.insert(0, r"D:\AudioSync-Studio\models\Wav2Lip")
    import audio
    
    mel = audio.melspectrogram(y)
    print(f"Mel Max: {np.max(mel)}")
    print(f"Mel Min: {np.min(mel)}")
    print(f"Mel Mean: {np.mean(mel)}")
    
    # Check silence threshold
    silence_mask = mel < -3.9
    print(f"Mel Time-steps below -3.9 (Silence): {np.sum(silence_mask)} / {mel.size} ({np.sum(silence_mask)/mel.size:.2%})")

if __name__ == "__main__":
    analyze()
