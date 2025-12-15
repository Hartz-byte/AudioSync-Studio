import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
# Patch torch.load to default weights_only=False for legacy models (Bark, Wav2Lip)
_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
         kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

from utils import check_gpu_availability, get_gpu_memory_info, create_directories, verify_model_files
from audio_processor import AudioProcessor
from voice_synthesis import VoiceSynthesizer
from wav2lip_processor import Wav2LipProcessor

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def run_setup_check(base_path):
    """Run complete setup verification"""
    print_header("🎬 AudioSync Studio - Complete Setup Check")
    
    # 1. Check GPU
    print("1️⃣  GPU Configuration")
    print("-" * 40)
    gpu_available = check_gpu_availability()
    if gpu_available:
        mem_info = get_gpu_memory_info()
        print(f"   Allocated: {mem_info['allocated']:.2f} GB")
        print(f"   Reserved: {mem_info['reserved']:.2f} GB")
        print(f"   Free: {mem_info['free']:.2f} GB")
    
    # 2. Create directories
    print("\n2️⃣  Directory Structure")
    print("-" * 40)
    create_directories(base_path)
    
    # 3. Verify model files
    print("\n3️⃣  Model Files")
    print("-" * 40)
    models_ready = verify_model_files(base_path)
    
    if not models_ready:
        print("\n⚠️  Some models are missing!")
        print("   Please download:")
        print("   - Wav2Lip: models/Wav2Lip/checkpoints/wav2lip.pth")
        print("   - Face Detection: models/Wav2Lip/face_detection/detection/sfd/s3fd.pth")
    
    # 4. Test Audio Processing
    print("\n4️⃣  Audio Processing")
    print("-" * 40)
    try:
        audio_processor = AudioProcessor()
        print("✓ AudioProcessor initialized")
        
        # Create sample audio for testing
        import numpy as np
        sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        duration = audio_processor.get_duration(sample_audio)
        print(f"✓ Test audio: {duration:.2f} seconds")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 5. Test Voice Synthesis
    print("\n5️⃣  Voice Synthesis")
    print("-" * 40)
    try:
        synthesizer = VoiceSynthesizer()
        print("✓ VoiceSynthesizer initialized")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 6. Test Wav2Lip Processor
    print("\n6️⃣  Wav2Lip Processor")
    print("-" * 40)
    try:
        checkpoint = str(base_path / 'models/Wav2Lip/checkpoints/wav2lip.pth')
        if os.path.exists(checkpoint):
            processor = Wav2LipProcessor(checkpoint)
            print("✓ Wav2LipProcessor initialized")
        else:
            print("⚠️  Model checkpoint not found")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 7. Final Summary
    print_header("✅ Setup Complete!")
    
    print("Next Steps:")
    print("-" * 40)
    print("1. Download pre-trained models if not already done")
    print("2. Prepare sample audio and video files in data/ folders")
    print("3. Run: python process_lip_sync.py")
    print("4. Or start API server: python -m src.api_server")
    print("\nFor support:")
    print("- Check logs/ directory for error details")
    print("- Verify GPU memory with: nvidia-smi")

def process_video_command(base_path):
    """Handle video processing command"""
    print_header("🎬 AudioSync Studio - Video Processing")
    
    # Define paths
    video_path = base_path / 'data' / 'input_video' / 'sample_face.mp4'
    audio_path = base_path / 'data' / 'sample_data' / 'demo_speech.wav'
    output_path = base_path / 'data' / 'output_video' / 'lip_synced_output.mp4'
    checkpoint_path = base_path / 'models' / 'Wav2Lip' / 'checkpoints' / 'wav2lip.pth'
    
    # Validate input files
    print("Checking input files...")
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        print(f"  Place your video at: data/input_video/sample_face.mp4")
        return False
    else:
        print(f"✓ Video found: {video_path}")
    
    if not audio_path.exists():
        print(f"✗ Audio not found: {audio_path}")
        print(f"  Place your audio at: data/input_audio/demo_speech.wav")
        return False
    else:
        print(f"✓ Audio found: {audio_path}")
    
    if not checkpoint_path.exists():
        print(f"✗ Model checkpoint not found: {checkpoint_path}")
        return False
    else:
        print(f"✓ Model checkpoint found: {checkpoint_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process video
    print("\n" + "="*60)
    processor = Wav2LipProcessor(str(checkpoint_path))
    success = processor.process_video(str(video_path), str(audio_path), str(output_path))
    
    if success:
        print("\n✅ Video processing completed!")
        print(f"   Output: {output_path}")
    else:
        print("\n✗ Video processing failed")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="AudioSync Studio - AI Video Lip-Sync Generation")
    parser.add_argument('--process-video', action='store_true', help='Process video with lip-sync')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--audio', type=str, help='Path to input audio')
    parser.add_argument('--output', type=str, help='Path to output video')
    
    args = parser.parse_args()
    base_path = Path(__file__).parent
    
    if args.process_video:
        # Use custom paths if provided, otherwise use defaults
        if args.video and args.audio and args.output:
            print_header("🎬 AudioSync Studio - Custom Video Processing")
            checkpoint_path = base_path / 'models' / 'Wav2Lip' / 'checkpoints' / 'wav2lip.pth'
            processor = Wav2LipProcessor(str(checkpoint_path))
            success = processor.process_video(args.video, args.audio, args.output)
            if success:
                print(f"✅ Output saved: {args.output}")
        else:
            process_video_command(base_path)
    else:
        run_setup_check(base_path)

if __name__ == "__main__":
    main()
