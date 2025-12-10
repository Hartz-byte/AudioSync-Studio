import os
import torch
import numpy as np
from pathlib import Path

def check_gpu_availability():
    """Check if GPU is available and print info"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1e9
        print(f"✓ Total VRAM: {total_mem:.2f} GB")
        return True
    else:
        print("✗ GPU not available, using CPU")
        return False

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        return {
            'total': total,
            'allocated': allocated,
            'reserved': reserved,
            'free': free
        }
    return None

def create_directories(base_path):
    """Create required directories"""
    required_dirs = [
        'models/Wav2Lip/checkpoints',
        'models/Wav2Lip/face_detection/detection/sfd',
        'models/Real-ESRGAN',
        'data/input_audio',
        'data/input_video',
        'data/output_video',
        'data/sample_data',
        'logs'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        # print(f"✓ Directory ready: {dir_path}")

def verify_model_files(base_path):
    """Verify all required model files exist"""
    required_models = {
        'Wav2Lip GAN': 'models/Wav2Lip/checkpoints/wav2lip_gan.pth',
        'Face Detection': 'models/Wav2Lip/face_detection/detection/sfd/s3fd.pth'
    }
    
    all_present = True
    for name, path in required_models.items():
        full_path = os.path.join(base_path, path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path) / 1e6
            print(f"✓ {name}: {size:.1f} MB")
        else:
            print(f"✗ {name} missing: {path}")
            all_present = False
    
    return all_present
