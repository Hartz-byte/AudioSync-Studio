"""
AudioSync Studio - CORRECTED Wav2Lip Processor
Integrates Reference Face Detection & Inference Logic
"""

import os
import sys
import torch
import cv2
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm

# Add Wav2Lip to path
WAV2LIP_ROOT = Path(__file__).parent.parent / 'models' / 'Wav2Lip'
if str(WAV2LIP_ROOT) not in sys.path:
    # Insert at 0 to ensure we load modules from there (like audio, models, face_detection)
    sys.path.insert(0, str(WAV2LIP_ROOT))

# Import Wav2Lip modules
try:
    import face_detection
    import audio
    from models import Wav2Lip
except ImportError as e:
    print(f"Error importing Wav2Lip modules: {e}")
    print(f"sys.path: {sys.path}")
    raise

class Wav2LipProcessor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.img_size = 96
        self.batch_size = 16  # Batch size for inference
        self.face_det_batch_size = 16
        
        # Default padding from Wav2Lip inference.py [top, bottom, left, right]
        # (0, 10, 0, 0)
        self.pads = [0, 10, 0, 0] 
        
        print(f"✓ Processor initialized (device: {self.device})")
        
        self._load_model()
    
    def _load_model(self):
        print(f"Model Path: {self.checkpoint_path}")
        self.model = Wav2Lip()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        self.model.load_state_dict(new_s)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✓ Wav2Lip model loaded and ready")

    def _get_smoothened_boxes(self, boxes, T=5):
        """Smooth face bounding boxes over time"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def _face_detect(self, images):
        """
        Detect faces in a list of images (frames).
        Returns list of [x1, y1, x2, y2] coordinates.
        """
        # Try to use CUDA for face detection if available, but with small batch
        # to avoid OOM on consumer GPUs. 
        # det_device = str(self.device).split(':')[0]
        # Force CPU for face detection as CUDA is hanging on Windows
        det_device = 'cpu'
        print(f"  Initializing Face Detector ({det_device})...")
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                              flip_input=False, device=det_device)

        print(f"  Detecting faces in {len(images)} frames...")
        # Use smaller batch size for face detection to be safe
        batch_size = 4
        
        while 1:
            predictions = []
            try:
                # Iterate in batches
                for i in tqdm(range(0, len(images), batch_size), desc="Face Detection"):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError as e:
                # Handle OOM
                if batch_size == 1: 
                    print("Warn: OOM even at batch_size=1. Switching to CPU for detection.")
                    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                          flip_input=False, device='cpu')
                    batch_size = 4 
                    continue
                    
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        
        for rect, image in zip(predictions, images):
            if rect is None:
                if results:
                    results.append(results[-1]) # Reuse last known face
                else:
                    results.append([0, 0, image.shape[1], image.shape[0]])
                continue

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        # Smooth boxes to reduce jitter
        boxes = self._get_smoothened_boxes(boxes, T=5)
        
        return boxes

    def _datagen(self, frames, mels, faces_coords):
        """
        Generator that yields batches for inference
        """
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        # Loop through mels (audio chunks)
        # Wav2Lip logic: "idx = i % len(frames)" allows looping video if audio is longer
        for i, m in enumerate(mels):
            idx = i % len(frames)
            
            frame_to_save = frames[idx].copy()
            coords = faces_coords[idx] # (x1, y1, x2, y2)
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            
            # Extract face
            face = frame_to_save[y1:y2, x1:x2]
            
            # Resize
            face = cv2.resize(face, (self.img_size, self.img_size))
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append((x1, y1, x2, y2))

            if len(img_batch) >= self.batch_size:
                img_batch = np.asarray(img_batch)
                mel_batch = np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        # Yield remaining
        if len(img_batch) > 0:
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def process_video(self, video_path, audio_path, output_path):
        print(f"\n🎬 AudioSync Studio - Corrected Processing Pipeline")
        print(f"   Video: {video_path}")
        print(f"   Audio: {audio_path}")
        
        # 1. Load Video
        if not os.path.isfile(video_path):
             print(f"Error: Video file not found: {video_path}")
             return False
             
        video_stream = cv2.VideoCapture(str(video_path))
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print("  Reading video frames...")
        
        full_frames = []
        while 1:
            ret, frame = video_stream.read()
            if not ret: break
            full_frames.append(frame)
        video_stream.release()
        
        if not full_frames:
            print("Error: No frames read from video")
            return False
        
        print(f"  ✓ Read {len(full_frames)} frames @ {fps} FPS")

        # 2. Extract Audio Mel
        print("  Processing audio...")
        try:
            wav = audio.load_wav(str(audio_path), 16000)
            mel = audio.melspectrogram(wav)
        except Exception as e:
            print(f"Error processing audio: {e}")
            return False
            
        if np.isnan(mel.reshape(-1)).sum() > 0:
            print('Error: Mel contains nan! Audio file might be corrupted or silent.')
            return False
            
        print(f"  ✓ Audio mel shape: {mel.shape}")

        # 3. Create Mel Chunks
        mel_chunks = []
        mel_step_size = 16
        mel_idx_multiplier = 80./fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        
        print(f"  ✓ Generated {len(mel_chunks)} audio chunks")

        # 4. Detect Faces (only for frames we need)
        faces_coords = self._face_detect(full_frames)
        
        # 5. Output Writer setup
        frame_h, frame_w = full_frames[0].shape[:-1]
        temp_out = str(output_path).replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        # 6. Inference Loop
        gen = self._datagen(full_frames, mel_chunks, faces_coords)
        
        print("  Starting inference...")
        total_batches = int(np.ceil(float(len(mel_chunks))/self.batch_size))
        
        # Pre-compute mask for soft blending
        mask_template = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        # Create a soft circle/oval mask
        center = (self.img_size // 2, self.img_size // 2 + 8) # slightly lower for mouth
        cv2.ellipse(mask_template, center, (self.img_size // 2 - 8, self.img_size // 3), 0, 0, 360, 1, -1)
        # Blur the mask to create soft edge
        mask_template = cv2.GaussianBlur(mask_template, (21, 21), 0)
        # Reshape for broadcasting: (H, W, 1)
        mask_template = mask_template[..., np.newaxis]

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=total_batches)):
            
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames, coords):
                x1, y1, x2, y2 = c
                try:
                    # Resize prediction to match face rect
                    p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    
                    # Resize masking template to match face rect
                    mask_resized = cv2.resize(mask_template, (x2 - x1, y2 - y1))
                    
                    if len(mask_resized.shape) == 2:
                        mask_resized = mask_resized[..., np.newaxis] 

                    # Original face region
                    original_face_region = f[y1:y2, x1:x2].astype(np.float32)
                    p_resized_float = p_resized.astype(np.float32)
                    
                    # Soft Blending: alpha * pred + (1-alpha) * original
                    blended = mask_resized * p_resized_float + (1.0 - mask_resized) * original_face_region
                    
                    f[y1:y2, x1:x2] = blended.astype(np.uint8)
                    out.write(f)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    out.write(f)

        out.release()
        print("  ✓ Video generated")

        # 7. Merge Audio
        print("  Merging audio...")
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_out,
            '-i', str(audio_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Saved to: {output_path}")
            
            if os.path.exists(temp_out):
                os.remove(temp_out)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error merging audio: {e}")
            return False
