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
        if torch.cuda.is_available():
            det_device = 'cuda'
            # Disable cudnn benchmark to prevent hangs on some Windows setups
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            det_device = 'cpu'
            
        print(f"  Initializing Face Detector ({det_device})...")
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                              flip_input=False, device=det_device)

        print(f"  Detecting faces in {len(images)} frames...")
        # Use batch size 1 for maximum safety on Windows/CUDA
        batch_size = 1
        
        while 1:
            predictions = []
            try:
                # Iterate in batches
                for i in tqdm(range(0, len(images), batch_size), desc="Face Detection"):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError as e:
                # Handle OOM
                if 'cuda' in det_device and batch_size == 1: 
                    print("Warn: CUDA failed even at batch_size=1. Switching to CPU.")
                    det_device = 'cpu'
                    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                          flip_input=False, device='cpu')
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

    def _datagen(self, frames, mels, faces_coords, ref_face_unused):
        """
        Generator that yields batches for inference
        """
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        # Pre-calculate all faces to avoid random access overhead/complexity
        # (Optimization: could be lazy, but list is okay for short videos)
        
        # Loop through mels (audio chunks)
        # Wav2Lip logic: "idx = i % len(frames)" allows looping video if audio is longer
        for i, m in enumerate(mels):
            idx = i % len(frames)
            
            frame_to_save = frames[idx].copy()
            coords = faces_coords[idx] # (x1, y1, x2, y2)
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            
            # Extract face
            face = frame_to_save[y1:y2, x1:x2]
            
            # Resize with high quality
            face = cv2.resize(face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            
            # Silence Handling: If mel chunk is quiet, force it to minimum value (closed mouth)
            # Wav2Lip normalized range is [-4, 4]. Silence is -4.
            # Relaxed threshold to -3.8 to avoid            # Silence Handling
            mel_mean = np.mean(m)
            mel_max = np.max(m)
            
            # Logstats every 20 frames
            if i % 20 == 0:
                print(f"  Debug: Frame {i} Mel - Mean: {mel_mean:.2f}, Max: {mel_max:.2f}")

            # Disable manual silence clamping - let the model decide
            # if mel_mean < -3.8: m[:] = -10.0
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append((x1, y1, x2, y2))

            if len(img_batch) >= self.batch_size:
                img_batch = np.asarray(img_batch)
                mel_batch = np.asarray(mel_batch)

                img_masked = img_batch.copy()
                # Explicitly zero bottom half using 4-dim slice just in case
                img_masked[:, self.img_size//2:, :, :] = 0
                
                # Static Reference (Frame 0) to prevent flickering and ghosting
                # Use ref_face (passed to function, or generate it here using frames[0])
                # We need to ensure ref_face is available. 
                # Let's simple use frames[0] again, assuming faces_coords matches.
                
                ref_batch = []
                # Use Frame 0 for consistency
                ref_frame = frames[0] 
                rc = faces_coords[0]
                rx1, ry1, rx2, ry2 = int(rc[0]), int(rc[1]), int(rc[2]), int(rc[3])
                r_face = ref_frame[ry1:ry2, rx1:rx2]
                r_face = cv2.resize(r_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
                
                ref_batch = np.tile(r_face, (len(img_batch), 1, 1, 1))

                img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        # Yield remaining
        if len(img_batch) > 0:
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:, :, :] = 0
            
            # Static Reference (Frame 0)
            ref_frame = frames[0] 
            rc = faces_coords[0]
            rx1, ry1, rx2, ry2 = int(rc[0]), int(rc[1]), int(rc[2]), int(rc[3])
            r_face = ref_frame[ry1:ry2, rx1:rx2]
            r_face = cv2.resize(r_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            
            ref_batch = np.tile(r_face, (len(img_batch), 1, 1, 1))

            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def _linear_color_transfer(self, target_img, source_img):
        """
        Match the color distribution of the target image to the source image
        using Reinhard's method (Mean/Std transfer).
        target_img: The generated mouth (to be corrected), BGR, uint8
        source_img: The original face patch (reference), BGR, uint8
        """
        # Convert to LAB color space
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype("float32")
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype("float32")
        
        # Calculate stats
        t_mean, t_std = np.mean(target_lab, axis=(0,1)), np.std(target_lab, axis=(0,1))
        s_mean, s_std = np.mean(source_lab, axis=(0,1)), np.std(source_lab, axis=(0,1))
        
        # Avoid division by zero
        t_std = np.clip(t_std, 1e-5, None)
        s_std = np.clip(s_std, 1e-5, None)
        
        # Transfer
        # (Target - Mean) * (Std_Source / Std_Target) + Mean_Source
        target_lab = (target_lab - t_mean) * (s_std / t_std) + s_mean
        
        # Clip and convert back
        target_lab = np.clip(target_lab, 0, 255).astype("uint8")
        return cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)

    def process_video(self, video_path, audio_path, output_path):
        print(f"\n🎬 AudioSync Studio - Corrected Processing Pipeline")
        print(f"   Video: {video_path}")
        print(f"   Audio: {audio_path}")
        
        # 1. Load Video & Convert to 25FPS (Standard for Wav2Lip)
        if not os.path.isfile(video_path):
             print(f"Error: Video file not found: {video_path}")
             return False

        # Debug GPU
        print(f"  Debug: torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Debug: torch.version.cuda = {torch.version.cuda}")
            print(f"  Debug: Current Device = {torch.cuda.current_device()}")
            print(f"  Debug: Device Name = {torch.cuda.get_device_name(0)}")
        else:
            print("  Warn: Running on CPU. This will be slow.")

        print("  Converting video to 25fps for accurate sync...")
        temp_25fps = str(output_path).replace('.mp4', '_25fps_input.mp4')
        print(f"  Converting video to 25fps (and downscaling to 720p for performance)...")
        # Optimization: Downscale huge videos (e.g. 4K) to 720p to prevent GPU OOM/Freeze
        # scale=-2:720 ensures width is even (divisible by 2) and height is 720.
        cmd_fps = [
            'ffmpeg', '-y', 
            '-i', str(video_path), 
            '-vf', 'scale=-2:720,fps=25', 
            temp_25fps
        ]
        subprocess.run(cmd_fps, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Use the 25fps video for processing
        video_read_path = temp_25fps
             
        video_stream = cv2.VideoCapture(video_read_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print(f"  Reading video frames from {video_read_path}...")
        
        full_frames = []
        while 1:
            ret, frame = video_stream.read()
            if not ret: break
            full_frames.append(frame)
        video_stream.release()
        
        # Cleanup temp file
        if os.path.exists(temp_25fps):
            os.remove(temp_25fps)
        
        if not full_frames:
            print("Error: No frames read from video")
            return False
        
        print(f"  ✓ Read {len(full_frames)} frames @ {fps} FPS")

        # 2. Extract Audio Mel
        print("  Processing audio...")
        try:
            wav = audio.load_wav(str(audio_path), 16000)
            
            # Normalize audio volume to ensure clear lipsync
            # Weak audio leads to weak/closed mouth movements
            if len(wav) > 0:
                wav_max = np.max(np.abs(wav))
                if wav_max < 0.9:
                    print(f"  Audio too quiet (Peak: {wav_max:.2f}). Boosting to 0.95...")
                    wav = wav / (wav_max + 1e-8) * 0.95
            
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
        print(f"  ✓ Face coordinates ready for {len(faces_coords)} frames")
        
        # Prepare Static Reference (Frame 0) to force generation
        # This prevents the model from "leaking" the original mouth movement from the reference.
        # ref_img = full_frames[0].copy()
        # ref_coords = faces_coords[0]
        # rx1, ry1, rx2, ry2 = int(ref_coords[0]), int(ref_coords[1]), int(ref_coords[2]), int(ref_coords[3])
        # ref_face = ref_img[ry1:ry2, rx1:rx2]
        # ref_face = cv2.resize(ref_face, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        
        # 5. Output Writer setup
        frame_h, frame_w = full_frames[0].shape[:-1]
        temp_out = str(output_path).replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        # 6. Inference Loop
        gen = self._datagen(full_frames, mel_chunks, faces_coords, None)
        
        print("  Starting inference...")
        total_batches = int(np.ceil(float(len(mel_chunks))/self.batch_size))
        
        # Updated Mask: Soft Mouth Area Only
        mask_template = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Define ROI (Region of Interest) for the mouth:
        # Tighten margins to avoid over-smoothing large areas
        margin = 6 
        top_start = self.img_size // 2 # 48
         
        mask_template[top_start: -margin, margin : -margin] = 1.0
        
        # Reduce blur radius to keep texture definition while hiding edges
        mask_template = cv2.GaussianBlur(mask_template, (11, 11), 0)
        mask_template = np.clip(mask_template, 0, 1) 
        # Keep mask_template 2D
        
        # Sharpening Kernel (Mild)
        sharpen_kernel = np.array([[0, -1, 0], 
                                   [-1, 5, -1], 
                                   [0, -1, 0]])

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=total_batches)):
            
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for j, (p, f, c) in enumerate(zip(pred, frames, coords)):
                x1, y1, x2, y2 = c
                try:
                    p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                    
                    # Texture Synthesis (Fake Details) to fix "Smooth Mask"
                    # 1. Add Gaussian Noise (Skin Grain) - Monochromatic (Luma only) to avoid colored speckles
                    noise = np.random.randn(p_resized.shape[0], p_resized.shape[1], 1) * 4.0 
                    p_noisy = p_resized.astype(np.float32) + noise
                    p_noisy = np.clip(p_noisy, 0, 255).astype(np.uint8)
                    
                    # 2. Strong Sharpening
                    sharpen_kernel_strong = np.array([[-1,-1,-1], 
                                                      [-1, 9,-1], 
                                                      [-1,-1,-1]])
                    p_textured = cv2.filter2D(p_noisy, -1, sharpen_kernel_strong)
                    
                    # Resize masking template (2D) to match face rect
                    mask_resized = cv2.resize(mask_template, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                    
                    # Add channel dimension
                    mask_resized = mask_resized[..., np.newaxis] 

                    # Original face region
                    original_face_region = f[y1:y2, x1:x2].astype(np.uint8)
                    
                    # Color Transfer: Match prediction to original face skin tone
                    p_corrected = self._linear_color_transfer(p_textured, original_face_region)
                    
                    # Saturation Boost (To fix "Gray" look)
                    # Convert to HSV, boost S channel
                    hsv = cv2.cvtColor(p_corrected, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[..., 1] = hsv[..., 1] * 1.3 # Boost saturation by 30%
                    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
                    p_corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                    
                    p_corrected_float = p_corrected.astype(np.float32)
                    
                    p_corrected_float = p_corrected.astype(np.float32)
                    original_float = original_face_region.astype(np.float32)
                    
                    # Soft Blending: alpha * pred + (1-alpha) * original
                    blended = mask_resized * p_corrected_float + (1.0 - mask_resized) * original_float
                    
                    f[y1:y2, x1:x2] = blended.astype(np.uint8)
                    
                    # Debug Green Box Removed for final output
                    # cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
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
            '-c:v', 'libx264',      # Force H.264 for browser compatibility
            '-pix_fmt', 'yuv420p',  # Ensure YUV420P for broad compatibility
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',    # Ensure video is cut to audio length
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
            if e.stderr:
                print(f"Using FFmpeg stderr:\n{e.stderr.decode()}")
            return False
