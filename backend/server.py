
import os
import sys
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import uvicorn
import asyncio
import uuid

# Add root directory to sys.path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.voice_synthesis import VoiceSynthesizer
from src.wav2lip_processor import Wav2LipProcessor

app = FastAPI(title="AudioSync Studio API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
voice_synth = None
wav2lip_proc = None

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "output_video"
AUDIO_DIR = DATA_DIR / "sample_data"
CHECKPOINT_PATH = BASE_DIR / "models" / "Wav2Lip" / "checkpoints" / "wav2lip.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    voice: str = "pyttsx3"
    gender: str = None # "male" or "female"

@app.on_event("startup")
async def startup_event():
    global voice_synth, wav2lip_proc
    print("🚀 Starting AudioSync Studio Backend...")
    
    # Initialize Synthesizer (Force pyttsx3 for reliability as per recent fixes)
    print("🔹 Initializing Voice Synthesizer...")
    voice_synth = VoiceSynthesizer(engine="pyttsx3")
    
    # Initialize Wav2Lip
    print("🔹 Initializing Wav2Lip Processor...")
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Error: Model not found at {CHECKPOINT_PATH}")
        sys.exit(1)
        
    wav2lip_proc = Wav2LipProcessor(checkpoint_path=str(CHECKPOINT_PATH))
    print("✅ System Ready!")

@app.post("/api/tts")
async def generate_speech(req: TTSRequest):
    """Generate audio from text"""
    try:
        filename = f"speech_{uuid.uuid4().hex[:8]}.wav"
        output_path = AUDIO_DIR / filename
        
        # Use simple pyttsx3 generation
        # We need to handle the fact that pyttsx3 might block the loop?
        # Running in threadpool is safer for sync operations
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: voice_synth.generate_speech(req.text, str(output_path), gender=req.gender))
        
        if output_path.exists():
            return {
                "status": "success", 
                "audio_url": f"/files/sample_data/{filename}",
                "filename": filename,
                "path": str(output_path)
            }
        else:
            return JSONResponse(status_code=500, content={"error": "Audio generation failed"})
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload input face video"""
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "status": "success", 
            "filename": file.filename,
            "path": str(file_path),
            "url": f"/files/uploads/{file.filename}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/process")
async def process_video(
    video_filename: str = Form(...),
    audio_filename: str = Form(...)
):
    """Process Video + Audio"""
    try:
        video_path = UPLOAD_DIR / video_filename
        # Check if audio is in uploads or sample_data
        if (AUDIO_DIR / audio_filename).exists():
            audio_path = AUDIO_DIR / audio_filename
        else:
            audio_path = UPLOAD_DIR / audio_filename

        output_filename = f"output_{uuid.uuid4().hex[:8]}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        print(f"🎬 Processing: {video_filename} + {audio_filename}")
        
        # Run processing (blocking for now, can be background task)
        # For better UX, we should return a task ID and process in background.
        # But for MVP, simple blocking call works (with timeout risk).
        # Let's run in threadpool.
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, 
            lambda: wav2lip_proc.process_video(str(video_path), str(audio_path), str(output_path))
        )
        
        if success is False: 
             return JSONResponse(status_code=500, content={"error": "Processing failed inside Wav2Lip"})

        return {
            "status": "success",
            "output_url": f"/files/output_video/{output_filename}",
            "filename": output_filename
        }

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Static File Serving
from fastapi.staticfiles import StaticFiles
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
