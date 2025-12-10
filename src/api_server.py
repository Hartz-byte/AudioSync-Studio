from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
from pathlib import Path
import asyncio
from datetime import datetime

app = FastAPI(title="AudioSync Studio API", version="1.0.0")

class TaskQueue:
    def __init__(self, max_concurrent=1):
        self.tasks = {}
        self.queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.active_tasks = 0
        self.task_counter = 0
    
    async def add_task(self, task_type, data):
        """Add task to queue"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}_{datetime.now().timestamp()}"
        
        self.tasks[task_id] = {
            'status': 'queued',
            'type': task_type,
            'data': data,
            'result': None,
            'error': None
        }
        
        await self.queue.put(task_id)
        return task_id
    
    async def get_status(self, task_id):
        """Get task status"""
        return self.tasks.get(task_id, None)

task_queue = TaskQueue()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AudioSync Studio API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/generate-video": "Generate lip-sync video",
            "/status/{task_id}": "Check task status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import torch
    
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate-video")
async def generate_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    """Generate lip-sync video"""
    try:
        # Save uploaded files
        video_path = f"temp_video_{datetime.now().timestamp()}.mp4"
        audio_path = f"temp_audio_{datetime.now().timestamp()}.wav"
        
        with open(video_path, "wb") as f:
            f.write(await video.read())
        
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        
        # Add to task queue
        task_id = await task_queue.add_task('lip_sync', {
            'video': video_path,
            'audio': audio_path
        })
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Video processing queued"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get task status"""
    status = await task_queue.get_status(task_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
