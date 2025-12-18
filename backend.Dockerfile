FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
# ffmpeg is required for video processing
# build-essential is required for extensions
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Fix for some pytorch containers having weird hashlib issues
RUN pip install --upgrade pip

# Copy requirements first
COPY requirements.docker.txt requirements.txt

# Install Python dependencies (Torch is already in base image!)
# First install base requirements (excluding TTS)
RUN pip install Cython
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Manually install TTS to fix dependency hell (relax numpy constraint)
RUN git clone https://github.com/coqui-ai/TTS.git && \
    sed -i 's/numpy==1.22.0/numpy/' TTS/requirements.txt && \
    sed -i 's/numpy==1.22.0/numpy/' TTS/setup.py || true && \
    pip install --no-cache-dir --no-build-isolation ./TTS && \
    rm -rf TTS

# Copy the rest of the application
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
