FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
# ffmpeg is required for video processing
# build-essential is required for compiling TTS extensions
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg libsm6 libxext6 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set alias for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install Python dependencies
# Note: This might be slow. Consider installing torch specifically first.
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
