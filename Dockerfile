# Use a slim Python 3.10 base image
FROM python:3.10-slim

# Install OS-level dependencies
RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0

# Install Python packages
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn hf_transfer

# Set environment variable for Hugging Face fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Set working directory
WORKDIR /app

# Copy all local files into the image
COPY . /app

# Expose port for FastAPI
EXPOSE 7860

# Run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
