# ======================================================
#   FunASR SenseVoiceSmall Inference Server
# ======================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
	ffmpeg libsndfile1 git && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN ls -la /app
# Copy only requirements first
COPY requirements.txt /app/

# Install dependencies (cached if requirements.txt didn't change)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your code
COPY . /app


# Optional: preload model weights during build (saves runtime download)
# RUN python -c "from funasr import AutoModel; AutoModel(model='iic/SenseVoiceSmall')"

# Expose FastAPI port
EXPOSE 50000

# Environment variables
ENV SENSEVOICE_DEVICE=auto
ENV PYTHONUNBUFFERED=1
ENV MODELSCOPE_CACHE=/models

# Create model cache directory (helps reuse between restarts)
RUN mkdir -p /models

# Start FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "50000"]
