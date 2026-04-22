# ─────────────────────────────────────────────────────────────
# AIsha Pipeline — Dockerfile
# Build:   docker build -t aisha-pipeline .
# Run:     docker run -p 7861:7861 -v $(pwd)/outputs:/app/outputs aisha-pipeline
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies (ffmpeg for audio, libsndfile for soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p outputs downloads segments uploads_tmp templates

# Port
EXPOSE 7861

# Run
CMD ["python", "app.py"]
