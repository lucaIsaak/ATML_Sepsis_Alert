# SepsisAlert API — Docker image
# Build: docker build -t sepsis-alert-api .
# Or use docker-compose (recommended): docker-compose up

FROM python:3.12-slim

# ffmpeg is required by openai-whisper for audio transcription
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

EXPOSE 8000

# On first run: generate synthetic demo data + train demo model if no model exists.
# On subsequent runs: model/data already present — setup_demo.py skips training.
# Persistent data is stored in Docker volumes (see docker-compose.yml).
CMD ["sh", "-c", "python setup_demo.py && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2"]
