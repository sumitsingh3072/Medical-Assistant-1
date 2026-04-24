FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by OpenCV, nibabel, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



# Securely use the HF_TOKEN secret to bypass download rate limits


# Copy entire backend including model weights
COPY . .
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    python download_models.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
