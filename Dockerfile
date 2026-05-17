# ── Stage 1: builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim

# System deps: Tesseract + Poppler (for pdf2image) + libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      poppler-utils \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

# Download SpaCy model (baked into image)
RUN python -m spacy download en_core_web_sm

EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
