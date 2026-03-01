FROM python:3.11-slim

ARG VARIANT=light

# Base system dependencies (always needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# OpenCV + Tesseract OCR — only for "full" variant
RUN if [ "$VARIANT" = "full" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libgl1-mesa-glx \
            libglib2.0-0 \
            tesseract-ocr \
            tesseract-ocr-eng \
            tesseract-ocr-rus \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Install base Python dependencies
COPY requirements.txt requirements-full.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install full-variant extras (pytesseract, imutils)
RUN if [ "$VARIANT" = "full" ]; then \
        pip install --no-cache-dir -r requirements-full.txt; \
    fi

COPY . .

RUN mkdir -p /app/tmp

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "alembic upgrade head && python main.py"]