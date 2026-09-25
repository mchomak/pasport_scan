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
            libgl1 \
            libglib2.0-0t64 \
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

# Fail the full image build early if the compiled OCR stack is not importable.
RUN if [ "$VARIANT" = "full" ]; then \
        ADMIN_IDS=0 DATABASE_URL=postgresql+asyncpg://build:build@localhost/build \
        python -c "import numpy, cv2, utils.rupasportread"; \
    fi

RUN mkdir -p /app/tmp

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
