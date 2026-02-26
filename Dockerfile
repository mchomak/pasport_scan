FROM python:3.11-slim

# Install system dependencies (Tesseract OCR for MRZ reading, PostgreSQL client libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/tmp

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "alembic upgrade head && python main.py"]
