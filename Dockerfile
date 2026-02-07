FROM python:3.11-slim

# Install system dependencies (including Tesseract OCR for rupasportread)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create tmp directory
RUN mkdir -p /app/tmp

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run migrations and start bot
CMD alembic upgrade head && python -m app.main