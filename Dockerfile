# Telescope Detection System - Docker Image
# Multi-stage build for optimized image size

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 3: Final image
FROM dependencies AS final

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app:${PATH}"

# Copy application code
COPY src/ ./src/
COPY web/ ./web/
COPY main.py .
COPY *.md ./

# Create directories for runtime data
RUN mkdir -p /app/config \
    /app/logs \
    /app/clips \
    /app/models

# Expose web interface port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root user for security
RUN useradd -m -u 1000 -s /bin/bash telescope && \
    chown -R telescope:telescope /app
USER telescope

# Default command
CMD ["python3", "main.py"]
