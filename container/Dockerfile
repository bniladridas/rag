# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Avoid prompts and use faster mirrors
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install system dependencies (build tools, clean up cache)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .

# Upgrade pip & install dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source & setup
COPY src ./src
COPY setup.py .

# Editable install for development
RUN pip install -e .

# Create non-root user (avoid UID issues)
RUN adduser --disabled-password --gecos '' app && \
    chown -R app:app /app
USER app

# Default command
CMD ["rag", "--help"]
