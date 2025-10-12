FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY src ./src
COPY setup.py .
COPY README.md .

RUN pip install -e .

RUN adduser --disabled-password --gecos '' app && \
    chown -R app:app /app
USER app

# Default to a harmless help command
CMD ["python", "-m", "rag", "--help"]
