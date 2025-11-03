# Docker Configuration

This directory contains the Docker configuration for the RAG project.

## Files

- `Dockerfile`: Multi-stage build for the RAG application
- `.dockerignore`: Files to exclude from Docker build context

## Building

```bash
docker build -f container/Dockerfile -t rag .
```

## Features

- Python 3.10 slim base image
- Non-root user for security
- Minimal attack surface
- Cached dependency installation
