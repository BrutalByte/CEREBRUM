# CEREBRUM KG Reasoning API - Dockerfile
# v2.20.1 Release

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY pyproject.toml .
RUN pip install --no-cache-dir .[api,embeddings]

# Copy source code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8200

# Expose API port
EXPOSE 8200

# Start FastAPI server
# Usage: docker run -p 8200:8200 parallax
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8200"]
