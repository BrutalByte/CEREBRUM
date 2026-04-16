# CEREBRUM KG Reasoning API - Dockerfile
# v2.21.0

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .[api,embeddings]

COPY . .

# Persistent data lives here — mount a volume at /data in production
ENV CEREBRUM_DATA_DIR=/data
ENV PYTHONUNBUFFERED=1
ENV PORT=8200

RUN mkdir -p /data

EXPOSE 8200
# Optional WebSocket port for UE5 telemetry — expose if CEREBRUM_WS_PORT is set
EXPOSE 8765

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8200"]
