#!/usr/bin/env bash
set -e

# default to 7860 for local/hf; render provides dynamic $PORT (e.g. 10000)
PORT="${PORT:-7860}"

echo "booting unified streamlit asgi server on port ${PORT}..."
exec uvicorn src.server:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips='*'