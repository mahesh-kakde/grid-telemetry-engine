#!/usr/bin/env bash
set -e

# default to 7860 for hugging face spaces; render provides dynamic $port (e.g. 10000)
PORT="${PORT:-7860}"

echo "initializing internal fastapi backend on loopback (127.0.0.1:8000)..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1 &

# block until internal backend responds to health checks or timeout after 30s
count=0
until curl -s http://127.0.0.1:8000/health > /dev/null; do
  echo "awaiting fastapi availability..."
  sleep 1
  count=$((count + 1))
  if [ $count -ge 30 ]; then
    echo "fastapi startup timeout reached. proceeding with direct in-memory streamlit mode..."
    break
  fi
done

echo "fastapi active. booting streamlit on external port ${PORT}..."
exec streamlit run src/ui/app.py \
  --server.port="${PORT}" \
  --server.address="0.0.0.0" \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false