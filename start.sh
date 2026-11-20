#!/usr/bin/env bash
set -e

# default to 7860 for hugging face spaces; render provides dynamic $PORT (e.g. 10000)
PORT="${PORT:-7860}"

# on memory-constrained containers (<= 512mb), run unified single-process streamlit mode
# enable dual-process mode when running on hugging face spaces or instances with >= 1gb ram
if [ "$ENABLE_FASTAPI" = "true" ]; then
  echo "initializing internal fastapi backend on loopback (127.0.0.1:8000)..."
  uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1 &

  count=0
  until curl -s http://127.0.0.1:8000/health > /dev/null; do
    echo "awaiting fastapi availability..."
    sleep 1
    count=$((count + 1))
    if [ $count -ge 15 ]; then
      echo "fastapi startup timeout reached. proceeding with direct in-memory streamlit mode..."
      break
    fi
  done
  echo "fastapi active. booting streamlit on external port ${PORT}..."
else
  echo "running unified single-process streamlit mode (optimized for 512mb containers)..."
fi

exec streamlit run src/ui/app.py \
  --server.port="${PORT}" \
  --server.address="0.0.0.0" \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --browser.serverPort=443 \
  --browser.gatherUsageStats=false