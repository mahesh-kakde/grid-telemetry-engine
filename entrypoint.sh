#!/usr/bin/env bash
set -e

# start fastapi backend on port 8000 in background
echo "Starting FastAPI telemetry backend on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# start streamlit dashboard on port 7860
echo "Starting Streamlit UI dashboard on port 7860..."
streamlit run src/ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --theme.base light \
    --theme.primaryColor "#0071e3" \
    --theme.backgroundColor "#fbfbfd" \
    --theme.secondaryBackgroundColor "#f5f5f7" \
    --theme.textColor "#1d1d1f"