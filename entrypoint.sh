#!/usr/bin/env bash
set -e

# determine application port (Render injects $PORT, default to 7860)
APP_PORT="${PORT:-7860}"

# ensure directory structures exist
mkdir -p data models/forecast reports/drift

# pre-populate initial telemetry and model baselines if local database is absent
if [ ! -f "data/local_telemetry.db" ]; then
    echo "First boot: initializing database and seeding telemetry baseline..."
    python -m src.ingestion.runner --region ALL --days 14 || true
    python -m src.models.train_forecast --region ALL || true
    python -m src.models.anomaly_runner --region ALL || true
    python -m src.monitoring.drift_runner --region ALL || true
fi

# start fastapi backend on port 8000 in background
echo "Starting FastAPI telemetry backend on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# start streamlit dashboard
echo "Starting Streamlit UI dashboard on port ${APP_PORT}..."
streamlit run src/ui/app.py \
    --server.port "${APP_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true