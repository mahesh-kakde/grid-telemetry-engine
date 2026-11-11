#!/usr/bin/env bash
set -e

# determine external public port (Render injects $PORT, defaults to 8080)
PORT="${PORT:-8080}"

# ensure directory structures exist
mkdir -p data models/forecast reports/drift /run/nginx

# pre-populate initial telemetry and model baselines if local database is absent
if [ ! -f "data/local_telemetry.db" ]; then
    echo "First boot: initializing database and seeding telemetry baseline..."
    python -m src.ingestion.runner --region ALL --days 14 || true
    python -m src.models.train_forecast --region ALL || true
    python -m src.models.anomaly_runner --region ALL || true
    python -m src.monitoring.drift_runner --region ALL || true
fi

# 1. start fastapi backend on internal port 8000 in background
echo "Starting FastAPI telemetry backend on internal port 8000..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

# 2. start streamlit dashboard on internal port 7860 in background
echo "Starting Streamlit UI dashboard on internal port 7860..."
streamlit run src/ui/app.py \
    --server.port 7860 \
    --server.address 127.0.0.1 \
    --server.headless true &

# 3. generate nginx routing config dynamically with $PORT
echo "Configuring Nginx reverse proxy on external port ${PORT}..."
cat <<EOF > /etc/nginx/conf.d/default.conf
server {
    listen ${PORT};
    server_name _;

    client_max_body_size 50M;

    # FastAPI OpenAPI documentation, health check, and REST endpoints
    location ~ ^/(docs|redoc|openapi\.json|health|api) {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Streamlit UI dashboard and WebSockets
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
EOF

# remove default nginx welcome site if present
rm -f /etc/nginx/sites-enabled/default

# 4. start nginx in foreground (binds to $PORT)
echo "Starting Nginx on port ${PORT}..."
exec nginx -g "daemon off;"