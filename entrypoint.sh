#!/usr/bin/env bash
set -e

# determine external public port (Render injects $PORT, defaults to 8080)
PORT="${PORT:-8080}"

# ensure directory structures exist
mkdir -p data models/forecast reports/drift /run/nginx

# 1. start background worker for first-boot baseline seeding (non-blocking)
(
    if [ ! -f "data/local_telemetry.db" ]; then
        echo "Background task: initializing database and seeding telemetry baseline..."
        python -m src.ingestion.runner --region ALL --days 14 || true
        python -m src.models.train_forecast --region ALL || true
        python -m src.models.anomaly_runner --region ALL || true
        python -m src.monitoring.drift_runner --region ALL || true
        echo "Background task: baseline seeding finished."
    fi
) &

# 2. start fastapi backend on internal port 8000 in background
echo "Starting FastAPI telemetry backend on internal port 8000..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

# 3. start streamlit dashboard on internal port 7860 in background
echo "Starting Streamlit UI dashboard on internal port 7860..."
streamlit run src/ui/app.py \
    --server.port 7860 \
    --server.address 127.0.0.1 \
    --server.headless true &

# 4. generate pristine self-contained nginx.conf
echo "Configuring Nginx reverse proxy on external port ${PORT}..."
cat <<EOF > /etc/nginx/nginx.conf
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;

    # generous buffer sizes for Cloudflare headers & cookies (prevents 400 Bad Request)
    client_header_buffer_size 16k;
    large_client_header_buffers 8 64k;
    client_max_body_size 50M;

    map \$http_upgrade \$connection_upgrade {
        default upgrade;
        '' close;
    }

    server {
        listen ${PORT};
        server_name _;

        # FastAPI OpenAPI documentation, health check, and REST endpoints
        location ~ ^/(docs|redoc|openapi\.json|health|api) {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_buffer_size 128k;
            proxy_buffers 4 256k;
            proxy_busy_buffers_size 256k;
        }

        # Streamlit UI dashboard and WebSockets
        location / {
            proxy_pass http://127.0.0.1:7860;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection \$connection_upgrade;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_read_timeout 86400s;
            proxy_send_timeout 86400s;
            proxy_buffer_size 128k;
            proxy_buffers 4 256k;
            proxy_busy_buffers_size 256k;
        }
    }
}
EOF

# 5. verify nginx syntax and start in foreground to bind $PORT immediately
echo "Testing Nginx configuration..."
nginx -t

echo "Starting Nginx on port ${PORT}..."
exec nginx -g "daemon off;"