# grid-telemetry-engine

Live-Data Forecasting & Anomaly Engine with Drift Monitoring for the Indian National Power Grid.

Production Deployment: [grid.maheshkakde.com](https://grid.maheshkakde.com)  
API Documentation: [grid.maheshkakde.com/docs](https://grid.maheshkakde.com/docs)

---

## Overview

`grid-telemetry-engine` is an end-to-end machine learning system that ingests hourly electrical demand telemetry from India's regional power grids (POSOCO / GRID-INDIA), computes 72-hour calibrated quantile forecasts ($p_{10}, p_{50}, p_{90}$), detects structural anomalies with local SHAP feature attributions, and tracks data/concept drift via Evidently AI.

The engine covers all five operational grid regions of India:
- **Northern Region (NR)**
- **Western Region (WR)**
- **Southern Region (SR)**
- **Eastern Region (ER)**
- **North-Eastern Region (NER)**
- **National Grid Total (ALL)**

---

## Architectural Pipeline

```
[ POSOCO / GRID-INDIA API ]
             │
             ▼
[ Ingestion Engine (Cron / Backoff) ] ──> [ Supabase PostgreSQL / SQLite ]
             │
             ▼
[ Polars Feature Pipeline (< 50ms SIMD) ]
  ├── Fourier Cyclical Harmonics (24h, 168h, 365d)
  ├── Autoregressive Lags (1h, 2h, 3h, 24h, 48h, 168h)
  └── Shifted Rolling Aggregations (6h, 24h, 168h)
             │
             ├──────────────────────────┬──────────────────────────┐
             ▼                          ▼                          ▼
[ LightGBM Quantile Regressor ]   [ Isolation Forest ]   [ Evidently AI Drift Suite ]
  ├── Pinball Loss (p10, p50, p90)  ├── 2% Contamination   ├── 7d Live vs 30d Baseline
  ├── Recursive 72h Roll-Forward    └── SHAP TreeExplainer ├── KS / Wasserstein Tests
  └── Guaranteed Monotonic Bands        (Top 3 Drivers)    └── Interactive HTML Reports
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                       [ FastAPI Backend (Port 8000) ]
                                        │
                                        ▼
                     [ Streamlit Dashboard (Port 7860) ]
```

---

## Key Modules

### 1. Data Contracts & Database Client (`src/schemas/`, `src/db/`)
- Strict Pydantic v2 schemas (`TelemetryRecord`, `ForecastRecord`, `AnomalyRecord`, `DriftRecord`) enforcing UTC timezone normalization and quantile ordering ($p_{10} \le p_{50} \le p_{90}$).
- Dual-mode database adapter (`DatabaseClient`) providing seamless Supabase PostgreSQL integration with transparent local SQLite fallback.

### 2. Automated Ingestion Engine (`src/ingestion/`)
- Reliable HTTP fetcher with exponential backoff and POSOCO diurnal profile fallback.
- CLI batch runner (`python -m src.ingestion.runner`) and 6-hour scheduled GitHub Actions workflow (`.github/workflows/ingest_telemetry.yml`).

### 3. Polars Feature Engineering (`src/features/`)
- Zero-leakage temporal feature computation using Polars SIMD expressions and lazy execution.
- Evaluates 100,000 telemetry rows in under 42ms (< 500ms target).

### 4. LightGBM Quantile Forecast Engine (`src/models/`)
- Separate pinball quantile regressors for lower ($p_{10}$), median ($p_{50}$), and upper ($p_{90}$) uncertainty intervals.
- 72-hour recursive autoregressive roll-forward inference dynamically recomputing lag and calendar features across future timesteps.
- Verified on 5-fold temporal walk-forward evaluation: WAPE < 1.0% (target < 8.0%) and 80% empirical interval coverage = 78.6% (target 75%-85%).

### 5. Isolation Forest & SHAP Attribution (`src/models/`)
- Calibrated 2% contamination detector flagging sudden load surges or industrial dropouts.
- Efficient selective attribution: computes `shap.TreeExplainer` feature drivers solely for anomalous timestamps, serializing the top 3 drivers with magnitude and direction.

### 6. Evidently AI Drift Monitoring (`src/monitoring/`)
- Compares rolling 7-day live telemetry against the 30-day training baseline reference window.
- Isolated handling for constant features to prevent numerical binning errors.
- Generates interactive, standalone HTML reports saved to `/reports/drift/` and stores summary JSON metrics in `drift_metrics`.

### 7. FastAPI REST Backend (`src/api/`)
- High-performance asynchronous REST endpoints operating on port 8000 with sub-200ms latency SLAs:
  - `GET /health`: Health check and active database mode.
  - `GET /api/telemetry/recent`: Recent regional demand telemetry.
  - `GET /api/telemetry/historical`: Range-filtered historical records.
  - `GET /api/forecasts/latest`: 72-hour quantile interval bands.
  - `GET /api/anomalies/recent`: Outlier scores with SHAP explanations.
  - `GET /api/drift/status`: Summary drift statistics.
  - `GET /api/drift/report`: Interactive HTML report streaming.

### 8. Streamlit Observability Dashboard (`src/ui/`)
- Clean, Apple-inspired light interface designed for visual clarity:
  - Real-time regional grid telemetry ribbons with interactive Plotly uncertainty bands.
  - Structural anomaly stress cards with horizontal SHAP attribution bar charts.
  - Evidently AI drift badges, metric share indicators, and report viewers.

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
git clone https://github.com/mahesh-kakde/grid-telemetry-engine.git
cd grid-telemetry-engine

# create virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the repository root:

```env
TELEMETRY_ENV=development
SQLITE_DB_PATH=data/local_telemetry.db
MODEL_STORE_PATH=models

# optional: configure supabase postgresql credentials
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_KEY=your-service-role-key
```

### Running the Services

To run both services in separate terminals:

```bash
# terminal 1: start fastapi backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# terminal 2: start streamlit dashboard
streamlit run src/ui/app.py --server.port 7860
```

Access the dashboard at `http://localhost:7860` and the API documentation at `http://localhost:8000/docs`.

---

## Testing & Verification

Execute the complete test suite:

```bash
pytest tests/ -v
```

Run individual verification modules:

```bash
# test feature extraction performance
pytest tests/test_features.py -v

# test walk-forward forecast accuracy
pytest tests/test_forecast.py -v

# test anomaly & shap explanations
pytest tests/test_anomaly.py -v

# test evidently drift calculations
pytest tests/test_monitoring.py -v

# test fastapi endpoints & latency
pytest tests/test_api.py -v

# test ui components
pytest tests/test_ui_components.py -v
```

---

## Docker Deployment

Build and run the unified single-container image locally or deploy to Render / Hugging Face Spaces:

```bash
# build container image
docker build -t grid-telemetry-engine .

# run container
docker run -p 7860:7860 -p 8000:8000 grid-telemetry-engine
```

---

## License

MIT License. Designed and maintained by Mahesh Kakde.