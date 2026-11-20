# National Power Grid Telemetry & Forecasting Engine

Production AI/ML system delivering real-time telemetry ingestion, 72-hour multi-horizon probabilistic load forecasting, dual-trigger structural anomaly detection, SHAP model explainability, and automated statistical drift monitoring for India's National Power Grid.

Live Deployment: [grid.maheshkakde.com](https://grid.maheshkakde.com)

---

## 1. System Architecture & Core Capabilities

The engine coordinates five decoupled subsystems:

1. **Scheduled Ingestion Pipeline:** Runs every 6 hours via GitHub Actions (`15 */6 * * *`), pulling grid demand and frequency telemetry from POSOCO / NLDC / data.gov.in and meteorological observations from Open-Meteo (2m temperature, relative humidity, surface solar radiation, 100m wind speed). Normalizes timestamps to UTC and enforces physical boundary validation with Pydantic v2.
2. **Zero-Leakage Feature Pipeline:** Vectorized Polars LazyFrames computing autoregressive lags (1h, 2h, 3h, 24h, 48h, 168h), rolling window statistics (6h, 24h, 168h), and continuous Fourier diurnal and annual harmonics ($T=24\text{h}, T=365.25\text{d}$). Applies strict `.shift(1)` temporal fences before rolling aggregations to guarantee zero future lookahead bias.
3. **Multi-Horizon Quantile Forecaster:** LightGBM gradient boosters trained under Pinball loss producing probabilistic uncertainty bounds ($p10, p50, p90$) across a 72-hour recursive forward horizon. Applies post-inference isotonic sorting along axis 1 to guarantee quantile monotonicity ($p10 \le p50 \le p90$).
4. **Dual-Trigger Structural Anomaly Detection:** Outliers require simultaneous confirmation from **Trigger A** (unsupervised Isolation Forest feature isolation score, contamination 0.01) and **Trigger B** (predictive residual IQR envelope violation: $y > p90 + 1.5\text{IQR} \lor y < p10 - 1.5\text{IQR}$), eliminating false alarms from benign weather-driven demand surges.
5. **Explainability & Observability Layer:** Uses SHAP `TreeExplainer` with $k=150$ background centroids to extract the top-3 physical feature drivers per anomaly. Evaluates continuous feature drift using two-sample Kolmogorov-Smirnov tests ($p < 0.05$) and cyclical/categorical drift using 1D Wasserstein distances ($W_1 \ge 0.10$) comparing trailing 7-day telemetry against a 30-day reference baseline.
6. **Co-Located Container Serving:** Single multi-stage Docker container hosting an internal FastAPI REST service on loopback (`127.0.0.1:8000`) and an interactive Streamlit dashboard bound to external `$PORT` / 7860.

---

## 2. Regional Dispatch Coverage

The engine models India's 5 Regional Load Despatch Centres (RLDCs) and aggregate national demand:
- **NR:** Northern Region (high weather volatility, agricultural irrigation load)
- **WR:** Western Region (heavy industrial manufacturing, renewable penetration)
- **SR:** Southern Region (monsoonal shifts, IT hubs, coastal humidity)
- **ER:** Eastern Region (mining, steel, and heavy industrial centers)
- **NER:** North Eastern Region (hydro-dominated, high-altitude terrain)
- **ALL_INDIA:** Aggregate national active electrical load

---

## 3. Repository Structure

```
grid-telemetry-engine/
|-- .github/
|   \-- workflows/
|       |-- ingest.yml            # 6-hour cron (ETL, inference, drift monitoring)
|       \-- test.yml              # CI test runner (push/PR verification)
|-- .streamlit/
|   \-- config.toml               # Streamlit theme and edge configuration
|-- artifacts/                    # Pretrained LightGBM models, baselines, centroids
|   |-- feature_columns.json
|   |-- models/
|   \-- baselines/
|-- src/
|   |-- ingestion/
|   |   |-- fetch_telemetry.py    # REST client, backoff circuit breaker, gap filler
|   |   \-- schemas.py            # Pydantic v2 validation contracts
|   |-- features/
|   |   \-- build_features.py     # Polars zero-leakage feature pipeline
|   |-- models/
|   |   |-- infer_and_detect.py   # Quantile forecaster & dual-trigger detector
|   |   |-- load_artifacts.py     # In-memory artifact loader and bundle cache
|   |   \-- explainers.py         # SHAP TreeExplainer attribution engine
|   |-- monitoring/
|   |   \-- drift_check.py        # Evidently AI KS & Wasserstein drift engine
|   |-- api/
|   |   \-- main.py               # Internal FastAPI loopback service
|   \-- ui/
|       \-- app.py                # Streamlit dashboard
|-- tests/
|   |-- test_ingestion.py         # Ingestion and schema validation tests
|   |-- test_features.py          # Temporal causality and zero-leakage tests
|   \-- test_models.py            # Quantile monotonicity and anomaly tests
|-- static/
|   \-- reports/
|       \-- drift_report.html     # Generated standalone Evidently report
|-- Dockerfile                    # Multi-stage production container
|-- start.sh                      # Dual-process entrypoint supervisor
|-- render.yaml                   # Render deployment blueprint
|-- requirements.txt              # Pinned production dependencies
\-- README.md
```

---

## 4. Local Quickstart

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run src/ui/app.py
```

### Running the FastAPI Backend Standalone
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Running Automated Test Suite
```bash
pytest tests/ -v
```

---

## 5. Container Execution

```bash
docker build -t grid-telemetry-engine .
docker run -p 7860:7860 grid-telemetry-engine
```

---

## Author
Mahesh Kakde ([maheshkakde.com](https://maheshkakde.com))