-- schema.sql: postgresql schema for grid-telemetry-engine

-- raw telemetry ingestion table
create table if not exists telemetry_raw (
    id uuid primary key,
    timestamp timestamptz not null,
    region varchar(16) not null,
    metric_value double precision not null check (metric_value > 0),
    metric_type varchar(32) not null default 'demand_mw',
    created_at timestamptz not null default now(),
    constraint uq_telemetry_region_timestamp unique (region, timestamp, metric_type)
);

create index if not exists idx_telemetry_region_timestamp
    on telemetry_raw (region, timestamp desc);

-- 72-hour recursive quantile forecasts
create table if not exists forecasts (
    id uuid primary key,
    forecast_origin timestamptz not null,
    target_timestamp timestamptz not null,
    region varchar(16) not null,
    p10 double precision not null,
    p50 double precision not null,
    p90 double precision not null,
    model_version varchar(32) not null default 'v1.0',
    created_at timestamptz not null default now(),
    constraint uq_forecasts_origin_target_region unique (forecast_origin, target_timestamp, region, model_version),
    constraint chk_quantiles check (p10 >= 0 and p10 <= p50 and p50 <= p90)
);

create index if not exists idx_forecasts_region_target
    on forecasts (region, target_timestamp);

create index if not exists idx_forecasts_origin
    on forecasts (region, forecast_origin desc);

-- structural anomaly detections with shap explanations
create table if not exists anomalies (
    id uuid primary key,
    timestamp timestamptz not null,
    region varchar(16) not null,
    metric_value double precision not null,
    anomaly_score double precision not null,
    is_anomaly boolean not null default false,
    top_shap_features jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint uq_anomalies_region_timestamp unique (region, timestamp)
);

create index if not exists idx_anomalies_region_timestamp
    on anomalies (region, timestamp desc);

create index if not exists idx_anomalies_flagged
    on anomalies (is_anomaly) where is_anomaly = true;

-- continuous data and concept drift metrics
create table if not exists drift_metrics (
    id uuid primary key,
    run_timestamp timestamptz not null default now(),
    reference_window_start timestamptz not null,
    reference_window_end timestamptz not null,
    current_window_start timestamptz not null,
    current_window_end timestamptz not null,
    drift_share double precision not null check (drift_share >= 0.0 and drift_share <= 1.0),
    drifted_features jsonb not null default '{}'::jsonb,
    report_url text,
    created_at timestamptz not null default now()
);

create index if not exists idx_drift_run_timestamp
    on drift_metrics (run_timestamp desc);