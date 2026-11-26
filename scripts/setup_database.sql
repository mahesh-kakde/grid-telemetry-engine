-- enable uuid extension for primary keys
create extension if not exists "uuid-ossp";

-- 1. regional physical telemetry store
create table if not exists regional_telemetry (
    id uuid primary key default uuid_generate_v4(),
    timestamp_utc timestamptz not null,
    region varchar(10) not null,
    demand_mw double precision not null,
    frequency_hz double precision,
    temperature_2m double precision,
    relative_humidity_2m double precision,
    surface_solar_radiation double precision,
    wind_speed_100m double precision,
    created_at timestamptz default now(),
    constraint uq_region_timestamp unique (region, timestamp_utc)
);

create index if not exists idx_telemetry_region_time
on regional_telemetry (region, timestamp_utc desc);

alter table regional_telemetry enable row level security;

-- 2. forecast output repository (preserves per-run forecast vintages)
create table if not exists model_forecasts (
    id uuid primary key default uuid_generate_v4(),
    forecast_generated_at timestamptz not null,
    target_timestamp_utc timestamptz not null,
    region varchar(10) not null,
    forecast_p10 double precision not null,
    forecast_p50 double precision not null,
    forecast_p90 double precision not null,
    actual_demand_mw double precision,
    model_version varchar(32) not null,
    constraint uq_forecast_vintage unique (model_version, region, forecast_generated_at, target_timestamp_utc)
);

create index if not exists idx_forecasts_lookup
on model_forecasts (region, forecast_generated_at desc, target_timestamp_utc asc);

alter table model_forecasts enable row level security;

-- 3. anomaly and explainability events store
create table if not exists telemetry_anomalies (
    id uuid primary key default uuid_generate_v4(),
    timestamp_utc timestamptz not null,
    region varchar(10) not null,
    actual_demand_mw double precision not null,
    predicted_p10_mw double precision,
    predicted_p50_mw double precision,
    predicted_p90_mw double precision,
    anomaly_score double precision not null,
    is_structural boolean default false,
    residual_breach boolean default false,
    top_shap_features jsonb not null,
    created_at timestamptz default now(),
    constraint uq_anomaly_region_time unique (region, timestamp_utc)
);

create index if not exists idx_anomalies_region_time
on telemetry_anomalies (region, timestamp_utc desc);

alter table telemetry_anomalies enable row level security;

-- 4. live precomputed dashboard snapshot cache (one row per region)
create table if not exists precomputed_snapshots (
    region varchar(10) primary key,
    model_version varchar(32) not null,
    generated_at timestamptz not null,
    payload jsonb not null,
    drift_report_html text,
    updated_at timestamptz default now()
);

alter table precomputed_snapshots enable row level security;