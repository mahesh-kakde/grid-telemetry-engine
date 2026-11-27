-- idempotent v2 schema migration for existing supabase databases
-- safe to re-run; preserves existing regional_telemetry and forecast history

create extension if not exists "uuid-ossp";

-- 1. enable row-level security on regional_telemetry (blocks postgrest anon/authenticated access)
alter table if exists regional_telemetry enable row level security;

-- 2. upgrade model_forecasts to preserve per-run forecast vintages + enable rls
alter table if exists model_forecasts enable row level security;
alter table if exists model_forecasts drop constraint if exists uq_forecast_target;

do $$
begin
    if exists (
        select 1 from information_schema.tables
        where table_schema = 'public' and table_name = 'model_forecasts'
    ) and not exists (
        select 1 from pg_constraint where conname = 'uq_forecast_vintage'
    ) then
        alter table model_forecasts
            add constraint uq_forecast_vintage
            unique (model_version, region, forecast_generated_at, target_timestamp_utc);
    end if;
end $$;

-- 3. upgrade telemetry_anomalies with interval bounds, breach flag, unique key + enable rls
alter table if exists telemetry_anomalies add column if not exists predicted_p10_mw double precision;
alter table if exists telemetry_anomalies add column if not exists predicted_p90_mw double precision;
alter table if exists telemetry_anomalies add column if not exists residual_breach boolean default false;
alter table if exists telemetry_anomalies enable row level security;

-- remove any historical duplicate (region, timestamp_utc) rows before adding unique constraint
do $$
begin
    if exists (
        select 1 from information_schema.tables
        where table_schema = 'public' and table_name = 'telemetry_anomalies'
    ) and not exists (
        select 1 from pg_constraint where conname = 'uq_anomaly_region_time'
    ) then
        delete from telemetry_anomalies a
        using telemetry_anomalies b
        where a.id < b.id
          and a.region = b.region
          and a.timestamp_utc = b.timestamp_utc;

        alter table telemetry_anomalies
            add constraint uq_anomaly_region_time
            unique (region, timestamp_utc);
    end if;
end $$;

create index if not exists idx_telemetry_timestamp
on regional_telemetry (timestamp_utc);

create index if not exists idx_forecasts_target_timestamp
on model_forecasts (target_timestamp_utc);

create index if not exists idx_forecasts_unbackfilled
on model_forecasts (region, target_timestamp_utc)
where actual_demand_mw is null;

create index if not exists idx_anomalies_timestamp
on telemetry_anomalies (timestamp_utc);

-- 4. create live precomputed snapshot table for decoupled cron-to-ui serving
create table if not exists precomputed_snapshots (
    region varchar(10) primary key,
    model_version varchar(32) not null,
    generated_at timestamptz not null,
    payload jsonb not null,
    drift_report_html text,
    updated_at timestamptz default now()
);

alter table precomputed_snapshots enable row level security;