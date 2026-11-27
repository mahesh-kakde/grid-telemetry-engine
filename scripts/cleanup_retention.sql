-- automated database retention cleanup procedure
-- enforces 90-day rolling window across physical telemetry, forecasts, and anomalies
-- keeps relational database footprint well within supabase 500 mb free tier

do $$
declare
    v_retention_days int := 90;
    v_cutoff timestamptz;
    v_deleted_telemetry int := 0;
    v_deleted_forecasts int := 0;
    v_deleted_anomalies int := 0;
begin
    v_cutoff := now() - (v_retention_days || ' days')::interval;

    -- 1. prune expired physical telemetry records
    delete from regional_telemetry
    where timestamp_utc < v_cutoff;
    get diagnostics v_deleted_telemetry = row_count;

    -- 2. prune expired model forecast horizons
    delete from model_forecasts
    where target_timestamp_utc < v_cutoff;
    get diagnostics v_deleted_forecasts = row_count;

    -- 3. prune expired anomaly and explainability events
    delete from telemetry_anomalies
    where timestamp_utc < v_cutoff;
    get diagnostics v_deleted_anomalies = row_count;

    raise notice 'retention cleanup completed (cutoff: %): % telemetry rows, % forecast rows, % anomaly rows pruned.',
        v_cutoff, v_deleted_telemetry, v_deleted_forecasts, v_deleted_anomalies;
end $$;