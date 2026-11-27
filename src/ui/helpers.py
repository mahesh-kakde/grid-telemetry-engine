from datetime import datetime, timezone
import html
import os
from typing import Any, Dict, Optional


SNAPSHOT_SOURCE_LABELS = {
    "supabase_live": "Live Supabase Cache",
    "local_snapshot": "Bundled Snapshot",
    "local_runtime_fallback": "Local Runtime Fallback",
}

REGION_SHORT_LABELS = {
    "ALL_INDIA": "All India",
    "NR": "NR - North",
    "WR": "WR - West",
    "SR": "SR - South",
    "ER": "ER - East",
    "NER": "NER - Northeast",
}

REGION_FULL_NAMES = {
    "ALL_INDIA": "All India",
    "NR": "NR — Northern Region",
    "WR": "WR — Western Region",
    "SR": "SR — Southern Region",
    "ER": "ER — Eastern Region",
    "NER": "NER — North Eastern Region",
}


def format_region_option(region_code: str) -> str:
    return REGION_SHORT_LABELS.get(region_code, region_code)


def render_not_found_html(favicon_href: Optional[str] = None) -> str:
    """Render shared accessible 404 HTML page for ASGI/API routes."""
    favicon_tag = (
        f'\n    <link rel="icon" type="image/png" href="{html.escape(favicon_href, quote=True)}">'
        if favicon_href
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 Not Found | grid-telemetry-engine</title>{favicon_tag}
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", sans-serif;
            background-color: #f5f5f7;
            color: #1d1d1f;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 16px;
        }}
        .error-card {{
            background-color: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 18px;
            padding: 52px 40px;
            text-align: center;
            max-width: 500px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
        }}
        .error-code {{ font-size: 76px; font-weight: 700; line-height: 1; margin-bottom: 10px; letter-spacing: -0.04em; color: #1d1d1f; }}
        .error-title {{ font-size: 22px; font-weight: 650; color: #1d1d1f; margin: 0 0 10px 0; }}
        .error-desc {{ font-size: 14.5px; color: #515154; line-height: 1.55; margin-bottom: 28px; }}
        .actions {{ display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }}
        .btn-home {{
            background-color: #0066cc; color: #ffffff; text-decoration: none; font-weight: 600;
            font-size: 14px; padding: 10px 26px; border-radius: 999px;
        }}
        .btn-home:hover {{ background-color: #0055b3; }}
        .btn-portfolio {{
            background-color: #ffffff; color: #1d1d1f; text-decoration: none; font-weight: 600;
            font-size: 14px; padding: 10px 26px; border-radius: 999px; border: 1px solid #d2d2d7;
        }}
        .btn-portfolio:hover {{ background-color: #f5f5f7; }}
    </style>
</head>
<body>
    <main class="error-card">
        <div class="error-code" aria-hidden="true">404</div>
        <h1 class="error-title">Page Not Found</h1>
        <div class="error-desc">The requested page or route does not exist or has been moved.</div>
        <div class="actions">
            <a href="/" class="btn-home">Return to Dashboard</a>
            <a href="https://maheshkakde.com" class="btn-portfolio" target="_blank" rel="noopener noreferrer">Portfolio</a>
        </div>
    </main>
</body>
</html>"""


def format_info_tip(tip: str, tip_right: bool = False) -> str:
    """Renders keyboard-focusable, ARIA-labeled tooltip trigger HTML."""
    escaped_tip = html.escape(tip, quote=True)
    extra_cls = " tip-right" if tip_right else ""
    return (
        f'<span class="info-tip{extra_cls}" tabindex="0" role="button" '
        f'aria-label="{escaped_tip}" data-tip="{escaped_tip}">?</span>'
    )


def evaluate_snapshot_freshness(
    snapshot: Dict[str, Any],
    stale_after_hours: float = 12.0,
    db_configured: Optional[bool] = None,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Computes snapshot source metadata, age in hours, and operator staleness warnings."""
    snap_src = str(snapshot.get("snapshot_source", "local_snapshot"))
    src_label = SNAPSHOT_SOURCE_LABELS.get(snap_src, "Local Runtime Fallback")
    gen_ts_raw = snapshot.get("generated_at", "")
    is_db_configured = bool(os.environ.get("SUPABASE_DB_URL")) if db_configured is None else bool(db_configured)
    ref_now = now_utc or datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    age_hours: Optional[float] = None
    gen_str = str(gen_ts_raw)[:19] if gen_ts_raw else "Unknown"
    if gen_ts_raw:
        try:
            iso_raw = str(gen_ts_raw).replace("Z", "+00:00")
            gen_dt = datetime.fromisoformat(iso_raw)
            if gen_dt.tzinfo is None:
                gen_dt = gen_dt.replace(tzinfo=timezone.utc)
            gen_utc = gen_dt.astimezone(timezone.utc)
            gen_str = gen_utc.strftime("%d %b %Y %H:%M UTC")
            age_hours = max(0.0, (ref_now.astimezone(timezone.utc) - gen_utc).total_seconds() / 3600.0)
        except Exception:
            age_hours = None

    is_degraded_fallback = is_db_configured and snap_src != "supabase_live"
    is_stale = age_hours is not None and age_hours > stale_after_hours

    warning_message: Optional[str] = None
    if is_degraded_fallback and is_stale and age_hours is not None:
        warning_message = (
            f"Live Supabase cache unavailable; serving {src_label.lower()} "
            f"generated {age_hours:.1f}h ago ({gen_str})."
        )
    elif is_degraded_fallback:
        warning_message = (
            f"Live Supabase cache unavailable; serving {src_label.lower()} ({gen_str})."
        )
    elif is_stale and age_hours is not None:
        warning_message = (
            f"Active {src_label.lower()} is {age_hours:.1f}h old "
            f"(exceeds {stale_after_hours:.0f}h cron window; refreshed {gen_str})."
        )

    return {
        "source": snap_src,
        "source_label": src_label,
        "generated_at_display": gen_str,
        "age_hours": age_hours,
        "is_degraded_fallback": is_degraded_fallback,
        "is_stale": is_stale,
        "warning_message": warning_message,
    }