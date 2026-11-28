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


NOT_FOUND_PAGE_TITLE = "404 Not Found | grid-telemetry-engine"
NOT_FOUND_HEADING = "This Page Went Off the Grid"
NOT_FOUND_DESCRIPTION = (
    "The telemetry feed or route you are looking for does not exist or has been rerouted."
)
NOT_FOUND_HOME_LABEL = "Home"

FAVICON_SVG_DATA_URI = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect x='1' y='1' width='30' height='30' rx='7' fill='%23ffffff' stroke='%23d2d2d7' stroke-width='1.2'/%3E"
    "%3Cpolygon points='5,19.5 11.5,16 18.5,9.5 26.5,11.2 26.5,19.8 18.5,18 11.5,23 5,24.5' fill='%230066cc' fill-opacity='0.22'/%3E"
    "%3Cpolyline points='5,22 11.5,19.5 18.5,13.8 26.5,15.5' fill='none' stroke='%230066cc' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round'/%3E"
    "%3Ccircle cx='18.5' cy='13.8' r='2.4' fill='%23b91c1c'/%3E%3C/svg%3E"
)


def render_not_found_html(favicon_href: Optional[str] = None) -> str:
    """Render shared accessible 404 HTML page matching the dashboard light design system."""
    if favicon_href:
        favicon_tag = f'\n    <link rel="icon" type="image/png" href="{html.escape(favicon_href, quote=True)}">'
    else:
        favicon_tag = f'\n    <link rel="icon" type="image/svg+xml" href="{FAVICON_SVG_DATA_URI}">'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{NOT_FOUND_PAGE_TITLE}</title>{favicon_tag}
    <style>
        :root {{
            --bg-canvas: #f5f5f7;
            --bg-surface: #ffffff;
            --text-primary: #1d1d1f;
            --text-secondary: #515154;
            --text-muted: #6e6e73;
            --accent-blue: #0066cc;
            --accent-blue-hover: #0055b3;
            --border-hairline: #d2d2d7;
            --radius-card: 16px;
            --shadow-card: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 14px rgba(0, 0, 0, 0.03);
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }}
        .site-header {{
            position: sticky;
            top: 0;
            z-index: 50;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-hairline);
        }}
        .site-header-inner {{
            max-width: 1240px;
            margin: 0 auto;
            padding: 14px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .brand-title {{
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: var(--text-primary);
            line-height: 1.2;
        }}
        .page-main {{
            flex: 1 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 36px 24px;
        }}
        .error-card {{
            background-color: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-card);
            padding: 48px 40px;
            text-align: center;
            max-width: 500px;
            width: 100%;
            box-shadow: var(--shadow-card);
        }}
        .error-code {{
            font-size: 72px;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 10px;
            letter-spacing: -0.04em;
            font-variant-numeric: tabular-nums;
            color: var(--text-primary);
        }}
        .error-title {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
            margin: 0 0 8px 0;
            line-height: 1.25;
        }}
        .error-desc {{
            font-size: 14.5px;
            color: var(--text-secondary);
            line-height: 1.55;
            margin: 0 0 28px 0;
        }}
        .actions {{
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .btn-home {{
            background-color: var(--accent-blue);
            color: #ffffff;
            text-decoration: none;
            font-weight: 600;
            font-size: 13.5px;
            padding: 10px 22px;
            border-radius: 10px;
            border: 1px solid var(--accent-blue);
            transition: background-color 0.15s ease;
        }}
        .btn-home:hover {{
            background-color: var(--accent-blue-hover);
            border-color: var(--accent-blue-hover);
        }}
        .btn-portfolio {{
            background-color: var(--bg-surface);
            color: var(--text-primary);
            text-decoration: none;
            font-weight: 600;
            font-size: 13.5px;
            padding: 10px 22px;
            border-radius: 10px;
            border: 1px solid var(--border-hairline);
            transition: background-color 0.15s ease;
        }}
        .btn-portfolio:hover {{
            background-color: #f2f2f7;
        }}
        .site-footer {{
            flex-shrink: 0;
            background: var(--bg-surface);
            border-top: 1px solid var(--border-hairline);
            margin-top: auto;
        }}
        .site-footer-inner {{
            max-width: 1240px;
            margin: 0 auto;
            padding: 14px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            font-size: 13px;
            color: var(--text-muted);
        }}
        .site-footer a {{
            color: var(--accent-blue);
            text-decoration: none;
            font-weight: 500;
        }}
        .site-footer a:hover {{
            text-decoration: underline;
        }}
        @media (max-width: 640px) {{
            .site-header-inner,
            .site-footer-inner {{
                padding: 12px 16px;
            }}
            .page-main {{
                padding: 24px 16px;
            }}
            .error-card {{
                padding: 36px 24px;
            }}
            .actions {{
                flex-direction: column;
            }}
            .btn-home,
            .btn-portfolio {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <header class="site-header">
        <div class="site-header-inner">
            <div class="brand-title">National Power Grid Telemetry</div>
        </div>
    </header>

    <main class="page-main">
        <section class="error-card">
            <div class="error-code" aria-hidden="true">404</div>
            <h1 class="error-title">{html.escape(NOT_FOUND_HEADING)}</h1>
            <p class="error-desc">{html.escape(NOT_FOUND_DESCRIPTION)}</p>
            <div class="actions">
                <a href="/" class="btn-home">{html.escape(NOT_FOUND_HOME_LABEL)}</a>
                <a href="https://maheshkakde.com" class="btn-portfolio" target="_blank" rel="noopener noreferrer">Portfolio</a>
            </div>
        </section>
    </main>

    <footer class="site-footer">
        <div class="site-footer-inner">
            <span>Created by <a href="https://maheshkakde.com" target="_blank" rel="noopener noreferrer">Mahesh Kakde</a></span>
            <a href="https://github.com/mahesh-kakde/grid-telemetry-engine" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
    </footer>
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