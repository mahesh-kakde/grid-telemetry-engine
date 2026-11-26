import os
import sys

# ensure project root is present in sys.path when running standalone streamlit commands
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

REGIONS = ["ALL_INDIA", "NR", "WR", "SR", "ER", "NER"]
REGION_NAMES = {
    "ALL_INDIA": "ALL_INDIA — National Grid",
    "NR": "NR — Northern Region",
    "WR": "WR — Western Region",
    "SR": "SR — Southern Region",
    "ER": "ER — Eastern Region",
    "NER": "NER — North Eastern Region",
}
MODEL_VERSION = "2.0.0"
DEFAULT_REPORT_PATH = os.path.join(root_dir, "static", "reports", "drift_report.html")

from src.monitoring.drift_check import (
    FEATURE_METADATA,
    build_html_drift_report,
)


# 404 error page routing check
query_page = str(st.query_params.get("page", "")).lower()
query_error = str(st.query_params.get("error", "")).lower()
is_404 = query_page in ["404", "not_found", "notfound"] or query_error == "404"

if hasattr(st, "context") and getattr(st.context, "url", None):
    try:
        from urllib.parse import urlparse
        req_path = urlparse(st.context.url).path.strip("/").lower()
        if req_path and req_path not in ["", "app"]:
            is_404 = True
    except Exception:
        pass

# page configuration
FAVICON_PATH = os.path.join(root_dir, "static", "favicon.png")
app_page_title = (
    "404 - Page Not Found | grid-telemetry-engine"
    if is_404
    else "grid-telemetry-engine | Mahesh Kakde"
)

st.set_page_config(
    page_title=app_page_title,
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
    initial_sidebar_state="collapsed" if is_404 else "expanded",
)

if is_404:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container {
            max-width: 800px !important;
            padding-top: 5rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            margin: 0 auto !important;
        }
        .error-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 56px 36px;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -2px rgba(15, 23, 42, 0.03);
            margin-top: 2rem;
        }
        .error-code {
            font-size: 88px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1;
            letter-spacing: -0.04em;
            margin-bottom: 8px;
        }
        .error-title {
            font-size: 22px;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 12px;
            letter-spacing: -0.01em;
        }
        .error-description {
            font-size: 14.5px;
            color: #64748b;
            max-width: 480px;
            line-height: 1.6;
            margin-bottom: 32px;
        }
        .error-actions {
            display: flex;
            gap: 16px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }
        .btn-home {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: #2563eb;
            color: #ffffff !important;
            text-decoration: none !important;
            font-weight: 600;
            font-size: 14px;
            padding: 10px 28px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(37, 99, 235, 0.25);
            transition: background-color 0.15s ease;
        }
        .btn-home:hover {
            background-color: #1d4ed8;
            color: #ffffff !important;
        }
        .btn-portfolio {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: #ffffff;
            color: #0f172a !important;
            text-decoration: none !important;
            font-weight: 600;
            font-size: 14px;
            padding: 10px 28px;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
            transition: all 0.15s ease;
        }
        .btn-portfolio:hover {
            background-color: #f8fafc;
            border-color: #94a3b8;
            color: #0f172a !important;
        }
        .fixed-footer-404 {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(8px);
            border-top: 1px solid #e2e8f0;
            padding: 8px 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: #64748b;
            z-index: 99999;
            white-space: nowrap;
            box-sizing: border-box;
        }
        .fixed-footer-404 a {
            color: #2563eb;
            text-decoration: none;
            font-weight: 500;
        }
        .fixed-footer-404 a:hover {
            text-decoration: underline;
        }
        @media (max-width: 640px) {
            .error-container {
                padding: 36px 18px !important;
                margin-top: 1rem !important;
            }
            .error-code {
                font-size: 64px !important;
            }
            .error-title {
                font-size: 18px !important;
            }
            .error-description {
                font-size: 13.5px !important;
                margin-bottom: 24px !important;
            }
            .error-actions {
                flex-direction: column !important;
                width: 100% !important;
                gap: 10px !important;
            }
            .btn-home, .btn-portfolio {
                width: 100% !important;
                text-align: center !important;
            }
            .fixed-footer-404 {
                flex-direction: row !important;
                justify-content: space-between !important;
                align-items: center !important;
                white-space: nowrap !important;
                gap: 0 !important;
                padding: 4px 10px !important;
                font-size: 10px !important;
                height: 30px !important;
                box-sizing: border-box !important;
            }
        }
        </style>
        <div class="error-container">
            <div class="error-code">404</div>
            <div class="error-title">Page Not Found</div>
            <div class="error-description">
                The power grid telemetry dispatch route or resource you are looking for does not exist or has been moved.
            </div>
            <div class="error-actions">
                <a href="https://grid.maheshkakde.com" class="btn-home" target="_self">Home</a>
                <a href="https://maheshkakde.com" class="btn-portfolio" target="_blank" rel="noopener noreferrer">Portfolio</a>
            </div>
        </div>
        <div class="fixed-footer-404">
            <div>
                Created by <a href="https://maheshkakde.com" target="_blank" rel="noopener noreferrer">Mahesh Kakde</a>
                <span style="margin: 0 4px; color: #cbd5e1;">·</span>
                <a href="https://github.com/mahesh-kakde" target="_blank" rel="noopener noreferrer">GitHub</a>
            </div>
            <div>
                <a href="https://github.com/mahesh-kakde/grid-telemetry-engine" target="_blank" rel="noopener noreferrer">Repository</a>
            </div>
        </div>
        <script>
            document.title = "404 - Page Not Found | grid-telemetry-engine";
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# custom css aligned with modern enterprise light-mode design system
st.markdown(
    """
    <style>
    /* modern enterprise light mode design system (Stripe / Linear / Vercel style) */
    /* zero-height container for iframe meta injectors to prevent layout shift */
    iframe[title="streamlit_components.v1.html"],
    div[data-testid="stCustomComponentV1"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }
    html, body, [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif;
        color: #0f172a;
        background-color: #f8fafc !important;
    }
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 100 !important;
    }
    /* sidebar mobile collapsed control */
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
        margin-left: 0.5rem !important;
        margin-top: 0.25rem !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
    }
    /* sidebar refined styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebarHeader"] {
        padding-top: 0.5rem !important;
        padding-bottom: 0rem !important;
        background-color: #ffffff !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        color: #64748b !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover {
        color: #0f172a !important;
        background-color: #f1f5f9 !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 0.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-bottom: 1rem !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.75rem 0 !important;
        border-color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h4 {
        margin-top: 0rem !important;
        margin-bottom: 0.5rem !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        letter-spacing: -0.01em;
    }
    /* typography */
    h1 {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
        margin-bottom: 0.25rem !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        font-size: 17px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        letter-spacing: -0.01em;
        margin-top: 0.75rem !important;
        margin-bottom: 0.25rem !important;
    }
    h4 {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    /* clean tab navigation (no pill box, clean modern underline) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid #e2e8f0 !important;
        border-radius: 0 !important;
        padding: 0 !important;
        gap: 24px !important;
        margin-top: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #64748b !important;
        padding: 8px 0 !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        transition: all 0.15s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent !important;
        color: #2563eb !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2563eb !important;
        box-shadow: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    /* cards and elevation */
    .kpi-card, .metric-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        background-color: #ffffff;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02);
        transition: all 0.15s ease-in-out;
        overflow: visible !important;
    }
    .kpi-card:hover, .metric-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.03);
    }
    .kpi-title, .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-value, .metric-value {
        font-size: clamp(16px, 1.35vw, 21px);
        font-weight: 700;
        color: #0f172a;
        font-feature-settings: "tnum";
        font-variant-numeric: tabular-nums;
        line-height: 1.25;
        margin-bottom: 3px;
        word-break: normal !important;
        overflow-wrap: normal !important;
    }
    .kpi-caption, .metric-caption {
        font-size: 11.5px;
        color: #64748b;
        line-height: 1.35;
    }
    /* text status styling (no pill badges or containers) */
    .badge-normal {
        display: inline;
        font-weight: 700;
        color: #047857;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        border-radius: 0 !important;
    }
    .badge-alert {
        display: inline;
        font-weight: 700;
        color: #dc2626;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        border-radius: 0 !important;
    }
    .mono-tag {
        font-family: inherit;
        font-weight: 600;
        color: inherit;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        border-radius: 0 !important;
    }
    /* clean info tooltips */
    .info-tip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #64748b;
        font-size: 10px;
        font-weight: 600;
        margin-left: 6px;
        cursor: pointer;
        outline: none;
        -webkit-tap-highlight-color: transparent;
        position: relative;
        vertical-align: middle;
        line-height: 1;
        transition: all 0.15s ease-in-out;
    }
    .info-tip:hover,
    .info-tip:focus,
    .info-tip:active {
        background-color: #2563eb;
        border-color: #2563eb;
        color: #ffffff;
    }
    .info-tip:hover::after {
        content: attr(data-tip);
        position: absolute;
        bottom: 145%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #0f172a;
        color: #f8fafc;
        font-size: 11.5px;
        font-weight: 400;
        line-height: 1.45;
        padding: 8px 12px;
        border-radius: 6px;
        white-space: normal;
        width: 250px;
        z-index: 9999999;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.15), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        pointer-events: none;
        text-transform: none;
        letter-spacing: normal;
    }
    .info-tip:hover::before {
        content: "";
        position: absolute;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 5px;
        border-style: solid;
        border-color: #0f172a transparent transparent transparent;
        z-index: 9999999;
    }
    /* right-anchored tooltip for cards on the right side to prevent viewport clipping */
    .info-tip.tip-right:hover::after,
    [data-testid="column"]:nth-child(3) .info-tip:hover::after,
    [data-testid="column"]:nth-child(4) .info-tip:hover::after,
    [data-testid="column"]:last-child .info-tip:hover::after {
        left: auto !important;
        right: -8px !important;
        transform: none !important;
    }
    .info-tip.tip-right:hover::before,
    [data-testid="column"]:nth-child(3) .info-tip:hover::before,
    [data-testid="column"]:nth-child(4) .info-tip:hover::before,
    [data-testid="column"]:last-child .info-tip:hover::before {
        left: auto !important;
        right: 2px !important;
        transform: none !important;
    }
    /* fixed footer */
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 300px;
        width: calc(100% - 300px);
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        border-top: 1px solid #e2e8f0;
        padding: 6px 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
        color: #64748b;
        z-index: 99999;
        box-shadow: 0 -1px 3px rgba(0, 0, 0, 0.02);
        white-space: nowrap;
        box-sizing: border-box;
    }
    .stApp:has([data-testid="stSidebar"][aria-expanded="false"]) .fixed-footer {
        left: 0 !important;
        width: 100% !important;
    }
    .fixed-footer a {
        color: #2563eb;
        text-decoration: none;
        font-weight: 500;
    }
    .fixed-footer a:hover {
        text-decoration: underline;
        color: #1d4ed8;
    }
    .footer-sep {
        margin: 0 8px;
        color: #cbd5e1;
    }
    /* mobile chart hint badge (hidden on desktop) */
    .chart-mobile-hint {
        display: none;
    }
    /* prevent cumulative layout shift (CLS) by reserving minimum height for charts */
    [data-testid="stPlotlyChart"],
    .stPlotlyChart {
        min-height: 440px;
    }
    /* =====================================================================
       MOBILE & TABLET RESPONSIVE STYLES
       ===================================================================== */
    @media (max-width: 768px) {
        /* lock mobile viewport: zero horizontal overflow */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            overflow-x: hidden !important;
            max-width: 100% !important;
            width: 100% !important;
        }

        /* mobile main content full viewport - zero layout shift */
        [data-testid="stAppViewContainer"] > div,
        .stAppViewContainer > div,
        .stAppViewContainer > div:not([data-testid="stSidebar"]),
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .main,
        section[data-testid="stMain"] {
            width: 100% !important;
            max-width: 100% !important;
            left: 0 !important;
            margin-left: 0 !important;
            transform: none !important;
            transition: none !important;
        }

        /* container padding */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 4.5rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            max-width: 100% !important;
        }

        /* mobile sidebar slide-over overlay drawer */
        [data-testid="stSidebar"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            bottom: 0 !important;
            height: 100vh !important;
            z-index: 999999 !important;
            width: min(300px, 82vw) !important;
            max-width: 82vw !important;
            box-shadow: 4px 0 24px rgba(15, 23, 42, 0.16) !important;
            transition: none !important;
            animation: none !important;
        }

        /* typography scaling */
        h1 {
            font-size: 19px !important;
            line-height: 1.25 !important;
            margin-top: 0.25rem !important;
            margin-bottom: 0.25rem !important;
        }
        h2, h3 {
            font-size: 15px !important;
            line-height: 1.3 !important;
        }

        /* tabs touch-friendly horizontal scrolling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px !important;
            overflow-x: auto !important;
            scrollbar-width: none !important;
            -webkit-overflow-scrolling: touch !important;
            flex-wrap: nowrap !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.85rem !important;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 12.5px !important;
            white-space: nowrap !important;
            padding: 6px 2px !important;
            flex-shrink: 0 !important;
        }

        /* column layout: responsive wrapping */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 calc(50% - 8px) !important;
            min-width: calc(50% - 8px) !important;
        }

        /* card scaling */
        .kpi-card, .metric-card {
            padding: 10px 12px !important;
            margin-bottom: 6px !important;
        }
        .kpi-title, .metric-label {
            font-size: 10px !important;
        }
        .kpi-value, .metric-value {
            font-size: 18px !important;
            margin-bottom: 2px !important;
        }
        .kpi-caption, .metric-caption {
            font-size: 10.5px !important;
        }

        /* tooltips on mobile: anchored right above the ? icon with arrow pointer */
        .info-tip:hover::after,
        .info-tip:focus::after,
        .info-tip:active::after {
            position: absolute !important;
            bottom: calc(100% + 8px) !important;
            left: 50% !important;
            right: auto !important;
            top: auto !important;
            transform: translateX(-50%) !important;
            width: min(210px, 72vw) !important;
            max-width: 72vw !important;
            font-size: 11px !important;
            line-height: 1.4 !important;
            padding: 8px 10px !important;
            border-radius: 6px !important;
            background-color: #0f172a !important;
            color: #f8fafc !important;
            box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.3) !important;
            border: 1px solid #334155 !important;
            z-index: 9999999 !important;
            white-space: normal !important;
            pointer-events: none !important;
        }
        .info-tip:hover::before,
        .info-tip:focus::before,
        .info-tip:active::before {
            display: block !important;
            content: "" !important;
            position: absolute !important;
            bottom: calc(100% + 2px) !important;
            left: 50% !important;
            right: auto !important;
            top: auto !important;
            transform: translateX(-50%) !important;
            border-width: 6px !important;
            border-style: solid !important;
            border-color: #0f172a transparent transparent transparent !important;
            z-index: 9999999 !important;
        }

        /* right-anchored tooltips on mobile (e.g. second column cards or tip-right) */
        .info-tip.tip-right:hover::after,
        .info-tip.tip-right:focus::after,
        .info-tip.tip-right:active::after,
        [data-testid="column"]:nth-child(even) .info-tip:hover::after,
        [data-testid="column"]:nth-child(even) .info-tip:focus::after,
        [data-testid="column"]:nth-child(even) .info-tip:active::after {
            left: auto !important;
            right: -6px !important;
            transform: none !important;
        }
        .info-tip.tip-right:hover::before,
        .info-tip.tip-right:focus::before,
        .info-tip.tip-right:active::before,
        [data-testid="column"]:nth-child(even) .info-tip:hover::before,
        [data-testid="column"]:nth-child(even) .info-tip:focus::before,
        [data-testid="column"]:nth-child(even) .info-tip:active::before {
            left: auto !important;
            right: 2px !important;
            transform: none !important;
        }

        /* fixed footer mobile alignment: left & right spaced */
        .fixed-footer {
            left: 0 !important;
            width: 100% !important;
            padding: 5px 12px !important;
            font-size: 11px !important;
            display: flex !important;
            flex-direction: row !important;
            justify-content: space-between !important;
            align-items: center !important;
            white-space: nowrap !important;
            height: 32px !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }

        /* mobile chart hint badge */
        .chart-mobile-hint {
            display: block !important;
            font-size: 11px !important;
            color: #475569 !important;
            background-color: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 6px 10px !important;
            margin-bottom: 8px !important;
            margin-top: 2px !important;
            line-height: 1.4 !important;
        }
        .js-plotly-plot {
            max-height: 400px !important;
        }
        [data-testid="stPlotlyChart"],
        .stPlotlyChart {
            min-height: 360px !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 4px !important;
        }
        div[data-testid="stRadio"] label {
            font-size: 11px !important;
        }

        /* download button full-width touch friendly */
        div[data-testid="stDownloadButton"] button {
            width: 100% !important;
            padding: 12px 16px !important;
            text-align: center !important;
        }
    }

    @media (max-width: 480px) {
        /* full width single column on compact phones */
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        .fixed-footer {
            left: 0 !important;
            width: 100% !important;
            flex-direction: row !important;
            justify-content: space-between !important;
            align-items: center !important;
            white-space: nowrap !important;
            padding: 4px 10px !important;
            font-size: 10px !important;
            height: 30px !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        .footer-sep {
            margin: 0 4px !important;
        }
    }
    /* select and controls */
    [data-baseweb="select"] {
        font-size: 13px !important;
    }
    [data-baseweb="popover"] {
        min-width: 280px !important;
    }
    li[role="option"] {
        font-size: 13px !important;
        white-space: normal !important;
        word-break: normal !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        margin-top: 12px;
    }
    /* prominent high-contrast download report button */
    div[data-testid="stDownloadButton"] button {
        background-color: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        box-shadow: 0 1px 2px 0 rgba(37, 99, 235, 0.2) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1e40af !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3) !important;
    }
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stDownloadButton"] button *,
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stDownloadButton"] button span,
    div[data-testid="stDownloadButton"] button div {
        color: #ffffff !important;
        fill: #ffffff !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# persistent fixed footer rendered immediately on initial load
st.markdown(
    """
    <div class="fixed-footer">
        <div>
            Created by <a href="https://maheshkakde.com" target="_blank" rel="noopener noreferrer">Mahesh Kakde</a>
            <span class="footer-sep">·</span>
            <a href="https://github.com/mahesh-kakde" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
        <div>
            <a href="https://github.com/mahesh-kakde/grid-telemetry-engine" target="_blank" rel="noopener noreferrer">Repository</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_cached_artifacts():
    # caches in-memory bundle with on-demand lazy loading
    from src.models.load_artifacts import ModelArtifactsBundle
    from src.models.explainers import AnomalySHAPExplainer
    bundle = ModelArtifactsBundle()
    explainer = AnomalySHAPExplainer(bundle)
    return bundle, explainer


@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot(region: str) -> Dict[str, Any]:
    # loads live snapshot from supabase precomputed_snapshots or local json fallback
    from src.models.precompute import load_precomputed_data
    snap = load_precomputed_data(region)
    if snap is not None and snap.get("forecasts") and snap.get("nominal_anomaly") and snap.get("drift_summary"):
        return snap

    # local inference fallback if neither supabase nor bundled json snapshot is available
    from src.models.precompute import precompute_region
    bundle, explainer = get_cached_artifacts()
    snap = precompute_region(region, bundle, explainer, use_live_weather=False)
    snap["snapshot_source"] = "local_runtime_fallback"
    return snap


def fetch_forecasts(region: str) -> List[Dict[str, Any]]:
    # parses 72-hour probabilistic forecast series from active snapshot
    snap = load_snapshot(region)
    parsed = []
    for fc in snap["forecasts"]:
        parsed.append({
            "target_timestamp_utc": pd.to_datetime(fc["target_timestamp_utc"]),
            "forecast_p10": float(fc["forecast_p10"]),
            "forecast_p50": float(fc["forecast_p50"]),
            "forecast_p90": float(fc["forecast_p90"]),
            "region": fc["region"],
        })
    return parsed


@st.cache_data(ttl=300, show_spinner=False)
def evaluate_live_anomaly(region: str, custom_demand: Optional[float] = None) -> Dict[str, Any]:
    # evaluates nominal sentinel state from live snapshot or scores custom stress-test load locally
    if custom_demand is None:
        return load_snapshot(region)["nominal_anomaly"]

    from src.features.build_features import compute_telemetry_features
    from src.models.infer_and_detect import (
        fetch_live_or_baseline_history,
        predict_quantiles,
        detect_structural_anomalies,
    )

    bundle, explainer = get_cached_artifacts()
    df_history, _ = fetch_live_or_baseline_history(region, lookback_hours=240, bundle=bundle)
    feat_pl = compute_telemetry_features(df_history, drop_nulls_flag=True)
    feat_df = bundle.reference_baselines[region] if feat_pl.is_empty() else feat_pl.to_pandas()
    feature_row = feat_df.iloc[-1].copy()
    features_df = pd.DataFrame([feature_row[bundle.feature_columns]])

    widened = bool(load_snapshot(region).get("widened", False))
    p10, p50, p90 = predict_quantiles(features_df, region, bundle, widened=widened)
    actual = float(custom_demand)

    feature_row["demand_mw"] = actual
    if "demand_lag_1h" in feature_row:
        feature_row["demand_delta_1h"] = actual - float(feature_row["demand_lag_1h"])
    if "demand_lag_24h" in feature_row:
        feature_row["demand_delta_24h"] = actual - float(feature_row["demand_lag_24h"])
    if "demand_rolling_mean_24h" in feature_row and "demand_rolling_std_24h" in feature_row:
        denom = max(float(feature_row["demand_rolling_std_24h"]), 1.0)
        feature_row["demand_zscore_24h"] = (actual - float(feature_row["demand_rolling_mean_24h"])) / denom

    scores, is_structural, trigger_b = detect_structural_anomalies(
        X=features_df,
        y_actual=np.array([actual]),
        p10=p10,
        p50=p50,
        p90=p90,
        region=region,
        bundle=bundle,
    )
    top_shap = explainer.explain_anomaly_point(feature_row, region, top_k=3)

    return {
        "region": region,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "actual_demand_mw": round(actual, 2),
        "predicted_p10_mw": round(float(p10[0]), 2),
        "predicted_p50_mw": round(float(p50[0]), 2),
        "predicted_p90_mw": round(float(p90[0]), 2),
        "anomaly_score": round(float(scores[0]), 4),
        "is_structural": bool(is_structural[0]),
        "residual_breach": bool(trigger_b[0]),
        "top_shap_features": top_shap,
    }


def fetch_drift_summary(region: str) -> Dict[str, Any]:
    # retrieves statistical covariate drift summary from active snapshot
    return load_snapshot(region)["drift_summary"]


@st.cache_data(ttl=300, show_spinner=False)
def get_forecast_dataframe(region: str) -> pd.DataFrame:
    # retrieves and caches tabular 72-hour forecast data
    forecasts = fetch_forecasts(region)
    return pd.DataFrame(forecasts)


@st.cache_data(ttl=300, show_spinner=False)
def get_forecast_figure(region: str, horizon_hours: int = 72) -> go.Figure:
    # generates and caches forecast plotly figure with responsive horizon slice
    df_fc = get_forecast_dataframe(region)
    widened = bool(load_snapshot(region).get("widened", False))
    ci_label = "90% Widened CQR CI" if widened else "80% CQR CI"
    if horizon_hours and horizon_hours < len(df_fc):
        df_fc = df_fc.head(horizon_hours).copy()

    fig = go.Figure()

    # confidence interval ribbon (p10 to p90)
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="p90 Upper Bound",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p10"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.08)",
        name=ci_label,
        hoverinfo="skip",
    ))

    # median forecast line
    dot_size = 5 if horizon_hours <= 24 else (4 if horizon_hours <= 48 else 3)
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p50"],
        mode="lines+markers",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=dot_size, color="#2563eb"),
        name="p50 Demand",
        hovertemplate="<b>%{x|%d %b %H:%M UTC}</b><br>Expected Demand: <b>%{y:,.0f} MW</b><extra></extra>",
    ))

    # bound lines for hover clarity
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p90"],
        mode="lines",
        line=dict(color="#334155", width=1, dash="dot"),
        name="p90 Upper",
        hovertemplate="Upper Limit (p90): %{y:,.0f} MW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_fc["target_timestamp_utc"],
        y=df_fc["forecast_p10"],
        mode="lines",
        line=dict(color="#64748b", width=1, dash="dot"),
        name="p10 Lower",
        hovertemplate="Lower Limit (p10): %{y:,.0f} MW<extra></extra>",
    ))

    # peak and baseload callouts for active horizon
    peak_idx = df_fc["forecast_p50"].idxmax()
    peak_val = df_fc.loc[peak_idx, "forecast_p50"]
    peak_time = df_fc.loc[peak_idx, "target_timestamp_utc"]

    min_idx = df_fc["forecast_p50"].idxmin()
    min_val = df_fc.loc[min_idx, "forecast_p50"]
    min_time = df_fc.loc[min_idx, "target_timestamp_utc"]

    fig.add_annotation(
        x=peak_time,
        y=peak_val,
        text=f"Peak: {peak_val:,.0f} MW",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#ef4444",
        ax=0,
        ay=-28,
        font=dict(size=10.5, color="#991b1b", family="sans-serif"),
        bgcolor="rgba(255, 255, 255, 0.92)",
        bordercolor="#fecaca",
        borderwidth=1,
        borderpad=2,
    )
    fig.add_annotation(
        x=min_time,
        y=min_val,
        text=f"Baseload: {min_val:,.0f} MW",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#2563eb",
        ax=0,
        ay=28,
        font=dict(size=10.5, color="#1e40af", family="sans-serif"),
        bgcolor="rgba(255, 255, 255, 0.92)",
        bordercolor="#bfdbfe",
        borderwidth=1,
        borderpad=2,
    )

    if horizon_hours <= 24:
        dtick_val = 4 * 3600 * 1000
        tick_format = "%H:%M<br>%d %b"
    elif horizon_hours <= 48:
        dtick_val = 8 * 3600 * 1000
        tick_format = "%H:%M<br>%d %b"
    else:
        dtick_val = 12 * 3600 * 1000
        tick_format = "%d %b<br>%H:%M"

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=28, r=18, t=25, b=35),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            font_size=11,
            font_color="#ffffff",
            font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            bordercolor="#334155",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=9.5, color="#475569"),
        ),
        xaxis=dict(
            title="Forecast Target Horizon (UTC)",
            title_font=dict(size=10.5, color="#475569"),
            tickfont=dict(size=9.5, color="#64748b"),
            dtick=dtick_val,
            tickformat=tick_format,
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
        ),
        yaxis=dict(
            title="Active Load (MW)",
            title_font=dict(size=10.5, color="#475569"),
            tickfont=dict(size=9.5, color="#64748b"),
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            tickformat=",.0f",
        ),
        height=400,
    )
    return fig


SHAP_SHORT_LABELS = {
    "demand_mw": "Active Demand",
    "demand_lag_1h": "Demand (1h Lag)",
    "demand_lag_2h": "Demand (2h Lag)",
    "demand_lag_3h": "Demand (3h Lag)",
    "demand_lag_24h": "Demand (24h Lag)",
    "demand_lag_48h": "Demand (48h Lag)",
    "demand_lag_168h": "Demand (168h Lag)",
    "demand_rolling_mean_6h": "Mean Demand (6h)",
    "demand_rolling_mean_24h": "Mean Demand (24h)",
    "demand_rolling_mean_168h": "Mean Baseline (7d)",
    "demand_rolling_std_6h": "Volatility (6h)",
    "demand_rolling_std_24h": "Volatility (24h)",
    "demand_rolling_std_168h": "Volatility (7d)",
    "temperature_2m": "Temperature (2m)",
    "relative_humidity_2m": "Relative Humidity",
    "surface_solar_radiation": "Solar Radiation",
    "wind_speed_100m": "Wind Speed (100m)",
    "temp_delta_24h": "Temp Shift (24h)",
    "rh_delta_24h": "Humidity Shift (24h)",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "day_of_year": "Day of Year",
    "is_holiday": "Holiday (IST)",
    "sin_hour_k1": "Diurnal Sin (24h)",
    "cos_hour_k1": "Diurnal Cos (24h)",
    "sin_hour_k2": "Semi-Diurnal Sin",
    "cos_hour_k2": "Semi-Diurnal Cos",
    "sin_doy": "Annual Sin",
    "cos_doy": "Annual Cos",
}


@st.cache_data(show_spinner=False)
def build_shap_figure(top_features: List[Dict[str, Any]]) -> go.Figure:
    # constructs horizontal shap attribution bar chart with color-coded drivers
    df_shap = pd.DataFrame(top_features)
    feature_names = [
        SHAP_SHORT_LABELS.get(f, FEATURE_METADATA.get(f, {}).get("name", f))
        for f in df_shap["feature"]
    ]
    fig_shap = go.Figure(go.Bar(
        x=df_shap["shap_value"],
        y=feature_names,
        orientation="h",
        marker=dict(
            color=["#dc2626" if v > 0 else "#2563eb" for v in df_shap["shap_value"]],
            line=dict(width=0),
        ),
        text=[f"{v:+,.1f} MW" for v in df_shap["shap_value"]],
        textposition="auto",
        insidetextanchor="middle",
        textfont=dict(size=10, color="#ffffff"),
        hovertemplate="<b>%{y}</b><br>AI Load Impact: <b>%{x:+,.1f} MW</b><extra></extra>",
        cliponaxis=False,
    ))
    fig_shap.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=120, r=25, t=10, b=35),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            font_size=11,
            font_color="#ffffff",
            font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            bordercolor="#334155",
        ),
        xaxis=dict(
            title="Shapley Value (MW Impact)",
            title_font=dict(size=10.5, color="#475569"),
            tickfont=dict(size=9.5, color="#64748b"),
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=10, color="#334155"),
            linecolor="#e2e8f0",
        ),
        height=240,
    )
    return fig_shap


@st.cache_data(ttl=300, show_spinner=False)
def get_drift_metrics_dataframe(region: str) -> pd.DataFrame:
    # converts drift feature metrics into cached formatted audit dataframe
    drift_data = fetch_drift_summary(region)
    metrics_records = []
    for f_name, m in drift_data["feature_metrics"].items():
        meta = FEATURE_METADATA.get(f_name, {"name": f_name, "desc": "Telemetry parameter"})
        friendly_title = meta["name"]
        friendly_desc = meta["desc"]
        t_code = m.get("test", "")
        if t_code == "wasserstein_ks":
            test_type = "Norm. Wasserstein + KS"
        elif t_code == "kolmogorov_smirnov":
            test_type = "Kolmogorov-Smirnov"
        else:
            test_type = "1D Wasserstein Distance"
        stat_val = f"{m.get('statistic', m.get('score', 0.0)):.4f}"
        p_or_thresh = f"{m.get('p_value'):.6f}" if "p_value" in m else "N/A (Wasserstein)"
        thresh_val = f"{m['threshold']:.2f}"
        status_txt = "SHIFT DETECTED" if m["drift_detected"] else "STABLE (NOMINAL)"
        metrics_records.append({
            "Parameter": f"{friendly_title} ({f_name})",
            "Description": friendly_desc,
            "Statistical Audit Test": test_type,
            "Observed Metric": stat_val,
            "P-Value / Score": p_or_thresh,
            "Threshold": thresh_val,
            "Status": status_txt,
        })
    return pd.DataFrame(metrics_records)


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_drift_report(region: str) -> str:
    # caches generated standalone html report per region
    snap = load_snapshot(region)
    drift_data = snap["drift_summary"]
    return build_html_drift_report(drift_data, region=region, generated_at=snap.get("generated_at"))


# sidebar controls initialized first to establish layout and dispatch state
with st.sidebar:
    st.markdown(
        """
        <div style="font-weight: 700; font-size: 14.5px; color: #0f172a; margin-bottom: 8px;">
            Grid Dispatch Controls
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_region = st.selectbox(
        "Select Regional Grid",
        options=REGIONS,
        format_func=lambda code: REGION_NAMES.get(code, code),
        index=0,
        help="Choose one of India's 5 Regional Load Despatch Centres (RLDCs) or aggregate ALL_INDIA",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-weight: 700; font-size: 14.5px; color: #0f172a; margin-bottom: 8px;">
            System Operations
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="font-size: 12.5px; line-height: 1.6; color: #334155;">
            <div><b>Engine Version:</b> v{MODEL_VERSION}</div>
            <div><b>Mode:</b> Weather-Elastic IST Simulation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 12px; line-height: 1.6; color: #475569;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">RLDC Coverage</div>
            <div>• <b>NR</b>: Northern Region (Delhi, Punjab, UP)</div>
            <div>• <b>WR</b>: Western Region (Maharashtra, Gujarat)</div>
            <div>• <b>SR</b>: Southern Region (Karnataka, TN, AP)</div>
            <div>• <b>ER</b>: Eastern Region (WB, Odisha, Bihar)</div>
            <div>• <b>NER</b>: North Eastern Region (Assam, Meghalaya)</div>
            <div>• <b>ALL_INDIA</b>: National Grid Aggregate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# resolve active regional telemetry snapshot
active_snapshot = load_snapshot(selected_region)
df_fc = get_forecast_dataframe(selected_region)
nom_state = active_snapshot["nominal_anomaly"]
drift_state = active_snapshot["drift_summary"]
snap_src = active_snapshot.get("snapshot_source", "local_snapshot")
snap_src_label = (
    "Live Supabase Telemetry Cache"
    if snap_src == "supabase_live"
    else ("Bundled Snapshot" if snap_src == "local_snapshot" else "Local Runtime Fallback")
)
gen_ts_raw = active_snapshot.get("generated_at", "")
try:
    gen_dt = pd.to_datetime(gen_ts_raw, utc=True)
    gen_str = gen_dt.strftime("%d %b %Y %H:%M UTC")
except Exception:
    gen_str = str(gen_ts_raw)[:19]


# header and executive navigation
st.title("National Power Grid Telemetry & Forecasting Engine")
st.markdown(
    f"<div style='font-size: 13.5px; color: #475569; margin-bottom: 8px;'>"
    f"Multi-horizon conformal quantile forecasting, observation-aware fault detection, and statistical drift monitoring across India's 5 Regional Load Despatch Centres."
    f"</div>"
    f"<div style='font-size: 12px; color: #475569; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 12px; margin-bottom: 14px;'>"
    f"<b>Simulation Disclosure:</b> Powered by live Open-Meteo meteorological feeds driving a weather-elastic POSOCO-calibrated IST demand model (non-SCADA). "
    f"&nbsp;·&nbsp; <b>Snapshot Source:</b> {snap_src_label} ({gen_str})"
    f"</div>",
    unsafe_allow_html=True,
)

# dynamic executive control room kpi banner driven by active region & telemetry snapshot
peak_mw = float(df_fc["forecast_p50"].max()) if not df_fc.empty else 0.0
nom_alert = bool(nom_state.get("is_structural", False))
sentinel_text = "Alert Active" if nom_alert else "Nominal"
sentinel_color = "#dc2626" if nom_alert else "#047857"
drift_alarm_flag = bool(drift_state.get("drift_alarm", False))
watchdog_text = "Retrain Advisory" if drift_alarm_flag else "Optimal"
watchdog_color = "#dc2626" if drift_alarm_flag else "#047857"
stability_pct = (1.0 - float(drift_state.get("drift_ratio", 0.0))) * 100.0
widened_flag = bool(active_snapshot.get("widened", False))

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{selected_region} Projected Peak</div>
            <div class="kpi-value">{peak_mw:,.0f} MW</div>
            <div class="kpi-caption">POSOCO-Calibrated Simulation Scale</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Forward Load Visibility</div>
            <div class="kpi-value">72 Hours</div>
            <div class="kpi-caption">{"Widened 90% CQR Envelope" if widened_flag else "Conformal 80% CQR Envelope"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Grid Fault Sentinel <span class="info-tip tip-right" tabindex="0" data-tip="Dual-algorithm safety filter combining an observation-aware Isolation Forest and empirical 1.5xIQR boundary to isolate physical faults from weather surges.">?</span></div>
            <div class="kpi-value" style="color: {sentinel_color};">{sentinel_text}</div>
            <div class="kpi-caption">Dual-Trigger Safety Guard Active</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Model Health Watchdog <span class="info-tip tip-right" tabindex="0" data-tip="Automated MLOps watchdog checking trailing 7-day telemetry covariates against 30-day baseline distributions.">?</span></div>
            <div class="kpi-value" style="color: {watchdog_color};">{watchdog_text}</div>
            <div class="kpi-caption">{stability_pct:.1f}% Distribution Stability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# tabs for main interface
tab_forecast, tab_anomalies, tab_drift, tab_system = st.tabs([
    "72-Hour Demand Forecast",
    "Grid Fault & Surge Sentinel",
    "Model Health & Reliability Watchdog",
    "Info for Nerds",
])

# tab 1: 72-hour forecast
with tab_forecast:
    st.markdown(f"## 72-Hour Forward Demand Forecast: {REGION_NAMES.get(selected_region, selected_region)}")
    st.markdown(
        """
        <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">
            Conformalized forward electrical demand projections across 72 hourly steps with monotonic confidence bounds.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Expected Peak Demand (p50)</div>
                <div class="metric-value">{df_fc['forecast_p50'].max():,.0f} MW</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Max projected load requiring generation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Expected Baseload (p50)</div>
                <div class="metric-value">{df_fc['forecast_p50'].min():,.0f} MW</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Minimum overnight baseload demand</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        avg_band = (df_fc['forecast_p90'] - df_fc['forecast_p10']).mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Reserve Margin (p90 - p10) <span class="info-tip tip-right" tabindex="0" data-tip="Conformalized spread between p90 stress limit and p10 conservative estimate. Dictates required standby spinning reserve.">?</span></div>
                <div class="metric-value">{avg_band:,.0f} MW</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Mean CQR uncertainty spread for spinning reserve</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Confidence Reliability <span class="info-tip tip-right" tabindex="0" data-tip="Mathematical validation confirming p10 <= p50 <= p90 across all 72 time steps, guaranteeing the uncertainty envelope never inverts.">?</span></div>
                <div class="metric-value" style="color: #047857;">100% Monotonic</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Guaranteed: p10 (Low) ≤ p50 ≤ p90 (Peak)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # interactive horizon selector and mobile ergonomics
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 13.5px; color: #1e293b; margin-top: 6px; margin-bottom: 2px;">
            Forecast Resolution Window
            <span class="info-tip" tabindex="0" data-tip="Switch between Next 24h, 48h, or Full 72h views. The 24h view provides 3x higher hourly resolution on mobile screens.">?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    horizon_label = st.radio(
        "Forecast Window Selection",
        options=["Next 24h (Detail)", "Next 48h", "Full 72h"],
        index=0,
        horizontal=True,
        key=f"horizon_window_{selected_region}",
        label_visibility="collapsed",
    )

    horizon_map = {
        "Next 24h (Detail)": 24,
        "Next 48h": 48,
        "Full 72h": 72,
    }
    selected_horizon = horizon_map.get(horizon_label, 72)

    st.markdown(
        """
        <div class="chart-mobile-hint">
            <span style="font-weight: 600;">Mobile Tip:</span> 24h view gives high-resolution hourly clarity. Rotate phone to landscape for widescreen inspection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = get_forecast_figure(selected_region, horizon_hours=selected_horizon)
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "responsive": True,
            "scrollZoom": False,
        },
    )

    with st.expander(f"Tabular Forecast Data ({selected_horizon} Hourly Steps)"):
        st.dataframe(df_fc.head(selected_horizon), width="stretch")

# tab 2: grid fault & surge sentinel
with tab_anomalies:
    st.markdown(f"## Real-Time Grid Fault & Surge Sentinel: {REGION_NAMES.get(selected_region, selected_region)}")
    st.markdown(
        """
        <div style="font-size: 13px; color: #444444; margin-bottom: 12px;">
            Dual-trigger safety guard cross-checking observation-aware anomaly isolation with physical boundary envelopes to eliminate false alarms.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # interactive test parameter injection
    col_input, col_status = st.columns([1, 2])
    with col_input:
        st.markdown(
            """
            <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">
                Simulate Grid Stress-Testing
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Select a scenario to evaluate how the AI classifies incoming load:")
        anomaly_scenario = st.radio(
            "Evaluation Scenario:",
            [
                "Normal Grid Operations (Standard Baseline)",
                "Simulate Severe Grid Shock (Extreme Load Surge)",
                "Custom Demand Stress-Test (Manual Input)",
            ],
            index=0,
            key=f"scenario_{selected_region}",
        )
        custom_val = None
        if anomaly_scenario == "Simulate Severe Grid Shock (Extreme Load Surge)":
            nom_p50 = float(nom_state.get("predicted_p50_mw", 50000.0))
            custom_val = round(nom_p50 * 2.0 + 15000.0, 0)
        elif anomaly_scenario == "Custom Demand Stress-Test (Manual Input)":
            default_custom = float(round(nom_state.get("predicted_p50_mw", 50000.0) * 1.15, 0))
            custom_val = st.number_input("Injected Demand (MW):", min_value=100.0, max_value=500000.0, value=default_custom)

    with col_status:
        st.markdown(
            """
            <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">
                Live Sentinel Classification Status
            </div>
            """,
            unsafe_allow_html=True,
        )
        anomaly_res = evaluate_live_anomaly(selected_region, custom_demand=custom_val)
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Observed vs Expected (p50)</div>
                    <div class="metric-value">{anomaly_res['actual_demand_mw']:,.0f} / {anomaly_res['predicted_p50_mw']:,.0f} MW</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Live Reading vs AI Median Projection</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with status_col2:
            trig_b_text = "Envelope Breached" if anomaly_res["residual_breach"] else "Within Safe Envelope"
            trig_b_color = "#dc2626" if anomaly_res["residual_breach"] else "#047857"
            trig_b_desc = "Exceeds physical 1.5×IQR boundary" if anomaly_res["residual_breach"] else "Operating within normal variance"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Operating Envelope Check <span class="info-tip" tabindex="0" data-tip="Evaluates whether the observed residual breaches the conformal 1.5xIQR safety boundary.">?</span></div>
                    <div class="metric-value" style="color: {trig_b_color};">{trig_b_text}</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{trig_b_desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with status_col3:
            struct_text = "CRITICAL GRID FAULT" if anomaly_res["is_structural"] else "Normal Operation"
            struct_color = "#dc2626" if anomaly_res["is_structural"] else "#047857"
            struct_desc = "Dual-Trigger Confirmed: Alert Active" if anomaly_res["is_structural"] else "No Action Required (Nominal)"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Actionable Sentinel Alert <span class="info-tip tip-right" tabindex="0" data-tip="Dual-trigger consensus: Emergency alert triggers ONLY when the observation-aware Isolation Forest flags structural surprise AND load breaches the operating envelope.">?</span></div>
                    <div class="metric-value" style="color: {struct_color};">{struct_text}</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{struct_desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Root-Cause Diagnostic: Why Did the AI Make This Assessment? (SHAP Explainability)")
    st.markdown(
        """
        <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">
            Path-dependent TreeExplainer Shapley values attribute exact Megawatt load contributions to physical drivers.
            <span class="info-tip tip-right" tabindex="0" data-tip="Grid operators cannot trust opaque black boxes during an emergency. Exact TreeExplainer Shapley values decompose each forecast into per-feature MW impacts. Red values push demand projections higher; blue values pull projections lower.">?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_features = anomaly_res["top_shap_features"]
    shap_cols = st.columns(len(top_features))
    for idx, (col_ui, feat) in enumerate(zip(shap_cols, top_features)):
        f_meta = FEATURE_METADATA.get(feat["feature"], {"name": feat["feature"], "desc": "Telemetry variable"})
        friendly_title = f_meta["name"]
        friendly_desc = f_meta["desc"]
        direction_color = "#dc2626" if feat["shap_value"] > 0 else "#2563eb"
        direction_label = "Pushes projection by" if feat["shap_value"] > 0 else "Lowers projection by"

        with col_ui:
            st.markdown(
                f"""
                <div class="metric-card" style="min-height: 155px;">
                    <div class="metric-label">Driver #{idx + 1}: {friendly_title}</div>
                    <div style="font-family: monospace; font-size: 11px; color: #64748b; margin-bottom: 4px;"><code>{feat['feature']}</code></div>
                    <div class="metric-value" style="color: {direction_color};">
                        {feat['shap_value']:+,.1f} MW
                    </div>
                    <div style="font-size: 11.5px; color: #334155; margin-top: 4px;">
                        {direction_label} <b>{abs(feat['shap_value']):,.1f} MW</b>
                    </div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 6px; border-top: 1px solid #f1f5f9; padding-top: 4px;">
                        Observed Value: <b>{feat['actual_value']:,.2f}</b><br>
                        <i>{friendly_desc}</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="chart-mobile-hint">
            <span style="font-weight: 600;">Mobile Tip:</span> Tap any bar for exact Megawatt impact. Rotate device to landscape for widescreen inspection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_shap = build_shap_figure(top_features)
    st.plotly_chart(
        fig_shap,
        width="stretch",
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "responsive": True,
            "scrollZoom": False,
        },
    )

    with st.expander("Tabular Feature Attribution Breakdown"):
        df_display = pd.DataFrame([
            {
                "Driver Rank": f"#{i+1}",
                "Parameter": FEATURE_METADATA.get(f["feature"], {}).get("name", f["feature"]),
                "Variable Code": f["feature"],
                "AI Impact (MW)": f"{f['shap_value']:+,.1f} MW",
                "Observed Value": f"{f['actual_value']:,.2f}",
                "Influence": "Pushes load higher" if f["shap_value"] > 0 else "Pulls load lower",
            }
            for i, f in enumerate(top_features)
        ])
        st.dataframe(df_display, width="stretch")

# tab 3: model health & reliability watchdog
with tab_drift:
    st.markdown(f"## Continuous Model Health & Reliability Watchdog: {REGION_NAMES.get(selected_region, selected_region)}")
    st.markdown(
        """
        <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">
            Continuous data distribution audit comparing trailing 7-day telemetry against the 30-day baseline across 12 physical and diurnal covariates.
        </div>
        """,
        unsafe_allow_html=True,
    )

    drift_data = fetch_drift_summary(selected_region)
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        alarm_text = "Recalibration Recommended" if drift_data["drift_alarm"] else "Healthy & Reliable"
        alarm_color = "#dc2626" if drift_data["drift_alarm"] else "#047857"
        alarm_caption = "Data pattern shift exceeds safe threshold" if drift_data["drift_alarm"] else "Telemetry matches baseline distributions"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall AI Health</div>
                <div class="metric-value" style="color: {alarm_color};">{alarm_text}</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{alarm_caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col2:
        stability_pct_tab = (1.0 - float(drift_data["drift_ratio"])) * 100.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Distribution Stability</div>
                <div class="metric-value">{stability_pct_tab:.1f}% Stable</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{drift_data['drifted_features']} of {drift_data['total_features']} features shifted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Watchdog Tolerance <span class="info-tip tip-right" tabindex="0" data-tip="Operational alarm threshold: if more than 30% of monitored features show statistical drift, the watchdog flags a data shift and widens forecast uncertainty bounds.">?</span></div>
                <div class="metric-value">30.0% Max Shift</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Alert triggers if &gt; 30% of features shift</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col4:
        alphas_val = drift_data["recommended_quantile_alpha"]
        alphas_display = str(alphas_val)
        model_action_desc = "Standard 80% Confidence Interval" if not drift_data["drift_alarm"] else "Widened 90% Confidence Interval"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recommended Model Action <span class="info-tip tip-right" tabindex="0" data-tip="Dynamic uncertainty adaptation: Under normal conditions, uses conformal 80% [p10, p50, p90] bounds. Under high drift, expands to 90% [p05, p50, p95] bounds.">?</span></div>
                <div class="metric-value">{alphas_display}</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{model_action_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">
            Feature-by-Feature Distribution Audit (Trailing 7-Day Telemetry vs 30-Day Baseline) <span class="info-tip" tabindex="0" data-tip="Normalized 1D Wasserstein distance (W1/std >= 0.50 gated by KS p < 0.01) for continuous physical metrics and 1D Wasserstein distance (W1 >= 0.15) for cyclical harmonics.">?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Continuous physical metrics use normalized Wasserstein + Kolmogorov-Smirnov gating (W1/std >= 0.50 and p < 0.01); cyclical harmonics use 1D Wasserstein distance.")
    df_metrics = get_drift_metrics_dataframe(selected_region)
    st.dataframe(df_metrics, width="stretch")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">
            Export Executive Reliability Audit Report
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Download a self-contained HTML audit report containing complete normalized Wasserstein and Kolmogorov-Smirnov metrics formatted for engineering and compliance review.")
    report_html = get_cached_drift_report(selected_region)
    st.download_button(
        label=f"Download Executive Audit Report ({selected_region})",
        data=report_html,
        file_name=f"drift_report_{selected_region.lower()}.html",
        mime="text/html",
    )

# tab 4: info for nerds
with tab_system:
    st.markdown("## Info for Nerds")
    st.caption("How the engine works under the hood — from live Open-Meteo weather and causal Polars features to 72-hour conformal forecasting and reliability monitoring.")

    # 1. Project Overview & The Physical Problem
    st.markdown("### The Problem: Why Power Grids Require Probabilistic ML")
    st.markdown(
        """
        Unlike software systems where compute and network requests can buffer or queue, an electrical power grid has
        virtually zero storage at utility scale. Every single Megawatt (MW) consumed across homes, railways, and factories
        must be generated and dispatched in real time.

        - **If generation lags demand:** System frequency drops below the 50.0 Hz nominal target, risking under-frequency load shedding or generator tripping.
        - **If generation exceeds demand:** System frequency surges above safe tolerances, stressing transformers and turbine governors.
        - **The Dispatcher Dilemma:** Grid operators at Regional Load Despatch Centres (RLDCs) must commit thermal units hours in advance. Probabilistic quantile envelopes (`p10`, `p50`, `p90`) quantify both expected baseload dispatch and spinning-reserve headroom.

        This engine models India's national and regional electrical demand (`ALL_INDIA` plus `NR`, `WR`, `SR`, `ER`, `NER`) calibrated to POSOCO/Grid-India operational scale using live Open-Meteo meteorological telemetry across an automated 6-stage pipeline.
        """
    )

    st.markdown("---")

    # Step 1
    st.markdown("### Step 1: Automated Ingestion & Physical Boundary Validation")
    st.markdown(
        """
        **How Data Enters the Engine:**
        Every 6 hours, an automated GitHub Actions cron workflow (`15 */6 * * *`) executes ingestion and batch inference:
        - **Exogenous Meteorological Feed:** Pulls live hourly ambient `temperature_2m`, `relative_humidity_2m`, `shortwave_radiation_instant`, and `wind_speed_100m` from the Open-Meteo Forecast API across regional grid centroids.
        - **Weather-Elastic Grid Simulation:** Evaluates an IST-aligned diurnal load profile calibrated to POSOCO Dec 2024 regional baselines with empirical cooling/heating degree elasticity, solar behind-the-meter offset, and Indian statutory holiday calendars (explicitly non-SCADA simulation).

        **At-the-Door Data Hygiene:**
        Inbound payloads pass through strict **Pydantic v2** validation contracts before reaching downstream tables:
        - Active power is enforced to be non-negative and bounded within physical regional envelopes.
        - Grid frequency is validated within standard grid limits (`47.5 - 52.5 Hz`).
        - Timestamps normalize strictly to UTC, and missing hourly packets (`<= 6h`) are repaired before idempotent upsert into PostgreSQL (`Supabase`, `us-west-1` with TLS and Row-Level Security).
        """
    )

    st.markdown("---")

    # Step 2
    st.markdown("### Step 2: Causal Feature Engineering (Zero-Leakage Polars Pipeline)")
    st.markdown(
        """
        **Preventing Lookahead & Gap-Shift Leakage:**
        In time-series ML, missing rows cause `.shift(k)` to silently alias older observations onto shorter lags, while computing rolling windows without shifting leaks current-step demand into features.

        **How the Pipeline Guarantees Strict Causality:**
        - **Explicit 1h Hourly Upsampling:** Input batches are sorted per region and upsampled to a strict `1h` grid (`df.upsample(every="1h")`) prior to computing lags (`1h, 2h, 3h, 24h, 48h, 168h`), rolling statistics, and `24h` weather differentials (`temp_delta_24h`, `rh_delta_24h`), then filtered back to observed rows.
        - **Mandatory `.shift(1)` Temporal Fence:** Rolling means and standard deviations (`6h, 24h, 168h`) apply a strict 1-hour lag prior to aggregation.
        - **IST Holiday & Harmonic Encodings:** Holidays resolve in `Asia/Kolkata` local dates via `holidays.India`, paired with Fourier diurnal and annual harmonics across **26 LightGBM features**:
        """
    )
    st.latex(r"\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right),\quad \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right),\quad \sin\left(\frac{2\pi \cdot \text{doy}}{365.25}\right),\quad \cos\left(\frac{2\pi \cdot \text{doy}}{365.25}\right)")

    st.markdown("---")

    # Step 3
    st.markdown("### Step 3: Conformal Multi-Horizon Forecasting (LightGBM Quantiles)")
    st.markdown(
        """
        **26-Feature Quantile Ensembles + Conformal Calibration (`CQR`):**
        We train 3 independent LightGBM quantile regressors (`p10`, `p50`, `p90`) per region across 18 weather-aware boosters, then calibrate raw intervals on held-out calibration residuals using symmetric Conformal Quantile Regression (`cqr_margin_80` / `cqr_margin_90` in `quantile_calibration.json`), followed by isotonic monotonicity sorting:
        """
    )
    st.latex(r"p_{10}(t) \le p_{50}(t) \le p_{90}(t) \quad \forall t \in [1, 72]")
    st.markdown(
        """
        **Iterative 72-Hour Exogenous Trajectory:**
        At inference time, each recursive step incorporates live Open-Meteo `forecast_days` exogenous weather predictions alongside updated autoregressive load lags and rolling windows.
        """
    )

    st.markdown("---")

    # Step 4
    st.markdown("### Step 4: Observation-Aware Dual-Trigger Fault Sentinel")
    st.markdown(
        """
        **Eliminating Alert Fatigue via Independent Dual Consensus:**
        Instead of evaluating static historical lags alone, the sentinel builds a dedicated **9-feature live observation matrix** (`demand_mw`, instantaneous `demand_delta_1h`, `demand_delta_24h`, `demand_zscore_24h`, ambient weather, and diurnal harmonics):
        - **Trigger A (Observation-Aware Isolation Forest):** Flags multivariate structural surprise (`predict == -1`) when live demand decouples sharply from weather and recent momentum.
        - **Trigger B (Conformal Residual Envelope):** Checks whether live demand breaches the calibrated physical safety band:
        """
    )
    st.latex(r"\text{Operating Boundary} = \Big[\max(0,\; p_{10} - 1.5 \cdot \text{IQR}),\; p_{90} + 1.5 \cdot \text{IQR}\Big] \quad \text{where} \quad \text{IQR} = p_{90} - p_{10}")
    st.markdown(
        """
        - **Consensus Gate (`Alert = Trigger_A AND Trigger_B`):** Normal weather-driven load shifts stay inside healthy consensus, while sudden physical surges fire both gates simultaneously.
        """
    )

    st.markdown("---")

    # Step 5
    st.markdown("### Step 5: Path-Dependent SHAP & Statistical Drift Watchdog")
    st.markdown(
        """
        **Exact SHAP TreeExplainer Attribution:**
        Using path-dependent `shap.TreeExplainer(booster)`, every flagged point is decomposed into exact Megawatt attributions across all 26 physical and calendar features without external centroid approximations.

        **Normalized Covariate Drift Monitoring:**
        Every 6 hours, trailing 7-day telemetry covariates are audited against each region's 30-day reference distribution across 12 physical and diurnal variables:
        - **Continuous Physical Features:** Normalized 1D Wasserstein distance (`W1 / std >= 0.50`) gated by two-sample Kolmogorov-Smirnov significance (`p < 0.01`).
        - **Cyclical & Calendar Indicators:** 1D Wasserstein distance (`W1 >= 0.15`).
        - **Dynamic Reserve Widening:** When drifted covariates exceed `30%`, `drift_alarm` engages and forecast generation automatically switches from standard `80%` conformal intervals to widened `90%` (`cqr_margin_90`) reserve bounds.
        """
    )

    st.markdown("---")

    # Step 6
    st.markdown("### Step 6: Production Serving & Persistence Architecture")
    st.markdown(
        """
        **Single-Process ASGI + Live Snapshot Architecture:**
        - **ASGI Wrapper (`src/server.py` + `st.App`):** Runs in a single container process on Render (`us-west1` alongside Supabase `us-west-1`), injecting strict HTTP security headers (`HSTS`, `CSP`, `X-Frame-Options`, `nosniff`), server-side SEO `<meta>` tags, HTTP `404` status handling, `/health`, and `/reports/drift_report.html`.
        - **Scheduled Supabase Pipeline:** GitHub Actions runs ingestion, dual-trigger anomaly scoring, vintage-tracked `model_forecasts` persistence, and `precomputed_snapshots` JSON/HTML updates every 6 hours, which the Streamlit UI reads directly (300s TTL cache) with bundled fallback.
        - **Repository Documentation:** Explore the full system specification and optional developer API reference in [README.md](https://github.com/mahesh-kakde/grid-telemetry-engine#readme).
        """
    )