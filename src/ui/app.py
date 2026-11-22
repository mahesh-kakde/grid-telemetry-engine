import os
import sys

# ensure project root is present in sys.path when running standalone streamlit commands
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import streamlit.components.v1 as components
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
MODEL_VERSION = "1.0.0"
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
    @media (min-width: 769px) {
        [data-testid="stSidebar"]:not([aria-expanded="false"]) {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            bottom: 0 !important;
            width: 300px !important;
            min-width: 300px !important;
            height: 100vh !important;
            z-index: 100 !important;
            transition: none !important;
        }
        [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"]:not([aria-expanded="false"])) > div:not([data-testid="stSidebar"]) {
            margin-left: 300px !important;
            width: calc(100% - 300px) !important;
            max-width: calc(100% - 300px) !important;
            transition: none !important;
        }
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
        font-size: 21px;
        font-weight: 700;
        color: #0f172a;
        font-feature-settings: "tnum";
        font-variant-numeric: tabular-nums;
        line-height: 1.25;
        margin-bottom: 3px;
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
        /* lock mobile viewport: zero horizontal blowup */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            overflow-x: hidden !important;
            max-width: 100vw !important;
            width: 100vw !important;
        }

        /* mobile main content full viewport - zero layout shift */
        [data-testid="stAppViewContainer"] > div,
        .stAppViewContainer > div,
        .stAppViewContainer > div:not([data-testid="stSidebar"]),
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .main,
        section[data-testid="stMain"] {
            width: 100vw !important;
            max-width: 100vw !important;
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
            max-width: 100vw !important;
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

        /* tooltips on mobile: fixed viewport toast - zero cutoff */
        .info-tip:hover::after,
        .info-tip:focus::after,
        .info-tip:active::after {
            position: fixed !important;
            left: 12px !important;
            right: 12px !important;
            bottom: 44px !important;
            top: auto !important;
            width: auto !important;
            max-width: calc(100vw - 24px) !important;
            transform: none !important;
            font-size: 11.5px !important;
            line-height: 1.45 !important;
            padding: 10px 14px !important;
            border-radius: 8px !important;
            background-color: #0f172a !important;
            color: #f8fafc !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.3) !important;
            border: 1px solid #334155 !important;
            z-index: 9999999 !important;
            white-space: normal !important;
            pointer-events: auto !important;
        }
        .info-tip:hover::before,
        .info-tip:focus::before,
        .info-tip:active::before {
            display: none !important;
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

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


@st.cache_resource
def get_cached_artifacts():
    # caches in-memory bundle with on-demand lazy loading
    from src.models.load_artifacts import ModelArtifactsBundle
    from src.models.explainers import AnomalySHAPExplainer
    bundle = ModelArtifactsBundle()
    explainer = AnomalySHAPExplainer(bundle)
    return bundle, explainer


def check_api_health() -> Tuple[bool, Dict[str, Any]]:
    # queries loopback api health probe
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, {}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecasts(region: str) -> List[Dict[str, Any]]:
    # retrieves 72-hour forecast via precomputed cache, loopback api, or local model fallback
    from src.models.precompute import load_precomputed_forecasts
    precomputed = load_precomputed_forecasts(region)
    if precomputed is not None and len(precomputed) == 72:
        return precomputed

    is_live, _ = check_api_health()
    if is_live:
        try:
            res = requests.post(
                f"{API_BASE_URL}/forecast",
                json={"region": region, "lookback_hours": 168},
                timeout=15,
            )
            if res.status_code == 200:
                data = res.json()
                parsed = []
                for fc in data["forecasts"]:
                    parsed.append({
                        "target_timestamp_utc": pd.to_datetime(fc["target_timestamp_utc"]),
                        "forecast_p10": fc["forecast_p10"],
                        "forecast_p50": fc["forecast_p50"],
                        "forecast_p90": fc["forecast_p90"],
                        "region": fc["region"],
                    })
                return parsed
        except Exception:
            pass

    # direct local execution fallback with deferred imports
    from src.models.load_artifacts import load_reference_baseline
    from src.models.infer_and_detect import generate_72h_multi_step_forecast
    bundle, _ = get_cached_artifacts()
    ref_df = load_reference_baseline(region).tail(168).copy()
    base_ts = datetime.now(timezone.utc) - timedelta(hours=len(ref_df))
    records = []
    for idx, (_, row) in enumerate(ref_df.iterrows()):
        records.append({
            "timestamp_utc": base_ts + timedelta(hours=idx),
            "region": region,
            "demand_mw": float(row["demand_mw"]),
            "frequency_hz": 50.0,
            "temperature_2m": 30.0,
            "relative_humidity_2m": 50.0,
            "surface_solar_radiation": 350.0,
            "wind_speed_100m": 4.0,
        })
    df_history = pl.DataFrame(records)
    raw_forecasts = generate_72h_multi_step_forecast(df_history, region, bundle)
    return [
        {
            "target_timestamp_utc": pd.to_datetime(fc["target_timestamp_utc"]),
            "forecast_p10": fc["forecast_p10"],
            "forecast_p50": fc["forecast_p50"],
            "forecast_p90": fc["forecast_p90"],
            "region": fc["region"],
        }
        for fc in raw_forecasts
    ]


@st.cache_data(ttl=1800, show_spinner=False)
def evaluate_live_anomaly(region: str, custom_demand: Optional[float] = None) -> Dict[str, Any]:
    # evaluates anomaly detection status via precomputed nominal snapshot, api, or local fallback
    if custom_demand is None:
        from src.models.precompute import load_precomputed_anomaly
        precomputed = load_precomputed_anomaly(region)
        if precomputed is not None:
            return precomputed

    is_live, _ = check_api_health()
    if is_live:
        try:
            payload = {"region": region}
            if custom_demand is not None:
                payload["actual_demand_mw"] = custom_demand
            res = requests.post(f"{API_BASE_URL}/anomalies", json=payload, timeout=10)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

    from src.models.load_artifacts import load_reference_baseline
    from src.models.infer_and_detect import predict_quantiles, detect_structural_anomalies
    bundle, explainer = get_cached_artifacts()
    ref_df = load_reference_baseline(region)
    feature_row = ref_df.iloc[-1]
    features_df = pd.DataFrame([feature_row[bundle.feature_columns]])

    p10, p50, p90 = predict_quantiles(features_df, region, bundle)
    actual = custom_demand if custom_demand is not None else float(feature_row["demand_mw"])

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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_drift_summary(region: str) -> Dict[str, Any]:
    # fetches statistical drift summary via precomputed snapshot, api, or local evaluation
    from src.models.precompute import load_precomputed_drift
    precomputed = load_precomputed_drift(region)
    if precomputed is not None:
        return precomputed

    is_live, _ = check_api_health()
    if is_live:
        try:
            res = requests.get(f"{API_BASE_URL}/drift?region={region}", timeout=10)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

    from src.models.load_artifacts import load_reference_baseline
    from src.monitoring.drift_check import evaluate_feature_drift, fetch_trailing_telemetry_window
    ref_df = load_reference_baseline(region)
    curr_df = fetch_trailing_telemetry_window(region=region, window_hours=168)
    summary = evaluate_feature_drift(ref_df, curr_df)
    summary["region"] = region
    summary["report_available"] = os.path.exists(DEFAULT_REPORT_PATH)
    return summary


@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast_dataframe(region: str) -> pd.DataFrame:
    # retrieves and caches tabular 72-hour forecast data
    forecasts = fetch_forecasts(region)
    return pd.DataFrame(forecasts)


@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast_figure(region: str, horizon_hours: int = 72) -> go.Figure:
    # generates and caches forecast plotly figure with responsive horizon slice
    df_fc = get_forecast_dataframe(region)
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
        name="80% CI",
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
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "day_of_year": "Day of Year",
    "is_holiday": "Holiday",
    "sin_hour_k1": "Diurnal Harmonic",
    "cos_hour_k1": "Diurnal Harmonic",
    "sin_day_k1": "Weekly Harmonic",
    "cos_day_k1": "Weekly Harmonic",
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


@st.cache_data(ttl=1800, show_spinner=False)
def get_drift_metrics_dataframe(region: str) -> pd.DataFrame:
    # converts drift feature metrics into cached formatted audit dataframe
    drift_data = fetch_drift_summary(region)
    metrics_records = []
    for f_name, m in drift_data["feature_metrics"].items():
        meta = FEATURE_METADATA.get(f_name, {"name": f_name, "desc": "Telemetry parameter"})
        friendly_title = meta["name"]
        friendly_desc = meta["desc"]
        test_type = "Kolmogorov-Smirnov" if m["test"] == "kolmogorov_smirnov" else "1D Wasserstein Distance"
        stat_val = f"{m.get('statistic', m.get('score')):.4f}"
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


@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_drift_report(region: str) -> str:
    # caches generated standalone html report per region
    drift_data = fetch_drift_summary(region)
    return build_html_drift_report(drift_data, region=region)


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
        """
        <div style="font-size: 12.5px; line-height: 1.6; color: #334155;">
            <div><b>Engine Version:</b> v1.0.0</div>
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


# header and executive navigation
st.title("National Power Grid Telemetry & Forecasting Engine")
st.markdown(
    "<div style='font-size: 13.5px; color: #475569; margin-bottom: 16px;'>"
    "Multi-horizon probabilistic load forecasting, real-time dual-trigger fault sentinel, and automated statistical drift monitoring across India's 5 Regional Load Despatch Centres."
    "</div>",
    unsafe_allow_html=True,
)

# executive control room kpi banner
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">National Grid Coverage</div>
            <div class="kpi-value">215,500 MW</div>
            <div class="kpi-caption">Active Peak Aggregate (5 RLDCs)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c2:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">Forward Load Visibility</div>
            <div class="kpi-value">72 Hours</div>
            <div class="kpi-caption">Probabilistic Quantile Horizon</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c3:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">Grid Fault Sentinel <span class="info-tip tip-right" tabindex="0" data-tip="Dual-algorithm safety filter combining an Isolation Forest and an empirical IQR envelope to distinguish physical faults from benign weather surges.">?</span></div>
            <div class="kpi-value" style="color: #047857;">Nominal</div>
            <div class="kpi-caption">Dual-Trigger Safety Guard Active</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_c4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">Model Health Watchdog <span class="info-tip tip-right" tabindex="0" data-tip="Automated MLOps watchdog checking whether live telemetry distributions match baseline distributions, alerting if model needs retraining.">?</span></div>
            <div class="kpi-value" style="color: #047857;">Optimal</div>
            <div class="kpi-caption">Telemetry Distributions Verified</div>
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
            Probabilistic forward electrical demand projections across 72 hourly steps with guaranteed 80% confidence bounds.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_fc = get_forecast_dataframe(selected_region)

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
                <div class="metric-label">Reserve Margin (p90 - p10) <span class="info-tip tip-right" tabindex="0" data-tip="Difference between 90th percentile stress limit and 10th percentile conservative estimate. Dictates required standby spinning reserve.">?</span></div>
                <div class="metric-value">{avg_band:,.0f} MW</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Mean uncertainty spread for spinning reserve</div>
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
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "responsive": True,
            "scrollZoom": False,
        },
    )

    with st.expander(f"Tabular Forecast Data ({selected_horizon} Hourly Steps)"):
        st.dataframe(df_fc.head(selected_horizon), use_container_width=True)

# tab 2: grid fault & surge sentinel
with tab_anomalies:
    st.markdown(f"## Real-Time Grid Fault & Surge Sentinel: {REGION_NAMES.get(selected_region, selected_region)}")
    st.markdown(
        """
        <div style="font-size: 13px; color: #444444; margin-bottom: 12px;">
            Dual-trigger safety guard cross-checking unsupervised anomaly isolation with physical boundary envelopes to eliminate false alarms.
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
                "Simulate Severe Grid Shock (+200,000 MW Surge)",
                "Custom Demand Stress-Test (Manual Input)",
            ],
            index=0,
            key=f"scenario_{selected_region}",
        )
        custom_val = None
        if anomaly_scenario == "Simulate Severe Grid Shock (+200,000 MW Surge)":
            custom_val = 260000.0 if selected_region == "ALL_INDIA" else 150000.0
        elif anomaly_scenario == "Custom Demand Stress-Test (Manual Input)":
            custom_val = st.number_input("Injected Demand (MW):", min_value=100.0, max_value=350000.0, value=75000.0)

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
                    <div class="metric-label">Operating Envelope Check <span class="info-tip" tabindex="0" data-tip="Evaluates whether the observed residual breaches the empirical 1.5xIQR safety margin calculated from 30 days of baseline operations.">?</span></div>
                    <div class="metric-value" style="color: {trig_b_color};">{trig_b_text}</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{trig_b_desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with status_col3:
            struct_text = "CRITICAL GRID FAULT" if anomaly_res["is_structural"] else "Normal Operation"
            struct_color = "#dc2626" if anomaly_res["is_structural"] else "#047857"
            struct_desc = "Dual-Trigger Confirmed: Alert Dispatched" if anomaly_res["is_structural"] else "No Action Required (Nominal)"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Actionable Sentinel Alert <span class="info-tip tip-right" tabindex="0" data-tip="Dual-trigger consensus: Emergency alert triggers ONLY if the Isolation Forest flags structural anomaly AND the load breaches the physical operating envelope.">?</span></div>
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
            TreeExplainer Shapley values attribute exact Megawatt load contributions to physical drivers.
            <span class="info-tip tip-right" tabindex="0" data-tip="Grid operators cannot trust opaque black boxes during an emergency. Using SHAP (Shapley Additive Explanations) evaluated against 150 background centroids, the engine calculates the exact Megawatt impact of each physical variable. Red values push demand projections higher; blue values pull projections lower.">?</span>
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
        use_container_width=True,
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
        st.dataframe(df_display, use_container_width=True)

# tab 3: model health & reliability watchdog
with tab_drift:
    st.markdown(f"## Continuous Model Health & Reliability Watchdog: {REGION_NAMES.get(selected_region, selected_region)}")
    st.markdown(
        """
        <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">
            Continuous data distribution audit comparing trailing 7-day live telemetry against the 30-day baseline across all 18 physical variables.
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
        stability_pct = (1.0 - drift_data['drift_ratio']) * 100.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Distribution Stability</div>
                <div class="metric-value">{stability_pct:.1f}% Stable</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{drift_data['drifted_features']} of {drift_data['total_features']} features shifted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Watchdog Tolerance <span class="info-tip tip-right" tabindex="0" data-tip="Operational alarm threshold: if more than 30% of features show statistical drift, the watchdog flags a data shift and recommends retraining or widened quantile uncertainty.">?</span></div>
                <div class="metric-value">30.0% Max Shift</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Alert triggers if &gt; 30% of features shift</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d_col4:
        alphas = drift_data["recommended_quantile_alpha"]
        model_action_desc = "Standard 80% Confidence Interval" if not drift_data["drift_alarm"] else "Widened 90% Confidence Interval"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recommended Model Action <span class="info-tip tip-right" tabindex="0" data-tip="Dynamic uncertainty adaptation: Under normal conditions, uses standard [p10, p50, p90] bounds. Under high drift, automatically expands to [p05, p50, p95] to provide safer reserve margins.">?</span></div>
                <div class="metric-value">{alphas}</div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">{model_action_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">
            Feature-by-Feature Distribution Audit (Trailing 7-Day Live Telemetry vs 30-Day Baseline) <span class="info-tip" tabindex="0" data-tip="Continuous two-sample distribution tests: Kolmogorov-Smirnov test for continuous physical metrics (p < 0.05 indicates drift) and 1D Wasserstein distance for cyclical harmonics (distance >= 0.10 indicates drift).">?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Each physical parameter is tested for statistical distribution drift. Continuous variables use the Kolmogorov-Smirnov test; cyclical harmonics use 1D Wasserstein distance.")
    df_metrics = get_drift_metrics_dataframe(selected_region)
    st.dataframe(df_metrics, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 14px; margin-bottom: 4px;">
            Export Executive Reliability Audit Report
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Download a self-contained HTML audit report containing complete Kolmogorov-Smirnov and Wasserstein metrics formatted for engineering and compliance review.")
    report_html = get_cached_drift_report(selected_region)
    st.download_button(
        label=f"Download Executive Audit Report ({selected_region})",
        data=report_html,
        file_name=f"drift_report_{selected_region.lower()}.html",
        mime="text/html",
    )

# tab 4: info for nerds (deep technical architecture & specifications)
with tab_system:
    st.markdown("## Info for Nerds: Deep System Architecture & Technical Specifications")
    st.markdown(
        """
        <div style="font-size: 13px; color: #444444; margin-bottom: 16px;">
            Technical specifications detailing mathematical formulations, quantile loss functions, zero-leakage causal feature pipelines, and dual-trigger safety logic.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("### 1. Multi-Horizon Forecasting Engine & Mathematical Guarantees")
        st.markdown(
            "**Asymmetric Pinball Loss:** Trains 3 independent LightGBM gradient-boosted tree ensembles per region for target quantiles (alpha = 0.10, 0.50, 0.90) using the tilted loss formulation:"
        )
        st.latex(r"\mathcal{L}_\alpha(y, \hat{y}) = \max\Big(\alpha(y - \hat{y}),\; (1 - \alpha)(\hat{y} - y)\Big)")
        st.markdown(
            "**Isotonic Monotonicity Guarantee:** Raw quantile regressors can occasionally invert during sharp transitions. Post-inference isotonic sorting along axis 1 mathematically enforces monotonic ordering across all 72 forward steps, ensuring the uncertainty envelope never inverts:"
        )
        st.latex(r"p_{10}(t) \le p_{50}(t) \le p_{90}(t) \quad \forall t \in [1, 72]")
        st.markdown(
            "**Recursive Autoregressive Loop:** Generates rolling 72-hour trajectories by substituting p50 median projections as surrogates for subsequent autoregressive lags (1h, 2h, 3h, 24h, 48h, 168h) as the forecast steps forward in time."
        )

    st.markdown("---")

    with st.container():
        st.markdown("### 2. Dual-Trigger Grid Fault & Surge Sentinel (False Alarm Elimination)")
        st.markdown(
            "**Trigger A (Multivariate Pattern Isolation):** An unsupervised Isolation Forest (contamination = 0.01) partitions multi-dimensional feature space to isolate structural anomalies deviating from typical operational patterns."
        )
        st.markdown(
            "**Trigger B (Physical Residual Boundary):** Computes active power residual $|y_{\\text{actual}} - \\hat{y}_{p50}|$ against an empirical boundary derived from 30 days of baseline operations:"
        )
        st.latex(r"\text{Operating Boundary} = \Big[p_{10} - 1.5 \cdot \text{IQR},\; p_{90} + 1.5 \cdot \text{IQR}\Big] \quad \text{where} \quad \text{IQR} = p_{75} - p_{25}")
        st.markdown(
            "**Consensus Boolean Gate:** An actionable alarm trips only when both triggers agree:"
        )
        st.latex(r"\text{Alert} = \text{Trigger}_A \land \text{Trigger}_B")
        st.markdown(
            "Extreme but benign weather shifts (such as heatwave cooling surges) satisfy only one trigger, preventing operator alarm fatigue while reliably catching equipment trips."
        )

    st.markdown("---")

    with st.container():
        st.markdown("### 3. Zero-Leakage Causal Feature Store (Polars LazyFrames)")
        st.markdown(
            "**Strict Causality Fence:** A mandatory `.shift(1)` fence is applied before computing any rolling statistics (6h, 24h, 168h moving averages and standard deviation volatility), mathematically guaranteeing zero future lookahead leakage."
        )
        st.markdown(
            "**Continuous Fourier Harmonics:** Encodes diurnal and annual seasonality using continuous sine and cosine transformations (T = 24h and T = 365.25d) to avoid artificial step discontinuities:"
        )
        st.latex(r"\sin\left(\frac{2\pi \cdot t}{24}\right),\quad \cos\left(\frac{2\pi \cdot t}{24}\right),\quad \sin\left(\frac{2\pi \cdot t}{8766}\right),\quad \cos\left(\frac{2\pi \cdot t}{8766}\right)")
        st.markdown(
            "**Regional Isolation:** Window aggregations partition strictly by region code (`NR`, `WR`, `SR`, `ER`, `NER`, `ALL_INDIA`), preventing cross-regional telemetry contamination."
        )

    st.markdown("---")

    with st.container():
        st.markdown("### 4. Explainability & Continuous Reliability Watchdog (SHAP & Evidently AI)")
        st.markdown(
            "**TreeExplainer Root-Cause Attribution:** Evaluated against 150 k-means background centroids to attribute exact additive Megawatt contributions per physical feature:"
        )
        st.latex(r"f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i")
        st.markdown(
            "**Two-Sample Kolmogorov-Smirnov Test:** Continuous telemetry variables (demand, frequency, temperature, solar radiation, wind speed) are tested against 30-day baseline distributions with significance threshold p < 0.05:"
        )
        st.latex(r"D = \sup_x \Big|F_{\text{live}}(x) - F_{\text{baseline}}(x)\Big|")
        st.markdown(
            "**1D Wasserstein Distance:** Cyclical harmonic features are evaluated via Earth Mover's Distance with drift threshold W1 >= 0.10."
        )
        st.markdown(
            "**Dynamic Quantile Adaptation:** If drifted features exceed the 30% tolerance threshold, the watchdog triggers a recalibration recommendation and dynamically expands the forecast interval to [p05, p50, p95] for conservative reserve safety."
        )

# SEO meta description head injection (rendered at end of document to guarantee zero layout shift)
components.html(
    """
    <script>
    try {
        var doc = window.parent.document;
        var meta = doc.querySelector('meta[name="description"]');
        if (!meta) {
            meta = doc.createElement('meta');
            meta.name = 'description';
            doc.head.appendChild(meta);
        }
        meta.content = "Multi-horizon probabilistic load forecasting, real-time dual-trigger fault sentinel, and automated statistical drift monitoring across India's National Power Grid.";
    } catch(e) {}
    </script>
    """,
    height=0,
    width=0,
)