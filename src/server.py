import gzip
import logging
import os
import threading
import time

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from collections import OrderedDict
from typing import Optional, Sequence
from urllib.parse import unquote
import streamlit as st
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.models.load_artifacts import REGIONS, validate_artifacts_presence
from src.ui.helpers import render_not_found_html

logger = logging.getLogger(__name__)
VALID_REGIONS = set(REGIONS)

STATIC_DIR = os.path.join(root_dir, "static")
ROBOTS_PATH = os.path.join(STATIC_DIR, "robots.txt")
FAVICON_PATH = os.path.join(STATIC_DIR, "favicon.png")
DRIFT_REPORT_PATH = os.path.join(STATIC_DIR, "reports", "drift_report.html")

SEO_TITLE = "grid-telemetry-engine | Mahesh Kakde"
SEO_DESCRIPTION = (
    "72-hour probabilistic electricity demand forecasting, dual-trigger anomaly detection, "
    "and feature drift tracking across India's power grid."
)

ALLOWED_EXACT_PATHS = {
    "/",
    "/app",
    "/robots.txt",
    "/favicon.png",
    "/favicon.ico",
    "/reports/drift_report.html",
    "/health",
    "/healthz",
}

ALLOWED_PREFIXES = (
    "/_stcore/",
    "/static/",
    "/assets-v2/static/",
    "/media/",
    "/component/",
    "/app/static/",
)
STATIC_ALIAS_PREFIX = "/assets-v2/static/"

SECURITY_HEADERS: Sequence[tuple[bytes, bytes]] = (
    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"SAMEORIGIN"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    (
        b"content-security-policy",
        b"default-src 'self'; "
        b"script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        b"style-src 'self' 'unsafe-inline'; "
        b"img-src 'self' data: blob:; "
        b"font-src 'self' data:; "
        b"connect-src 'self' ws: wss:; "
        b"frame-ancestors 'self'; "
        b"base-uri 'self'; "
        b"object-src 'none'",
    ),
)

NOT_FOUND_HTML = render_not_found_html(favicon_href="/favicon.png")


_BOOT_STYLE = """<style id="telemetry-boot-style">
      html, body {
        margin: 0;
        padding: 0;
        background-color: #f5f5f7;
        color: #1d1d1f;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", system-ui, sans-serif;
      }
      #telemetry-boot-shell {
        max-width: 1380px;
        margin: 0 auto;
        padding: 28px clamp(16px, 3vw, 44px);
        box-sizing: border-box;
      }
      .boot-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 18px;
      }
      .boot-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0 0 6px 0;
      }
      .boot-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12.5px;
        color: #515154;
        background: #ffffff;
        border: 1px solid #d2d2d7;
        border-radius: 10px;
        padding: 6px 12px;
      }
      .boot-track {
        height: 3px;
        background: #e5e5ea;
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 18px;
      }
      .boot-fill {
        width: 38%;
        height: 100%;
        background: #0066cc;
        border-radius: 999px;
        animation: boot-slide 1.35s ease-in-out infinite;
      }
      .boot-kpis {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 14px;
      }
      .boot-card {
        background: #ffffff;
        border: 1px solid #d2d2d7;
        border-radius: 16px;
        padding: 16px 18px;
        height: 92px;
        box-sizing: border-box;
      }
      .boot-line {
        background: linear-gradient(90deg, #ececf1 25%, #f5f5f7 50%, #ececf1 75%);
        background-size: 200% 100%;
        border-radius: 6px;
        animation: boot-shimmer 1.4s linear infinite;
      }
      .boot-chart {
        background: #ffffff;
        border: 1px solid #d2d2d7;
        border-radius: 16px;
        height: 320px;
        padding: 18px;
        box-sizing: border-box;
      }
      @keyframes boot-slide {
        0% { transform: translateX(-110%); }
        100% { transform: translateX(320%); }
      }
      @keyframes boot-shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
      @media (max-width: 900px) {
        .boot-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
      @media (max-width: 540px) {
        .boot-kpis { grid-template-columns: 1fr; }
      }
      @media (prefers-reduced-motion: reduce) {
        .boot-fill, .boot-line { animation: none; }
      }
    </style>"""

_BOOT_ROOT_HTML = (
    '<div id="root">'
    '<div id="telemetry-boot-shell" role="status" aria-live="polite">'
    '<div class="boot-head">'
    '<div><div class="boot-title">National Power Grid Telemetry</div></div>'
    '<div class="boot-status">Loading dashboard...</div>'
    '</div>'
    '<div class="boot-track"><div class="boot-fill"></div></div>'
    '<div class="boot-kpis">'
    '<div class="boot-card"><div class="boot-line" style="height:11px;width:52%;margin-bottom:12px;"></div><div class="boot-line" style="height:24px;width:72%;"></div></div>'
    '<div class="boot-card"><div class="boot-line" style="height:11px;width:48%;margin-bottom:12px;"></div><div class="boot-line" style="height:24px;width:68%;"></div></div>'
    '<div class="boot-card"><div class="boot-line" style="height:11px;width:44%;margin-bottom:12px;"></div><div class="boot-line" style="height:24px;width:58%;"></div></div>'
    '<div class="boot-card"><div class="boot-line" style="height:11px;width:50%;margin-bottom:12px;"></div><div class="boot-line" style="height:24px;width:62%;"></div></div>'
    '</div>'
    '<div class="boot-chart"><div class="boot-line" style="height:14px;width:30%;margin-bottom:18px;"></div><div class="boot-line" style="height:230px;width:100%;border-radius:10px;"></div></div>'
    '</div>'
    '</div>'
)


def _inject_seo_head(html_bytes: bytes) -> bytes:
    html_text = html_bytes.decode("utf-8", errors="ignore")
    if "<title>Streamlit</title>" in html_text:
        replacement = (
            f"<title>{SEO_TITLE}</title>\n"
            f'    <meta name="description" content="{SEO_DESCRIPTION}" />'
        )
        html_text = html_text.replace("<title>Streamlit</title>", replacement, 1)
    if "</head>" in html_text and "telemetry-boot-style" not in html_text:
        html_text = html_text.replace("</head>", f"    {_BOOT_STYLE}\n  </head>", 1)
    if '<div id="root"></div>' in html_text:
        html_text = html_text.replace('<div id="root"></div>', _BOOT_ROOT_HTML, 1)
    # route bundled static assets through a fresh versioned namespace so old immutable cdn edges cannot poison js imports
    html_text = html_text.replace('"./static/', '"./assets-v2/static/')
    html_text = html_text.replace("'./static/", "'./assets-v2/static/")
    return html_text.encode("utf-8")


class RouteAndSecurityMiddleware:
    # enforces http security headers, injects server-side seo meta & asset cache namespace, and returns http 404 for unknown routes

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @staticmethod
    def _has_unsafe_segments(path: str, raw_path: bytes = b"") -> bool:
        candidates = [path, unquote(path)]
        if raw_path:
            raw_str = raw_path.decode("latin-1", errors="ignore")
            candidates.extend([raw_str, unquote(raw_str)])
        for cand in candidates:
            if "\x00" in cand or "\\" in cand:
                return True
            if any(seg in (".", "..") for seg in cand.split("/")):
                return True
        return False

    @staticmethod
    def _is_allowed_path(path: str, raw_path: bytes = b"") -> bool:
        if RouteAndSecurityMiddleware._has_unsafe_segments(path, raw_path):
            return False
        if path in ALLOWED_EXACT_PATHS:
            return True
        return any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)

    @staticmethod
    def _apply_security_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
        existing_keys = {k.lower() for k, _ in headers}
        for key, val in SECURITY_HEADERS:
            if key not in existing_keys:
                headers.append((key, val))
        return headers

    @staticmethod
    def _drop_header(headers: list[tuple[bytes, bytes]], name: bytes) -> list[tuple[bytes, bytes]]:
        target = name.lower()
        return [(k, v) for k, v in headers if k.lower() != target]

    @staticmethod
    def _set_header(headers: list[tuple[bytes, bytes]], name: bytes, value: bytes) -> list[tuple[bytes, bytes]]:
        target = name.lower()
        filtered = RouteAndSecurityMiddleware._drop_header(headers, target)
        filtered.append((target, value))
        return filtered

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        raw_path = scope.get("raw_path", b"")
        method = scope.get("method", "GET").upper()

        if not self._is_allowed_path(path, raw_path):
            async def send_404(message: Message) -> None:
                if message["type"] == "http.response.start":
                    headers = self._apply_security_headers(list(message.get("headers", [])))
                    message = {**message, "headers": headers}
                await send(message)

            response = HTMLResponse(content=NOT_FOUND_HTML, status_code=404)
            await response(scope, receive, send_404)
            return

        is_static_asset = path.startswith(("/static/", STATIC_ALIAS_PREFIX))
        forward_scope = scope
        if path.startswith(STATIC_ALIAS_PREFIX):
            rewritten_path = "/static/" + path[len(STATIC_ALIAS_PREFIX):]
            raw_path = scope.get("raw_path", b"")
            rewritten_raw = (
                b"/static/" + raw_path[len(STATIC_ALIAS_PREFIX.encode("ascii")):]
                if raw_path.startswith(STATIC_ALIAS_PREFIX.encode("ascii"))
                else rewritten_path.encode("utf-8")
            )
            forward_scope = {**scope, "path": rewritten_path, "raw_path": rewritten_raw}

        if path in {"/", "/app"} and method == "GET":
            start_message: Message | None = None
            body_chunks: list[bytes] = []
            # request uncompressed html from upstream before rewriting head and asset paths
            upstream_headers = self._set_header(
                list(forward_scope.get("headers", [])),
                b"accept-encoding",
                b"identity",
            )
            html_scope = {**forward_scope, "headers": upstream_headers}

            async def buffer_and_inject(message: Message) -> None:
                nonlocal start_message
                if message["type"] == "http.response.start":
                    start_message = message
                elif message["type"] == "http.response.body":
                    body_chunks.append(message.get("body", b""))
                    if not message.get("more_body", False):
                        assert start_message is not None
                        headers = self._apply_security_headers(list(start_message.get("headers", [])))
                        headers = self._set_header(headers, b"cache-control", b"no-cache, no-store, must-revalidate")
                        content_type = next(
                            (v.decode("latin-1").lower() for k, v in headers if k.lower() == b"content-type"),
                            "",
                        )
                        content_encoding = next(
                            (v.decode("latin-1").strip().lower() for k, v in headers if k.lower() == b"content-encoding"),
                            "",
                        )
                        raw_body = b"".join(body_chunks)
                        if "text/html" in content_type:
                            if content_encoding == "gzip":
                                raw_body = gzip.decompress(raw_body)
                                headers = self._drop_header(headers, b"content-encoding")
                                content_encoding = ""
                            if content_encoding in ("", "identity"):
                                raw_body = _inject_seo_head(raw_body)
                                headers = self._drop_header(headers, b"etag")
                                headers = self._set_header(headers, b"content-length", str(len(raw_body)).encode("ascii"))
                        await send({**start_message, "headers": headers})
                        await send({"type": "http.response.body", "body": raw_body, "more_body": False})

            await self.app(html_scope, receive, buffer_and_inject)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = self._apply_security_headers(list(message.get("headers", [])))
                if path in {"/", "/app"}:
                    headers = self._set_header(headers, b"cache-control", b"no-cache, no-store, must-revalidate")
                elif is_static_asset:
                    status_code = int(message.get("status", 200))
                    content_type = next(
                        (v.decode("latin-1").lower() for k, v in headers if k.lower() == b"content-type"),
                        "",
                    )
                    # only cache actual static asset payloads, never html spa fallbacks
                    if status_code == 200 and "text/html" not in content_type:
                        headers = self._set_header(headers, b"cache-control", b"public, max-age=86400")
                    else:
                        headers = self._set_header(headers, b"cache-control", b"no-cache, no-store, must-revalidate")
                message = {**message, "headers": headers}
            await send(message)

        await self.app(forward_scope, receive, send_with_security_headers)


async def robots_txt(request: Request):
    if os.path.exists(ROBOTS_PATH):
        with open(ROBOTS_PATH, "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read(), media_type="text/plain")
    return PlainTextResponse("User-agent: *\nAllow: /\n", media_type="text/plain")


async def favicon_png(request: Request):
    if os.path.exists(FAVICON_PATH):
        return FileResponse(FAVICON_PATH, media_type="image/png")
    return HTMLResponse(content=NOT_FOUND_HTML, status_code=404)


DRIFT_REPORT_CACHE_TTL_SEC = 300.0
DRIFT_REPORT_CACHE_MAX_ENTRIES = 12
_DRIFT_CACHE_LOCK = threading.Lock()
_DRIFT_REPORT_CACHE: "OrderedDict[tuple[str, str], tuple[float, str]]" = OrderedDict()


def _clear_drift_report_cache() -> None:
    with _DRIFT_CACHE_LOCK:
        _DRIFT_REPORT_CACHE.clear()


def _get_cached_drift_report(region: str, db_url: str) -> Optional[str]:
    key = (region, db_url)
    now = time.monotonic()
    with _DRIFT_CACHE_LOCK:
        entry = _DRIFT_REPORT_CACHE.get(key)
        if entry is None:
            return None
        expires_at, html_body = entry
        if now >= expires_at:
            _DRIFT_REPORT_CACHE.pop(key, None)
            return None
        _DRIFT_REPORT_CACHE.move_to_end(key)
        return html_body


def _set_cached_drift_report(region: str, db_url: str, html_body: str) -> None:
    key = (region, db_url)
    expires_at = time.monotonic() + DRIFT_REPORT_CACHE_TTL_SEC
    with _DRIFT_CACHE_LOCK:
        if key in _DRIFT_REPORT_CACHE:
            _DRIFT_REPORT_CACHE.move_to_end(key)
        elif len(_DRIFT_REPORT_CACHE) >= DRIFT_REPORT_CACHE_MAX_ENTRIES:
            _DRIFT_REPORT_CACHE.popitem(last=False)
        _DRIFT_REPORT_CACHE[key] = (expires_at, html_body)


def _fetch_drift_report_from_db(region: str, db_url: str) -> Optional[str]:
    cached = _get_cached_drift_report(region, db_url)
    if cached is not None:
        return cached

    import psycopg2

    conn = psycopg2.connect(db_url, connect_timeout=3)
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 5000;")
            cur.execute(
                "SELECT drift_report_html FROM precomputed_snapshots WHERE region = %s;",
                (region,),
            )
            row = cur.fetchone()
            html_body = row[0] if row and row[0] else None
            if html_body:
                _set_cached_drift_report(region, db_url, html_body)
            return html_body
    finally:
        conn.close()


async def drift_report_html(request: Request):
    region = request.query_params.get("region", "ALL_INDIA").upper()
    if region not in VALID_REGIONS:
        return HTMLResponse(content=NOT_FOUND_HTML, status_code=404)

    db_url = os.environ.get("SUPABASE_DB_URL")
    if db_url:
        try:
            html_content = await run_in_threadpool(_fetch_drift_report_from_db, region, db_url)
            if html_content:
                return HTMLResponse(content=html_content, media_type="text/html")
        except Exception as exc:
            logger.warning(f"failed to load drift report from supabase for {region}: {exc}")

    regional_report_path = (
        DRIFT_REPORT_PATH
        if region == "ALL_INDIA"
        else os.path.join(STATIC_DIR, "reports", f"drift_report_{region}.html")
    )
    if os.path.exists(regional_report_path):
        return FileResponse(regional_report_path, media_type="text/html")
    if os.path.exists(DRIFT_REPORT_PATH):
        return FileResponse(DRIFT_REPORT_PATH, media_type="text/html")
    return HTMLResponse(content=NOT_FOUND_HTML, status_code=404)


async def health_json(request: Request):
    try:
        artifacts_ready = await run_in_threadpool(validate_artifacts_presence)
    except Exception as exc:
        logger.warning(f"health check artifact validation failed: {exc}")
        artifacts_ready = False

    status_code = 200 if artifacts_ready else 503
    return JSONResponse(
        {
            "status": "ok" if artifacts_ready else "unhealthy",
            "service": "grid-telemetry-engine",
            "artifacts_ready": artifacts_ready,
        },
        status_code=status_code,
    )


app = st.App(
    os.path.join(root_dir, "src", "ui", "app.py"),
    routes=[
        Route("/robots.txt", robots_txt, methods=["GET", "HEAD"]),
        Route("/favicon.png", favicon_png, methods=["GET", "HEAD"]),
        Route("/favicon.ico", favicon_png, methods=["GET", "HEAD"]),
        Route("/reports/drift_report.html", drift_report_html, methods=["GET", "HEAD"]),
        Route("/health", health_json, methods=["GET", "HEAD"]),
        Route("/healthz", health_json, methods=["GET", "HEAD"]),
    ],
    middleware=[Middleware(RouteAndSecurityMiddleware)],
)