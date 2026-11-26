from fastapi.testclient import TestClient


def test_rest_api_endpoints_and_cors():
    from src.api.main import app

    with TestClient(app) as client:
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_body = res_health.json()
        assert health_body["status"] == "healthy"
        assert health_body["model_version"] == "2.0.0"

        res_regions = client.get("/regions")
        assert res_regions.status_code == 200
        assert "ALL_INDIA" in res_regions.json()["regions"]

        res_fc = client.post("/forecast", json={"region": "ALL_INDIA", "lookback_hours": 168})
        assert res_fc.status_code == 200
        fc_body = res_fc.json()
        assert len(fc_body["forecasts"]) == 72

        res_bad = client.post("/forecast", json={"region": "INVALID_REGION", "lookback_hours": 168})
        assert res_bad.status_code == 400

        res_norm = client.post(
            "/anomalies",
            json={"region": "ALL_INDIA", "actual_demand_mw": 145000.0},
        )
        assert res_norm.status_code == 200
        norm_body = res_norm.json()

        res_shock = client.post(
            "/anomalies",
            json={"region": "ALL_INDIA", "actual_demand_mw": 310000.0},
        )
        assert res_shock.status_code == 200
        shock_body = res_shock.json()
        assert shock_body["residual_breach"] is True
        assert shock_body["is_structural"] is True
        assert len(shock_body["top_shap_features"]) == 3
        assert len(shock_body["anomaly_shap_features"]) == 3
        assert shock_body["anomaly_shap_features"] != norm_body["anomaly_shap_features"]
        assert {d["feature"] for d in shock_body["anomaly_shap_features"]} & {
            "demand_mw",
            "demand_delta_1h",
            "demand_delta_24h",
            "demand_zscore_24h",
        }

        res_drift = client.get("/drift?region=ALL_INDIA")
        assert res_drift.status_code == 200
        assert res_drift.json()["total_features"] == 12

        # untrusted origin must not receive permissive allow-credentials header
        res_cors = client.get("/health", headers={"Origin": "https://evil.example"})
        assert res_cors.headers.get("access-control-allow-credentials") is None
        assert res_cors.headers.get("access-control-allow-origin") != "*"

        res_404 = client.get("/nonexistent-api-route")
        assert res_404.status_code == 404


def test_asgi_server_security_headers_routes_and_404(monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    from src.server import app

    with TestClient(app) as client:
        for html_path in ("/", "/app"):
            for accept_encoding in ("identity", "gzip, deflate"):
                res_html = client.get(html_path, headers={"Accept-Encoding": accept_encoding})
                assert res_html.status_code == 200
                assert res_html.headers.get("x-content-type-options") == "nosniff"
                assert res_html.headers.get("x-frame-options") == "SAMEORIGIN"
                assert "max-age=" in res_html.headers.get("strict-transport-security", "")
                assert "frame-ancestors 'self'" in res_html.headers.get("content-security-policy", "")
                assert '<meta name="description"' in res_html.text
                assert "./assets-v2/static/js/" in res_html.text
                assert "no-store" in res_html.headers.get("cache-control", "")

        res_root = client.get("/")

        import re
        js_match = re.search(r"(\./assets-v2/static/js/[^\"]+)", res_root.text)
        assert js_match is not None
        asset_path = js_match.group(1).lstrip(".")
        res_asset = client.get(asset_path)
        assert res_asset.status_code == 200
        assert len(res_asset.content) > 100
        assert "max-age=86400" in res_asset.headers.get("cache-control", "")
        assert "immutable" not in res_asset.headers.get("cache-control", "").lower()

        # html spa fallback under /static/* must never be cached as an immutable/24h static asset
        res_static_fallback = client.get("/static/robots.txt")
        assert "no-store" in res_static_fallback.headers.get("cache-control", "")

        for health_path in ("/health", "/healthz"):
            res_health = client.get(health_path)
            assert res_health.status_code == 200
            health_json = res_health.json()
            assert health_json["status"] == "ok"
            assert health_json["artifacts_ready"] is True

        res_robots = client.get("/robots.txt")
        assert res_robots.status_code == 200
        assert "User-agent:" in res_robots.text

        assert client.get("/favicon.png").status_code == 200
        assert client.get("/favicon.ico").status_code == 200

        res_report = client.get("/reports/drift_report.html")
        assert res_report.status_code == 200
        assert "Statistical Drift Diagnostic Report" in res_report.text

        res_report_nr = client.get("/reports/drift_report.html?region=NR")
        assert res_report_nr.status_code == 200
        assert "NR — Northern Region" in res_report_nr.text

        res_report_invalid = client.get("/reports/drift_report.html?region=../src")
        assert res_report_invalid.status_code == 404

        res_404 = client.get("/nonexistent-route-audit-check")
        assert res_404.status_code == 404
        assert "Page Not Found" in res_404.text


def test_security_middleware_handles_upstream_gzip_html():
    import gzip
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import Response
    from starlette.routing import Route
    from src.server import RouteAndSecurityMiddleware, SEO_TITLE

    async def gzipped_html_endpoint(request):
        body = gzip.compress(
            b'<html><head><title>Streamlit</title></head><body><script src="./static/js/app.js"></script></body></html>'
        )
        return Response(
            content=body,
            media_type="text/html; charset=utf-8",
            headers={"Content-Encoding": "gzip"},
        )

    wrapped = Starlette(
        routes=[Route("/app", gzipped_html_endpoint)],
        middleware=[Middleware(RouteAndSecurityMiddleware)],
    )
    with TestClient(wrapped) as client:
        res = client.get("/app", headers={"Accept-Encoding": "gzip, deflate"})
        assert res.status_code == 200
        assert res.headers.get("content-encoding") is None
        assert f"<title>{SEO_TITLE}</title>" in res.text
        assert "./assets-v2/static/js/app.js" in res.text


def test_server_drift_db_fallback_and_health_readiness(monkeypatch):
    from unittest.mock import MagicMock, patch
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.routing import Route
    from src import server

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = ("<html><body>DB NR Report</body></html>",)

    test_srv = Starlette(
        routes=[
            Route("/reports/drift_report.html", server.drift_report_html, methods=["GET", "HEAD"]),
            Route("/health", server.health_json, methods=["GET", "HEAD"]),
            Route("/healthz", server.health_json, methods=["GET", "HEAD"]),
        ],
        middleware=[Middleware(server.RouteAndSecurityMiddleware)],
    )

    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://mock:mock@localhost:5432/mock")
    with TestClient(test_srv) as client:
        with patch("psycopg2.connect", return_value=mock_conn):
            res_db = client.get("/reports/drift_report.html?region=NR")
            assert res_db.status_code == 200
            assert "DB NR Report" in res_db.text
            mock_conn.close.assert_called_once()

        with patch.object(server, "validate_artifacts_presence", return_value=False):
            res_unhealthy = client.get("/healthz")
            assert res_unhealthy.status_code == 503
            assert res_unhealthy.json()["status"] == "unhealthy"
            assert res_unhealthy.json()["artifacts_ready"] is False