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

        res_shock = client.post(
            "/anomalies",
            json={"region": "ALL_INDIA", "actual_demand_mw": 310000.0},
        )
        assert res_shock.status_code == 200
        shock_body = res_shock.json()
        assert shock_body["residual_breach"] is True
        assert shock_body["is_structural"] is True
        assert len(shock_body["top_shap_features"]) == 3

        res_drift = client.get("/drift?region=ALL_INDIA")
        assert res_drift.status_code == 200
        assert res_drift.json()["total_features"] == 12

        # untrusted origin must not receive permissive allow-credentials header
        res_cors = client.get("/health", headers={"Origin": "https://evil.example"})
        assert res_cors.headers.get("access-control-allow-credentials") is None
        assert res_cors.headers.get("access-control-allow-origin") != "*"

        res_404 = client.get("/nonexistent-api-route")
        assert res_404.status_code == 404


def test_asgi_server_security_headers_routes_and_404():
    from src.server import app

    with TestClient(app) as client:
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert res_root.headers.get("x-content-type-options") == "nosniff"
        assert res_root.headers.get("x-frame-options") == "SAMEORIGIN"
        assert "max-age=" in res_root.headers.get("strict-transport-security", "")
        assert "frame-ancestors 'self'" in res_root.headers.get("content-security-policy", "")
        assert '<meta name="description"' in res_root.text

        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "ok"

        res_robots = client.get("/robots.txt")
        assert res_robots.status_code == 200
        assert "User-agent:" in res_robots.text

        assert client.get("/favicon.png").status_code == 200
        assert client.get("/favicon.ico").status_code == 200

        res_report = client.get("/reports/drift_report.html")
        assert res_report.status_code == 200
        assert "Statistical Drift Diagnostic Report" in res_report.text

        res_404 = client.get("/nonexistent-route-audit-check")
        assert res_404.status_code == 404
        assert "Page Not Found" in res_404.text