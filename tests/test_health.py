from fastapi.testclient import TestClient

from movie_reco_api.app import create_app


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_ready" in body

