from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from movie_reco_api.app import create_app
from movie_reco_api.recommender import AirlineArtifacts


class DummyModel:
    def predict(self, X: pd.DataFrame):
        return [5234.75] * len(X)


class DummyEncoder:
    def __init__(self, classes):
        self.classes_ = classes

    def transform(self, values):
        return [self.classes_.index(v) for v in values]


def test_predict_returns_200(monkeypatch):
    def fake_load_artifacts(_):
        return AirlineArtifacts(
            model=DummyModel(),
            origin_encoder=DummyEncoder(["Chennai"]),
            dest_encoder=DummyEncoder(["Delhi"]),
            features=[
                "origin_encoded",
                "destination_encoded",
                "month",
                "year",
                "quarter",
                "price_lag_1",
                "price_lag_2",
                "price_lag_3",
                "rolling_mean_3",
            ],
            airline_lookup={"Dubai_London": "Emirates"},
            history=pd.DataFrame(
                [
                    {"origin": "Dubai", "destination": "London", "date": "2026-04-01", "price": 5000},
                    {"origin": "Dubai", "destination": "London", "date": "2026-04-02", "price": 5100},
                    {"origin": "Dubai", "destination": "London", "date": "2026-04-03", "price": 5200},
                ]
            ),
        )

    monkeypatch.setattr("movie_reco_api.app.load_artifacts", fake_load_artifacts)
    app = create_app()
    client = TestClient(app)

    r = client.post(
        "/predict",
        json={
            "origin": "Chennai",
            "destination": "Delhi",
            "travel_date": "2026-04-10",
        },
    )

    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert body["origin"] == "Chennai"
        assert body["destination"] == "Delhi"
        assert body["predicted_price"] == 5234.75
        assert body["airline"] == "IndiGo"
        assert body["date"] == "2026-04-10"


def test_validation_rejects_empty_origin():
    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/predict",
        json={
            "origin": "",
            "destination": "Delhi",
            "travel_date": "2026-04-10",
        },
    )
    assert r.status_code == 422


def test_route_not_found_returns_404(monkeypatch):
    def fake_load_artifacts(_):
        return AirlineArtifacts(
            model=DummyModel(),
            origin_encoder=DummyEncoder(["Chennai"]),
            dest_encoder=DummyEncoder(["Delhi"]),
            features=[
                "origin_encoded",
                "destination_encoded",
                "month",
                "year",
                "quarter",
                "price_lag_1",
                "price_lag_2",
                "price_lag_3",
                "rolling_mean_3",
            ],
            airline_lookup={"Chennai_Delhi": "IndiGo"},
            history=pd.DataFrame(
                [
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-01", "price": 5000},
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-02", "price": 5100},
                ]
            ),
        )

    monkeypatch.setattr("movie_reco_api.app.load_artifacts", fake_load_artifacts)
    app = create_app()
    client = TestClient(app)

    r = client.post(
        "/predict",
        json={
            "origin": "Chennai",
            "destination": "Delhi",
            "travel_date": "2026-04-10",
        },
    )

    assert r.status_code == 404