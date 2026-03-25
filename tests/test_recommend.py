from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from ticket_reco_api.app import create_app
from ticket_reco_api.recommender import AirlineArtifacts


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
            origin_encoder=DummyEncoder(["Dubai"]),
            dest_encoder=DummyEncoder(["London"]),
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
                    {"origin": "Dubai", "destination": "London", "date": "2026-04-05", "price": 5000},
                    {"origin": "Dubai", "destination": "London", "date": "2026-04-12", "price": 5100},
                    {"origin": "Dubai", "destination": "London", "date": "2026-04-19", "price": 5200},
                ]
            ),
        )

    def fake_read_csv(_):
        return pd.DataFrame(
            [
                {"origin": "Dubai", "destination": "London"},
                {"origin": "Dubai", "destination": "Paris"},
                {"origin": "Chennai", "destination": "Delhi"},
            ]
        )

    monkeypatch.setattr("ticket_reco_api.app.load_artifacts", fake_load_artifacts)
    monkeypatch.setattr("ticket_reco_api.app.pd.read_csv", fake_read_csv)

    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/predict",
            json={
                "origin": "Dubai",
                "destination": "London",
                "travel_date": "2026-04-05",
            },
        )

    assert r.status_code == 200
    body = r.json()
    assert body["origin"] == "Dubai"
    assert body["destination"] == "London"
    assert body["predicted_price"] == 5234.75
    assert body["airline"] == "Emirates"
    assert body["date"] == "2026-04-05"


def test_validation_rejects_empty_origin(monkeypatch):
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
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-05", "price": 5000},
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-12", "price": 5100},
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-19", "price": 5200},
                ]
            ),
        )

    def fake_read_csv(_):
        return pd.DataFrame([{"origin": "Chennai", "destination": "Delhi"}])

    monkeypatch.setattr("ticket_reco_api.app.load_artifacts", fake_load_artifacts)
    monkeypatch.setattr("ticket_reco_api.app.pd.read_csv", fake_read_csv)

    app = create_app()
    with TestClient(app) as client:
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
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-05", "price": 5000},
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-12", "price": 5100},
                ]
            ),
        )

    def fake_read_csv(_):
        return pd.DataFrame([{"origin": "Chennai", "destination": "Delhi"}])

    monkeypatch.setattr("ticket_reco_api.app.load_artifacts", fake_load_artifacts)
    monkeypatch.setattr("ticket_reco_api.app.pd.read_csv", fake_read_csv)

    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/predict",
            json={
                "origin": "Chennai",
                "destination": "Delhi",
                "travel_date": "2026-04-05",
            },
        )

    assert r.status_code == 404


def test_sources_returns_unique_origins(monkeypatch):
    def fake_load_artifacts(_):
        return AirlineArtifacts(
            model=DummyModel(),
            origin_encoder=DummyEncoder(["Dubai"]),
            dest_encoder=DummyEncoder(["London"]),
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
                    {"origin": "Dubai", "destination": "London", "date": "2026-04-05", "price": 5000},
                    {"origin": "Dubai", "destination": "Paris", "date": "2026-04-12", "price": 5100},
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-19", "price": 5200},
                ]
            ),
        )

    def fake_read_csv(_):
        return pd.DataFrame(
            [
                {"origin": "Dubai", "destination": "London"},
                {"origin": "Dubai", "destination": "Paris"},
                {"origin": "Chennai", "destination": "Delhi"},
                {"origin": "Chennai", "destination": "Mumbai"},
            ]
        )

    monkeypatch.setattr("ticket_reco_api.app.load_artifacts", fake_load_artifacts)
    monkeypatch.setattr("ticket_reco_api.app.pd.read_csv", fake_read_csv)

    app = create_app()
    with TestClient(app) as client:
        r = client.get("/sources")

    assert r.status_code == 200
    assert r.json() == ["Chennai", "Dubai"]


def test_destinations_returns_unique_filtered_values(monkeypatch):
    def fake_load_artifacts(_):
        return AirlineArtifacts(
            model=DummyModel(),
            origin_encoder=DummyEncoder(["Dubai"]),
            dest_encoder=DummyEncoder(["London"]),
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
                    {"origin": "Dubai", "destination": "London", "date": "2026-04-05", "price": 5000},
                    {"origin": "Dubai", "destination": "Paris", "date": "2026-04-12", "price": 5100},
                    {"origin": "Chennai", "destination": "Delhi", "date": "2026-04-19", "price": 5200},
                ]
            ),
        )

    def fake_read_csv(_):
        return pd.DataFrame(
            [
                {"origin": "Dubai", "destination": "London"},
                {"origin": "Dubai", "destination": "Paris"},
                {"origin": "Dubai", "destination": "London"},
                {"origin": "Chennai", "destination": "Delhi"},
                {"origin": "Chennai", "destination": "Mumbai"},
            ]
        )

    monkeypatch.setattr("ticket_reco_api.app.load_artifacts", fake_load_artifacts)
    monkeypatch.setattr("ticket_reco_api.app.pd.read_csv", fake_read_csv)

    app = create_app()
    with TestClient(app) as client:
        r = client.get("/destinations", params={"origin": "Dubai"})

    assert r.status_code == 200
    assert r.json() == ["London", "Paris"]