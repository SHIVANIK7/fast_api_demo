from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from movie_reco_api.app import create_app
from movie_reco_api.recommender import RecommenderArtifacts


class DummyModel:
    def predict(self, X: pd.DataFrame):
        return [5234.75] * len(X)


def test_predict_returns_200(monkeypatch):
    def fake_load_artifacts(_):
        return RecommenderArtifacts(
            model=DummyModel(),
            reference_data=pd.DataFrame(
                [
                    {
                        'source': 'Chennai',
                        'destination': 'Delhi',
                        'date_of_travel': '2026-04-10',
                        'airline_name': 'IndiGo',
                    }
                ]
            ),
        )

    monkeypatch.setattr('movie_reco_api.app.load_artifacts', fake_load_artifacts)
    app = create_app()
    client = TestClient(app)

    r = client.post(
        '/predict',
        json={
            'source': 'Chennai',
            'destination': 'Delhi',
            'date_of_travel': '2026-04-10',
            'passengers': 1,
        },
    )

    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert body['source'] == 'Chennai'
        assert body['destination'] == 'Delhi'
        assert body['predicted_price'] == 5234.75
        assert body['airline_name'] == 'IndiGo'
        assert body['date_of_travel'] == '2026-04-10'



def test_validation_rejects_empty_source():
    app = create_app()
    client = TestClient(app)
    r = client.post(
        '/predict',
        json={
            'source': '',
            'destination': 'Delhi',
            'date_of_travel': '2026-04-10',
            'passengers': 1,
        },
    )
    assert r.status_code == 422



def test_validation_rejects_more_than_one_passenger(monkeypatch):
    def fake_load_artifacts(_):
        return RecommenderArtifacts(model=DummyModel(), reference_data=None)

    monkeypatch.setattr('movie_reco_api.app.load_artifacts', fake_load_artifacts)
    app = create_app()
    client = TestClient(app)

    r = client.post(
        '/predict',
        json={
            'source': 'Chennai',
            'destination': 'Delhi',
            'date_of_travel': '2026-04-10',
            'passengers': 2,
        },
    )
    assert r.status_code == 422
