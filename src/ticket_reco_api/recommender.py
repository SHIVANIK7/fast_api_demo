from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


class ModelNotReadyError(RuntimeError):
    pass


class RouteNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class RecommenderArtifacts:
    model: object
    reference_data: pd.DataFrame | None = None


def load_artifacts(artifacts_dir: Path) -> RecommenderArtifacts:
    model_path = artifacts_dir / "model.joblib"
    ref_path = artifacts_dir / "reference_data.parquet"

    if not model_path.exists():
        raise ModelNotReadyError(
            "Model artifact not found. Expected: artifacts/model.joblib"
        )

    model = joblib.load(model_path)
    reference_data = pd.read_parquet(ref_path) if ref_path.exists() else None

    return RecommenderArtifacts(model=model, reference_data=reference_data)


def _build_features(
    source: str,
    destination: str,
    date_of_travel: str,
    passengers: int,
) -> pd.DataFrame:
    travel_dt = pd.to_datetime(date_of_travel)

    return pd.DataFrame(
        [
            {
                "source": source,
                "destination": destination,
                "date_of_travel": str(travel_dt.date()),
                "passengers": passengers,
                "journey_day": travel_dt.day,
                "journey_month": travel_dt.month,
                "journey_weekday": travel_dt.dayofweek,
            }
        ]
    )


def _resolve_airline_name(
    reference_data: pd.DataFrame | None,
    source: str,
    destination: str,
    date_of_travel: str,
) -> str:
    if reference_data is None or reference_data.empty:
        return "Demo Airline"

    ref = reference_data.copy()
    if "date_of_travel" in ref.columns:
        ref["date_of_travel"] = pd.to_datetime(ref["date_of_travel"]).dt.date.astype(str)

    mask = (
        ref["source"].astype(str).str.lower().eq(source.lower())
        & ref["destination"].astype(str).str.lower().eq(destination.lower())
    )

    if "date_of_travel" in ref.columns:
        same_date = ref["date_of_travel"].eq(str(pd.to_datetime(date_of_travel).date()))
        exact = ref[mask & same_date]
        if not exact.empty and "airline_name" in exact.columns:
            return str(exact.iloc[0]["airline_name"])

    route_only = ref[mask]
    if not route_only.empty and "airline_name" in route_only.columns:
        return str(route_only.iloc[0]["airline_name"])

    raise RouteNotFoundError(
        f"No matching route found for source='{source}' and destination='{destination}'."
    )


def predict_ticket(
    art: RecommenderArtifacts,
    source: str,
    destination: str,
    date_of_travel: str,
    passengers: int,
) -> dict:
    features = _build_features(
        source=source,
        destination=destination,
        date_of_travel=date_of_travel,
        passengers=passengers,
    )

    predicted_price = float(art.model.predict(features)[0])
    airline_name = _resolve_airline_name(
        reference_data=art.reference_data,
        source=source,
        destination=destination,
        date_of_travel=date_of_travel,
    )

    return {
        "source": source,
        "destination": destination,
        "predicted_price": round(predicted_price, 2),
        "airline_name": airline_name,
        "date_of_travel": str(pd.to_datetime(date_of_travel).date()),
    }
