from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# =========================
# Exceptions
# =========================

class ModelNotReadyError(RuntimeError):
    pass


class RouteNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class AirlineArtifacts:
    model: object
    origin_encoder: object
    dest_encoder: object
    features: list[str]
    airline_lookup: dict
    history: pd.DataFrame

def load_artifacts(artifacts_dir: Path) -> AirlineArtifacts:
    model_path = artifacts_dir / "model.pkl"
    origin_enc_path = artifacts_dir / "origin_encoder.pkl"
    dest_enc_path = artifacts_dir / "dest_encoder.pkl"
    features_path = artifacts_dir / "features.pkl"
    airline_lookup_path = artifacts_dir / "airline_lookup.pkl"
    history_path = artifacts_dir / "data.parquet"

    if not all([
        model_path.exists(),
        origin_enc_path.exists(),
        dest_enc_path.exists(),
        features_path.exists(),
        airline_lookup_path.exists(),
        history_path.exists(),
    ]):
        raise ModelNotReadyError(
            "Model artifacts not found. Run training pipeline first."
        )

    return AirlineArtifacts(
        model=joblib.load(model_path),
        origin_encoder=joblib.load(origin_enc_path),
        dest_encoder=joblib.load(dest_enc_path),
        features=joblib.load(features_path),
        airline_lookup=joblib.load(airline_lookup_path),
        history=pd.read_parquet(history_path),
    )

def safe_label_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return -1

def predict_price_for_week(
    art: AirlineArtifacts,
    origin: str,
    destination: str,
    travel_date: str,   # e.g. "2026-05-10"
) -> dict:
    
    df = art.history.copy()
    travel_date = pd.to_datetime(travel_date)
    
    # Get route history
    route_mask = (df["origin"] == origin) & (df["destination"] == destination)
    history = df[route_mask].sort_values("date").copy()
    
    if len(history) < 3:
        raise RouteNotFoundError("Not enough historical data for this route.")
    
    # Encode safely
    origin_encoded = safe_label_transform(art.origin_encoder, origin)
    destination_encoded = safe_label_transform(art.dest_encoder, destination)
    
    # Create features for given week
    month = travel_date.month
    year = travel_date.year
    quarter = (month - 1) // 3 + 1
    
    # Lag features (latest known values)
    temp = {
        "origin_encoded": origin_encoded,
        "destination_encoded": destination_encoded,
        "month": month,
        "year": year,
        "quarter": quarter,
        "price_lag_1": history["price"].iloc[-1],
        "price_lag_2": history["price"].iloc[-2],
        "price_lag_3": history["price"].iloc[-3],
        "rolling_mean_3": history["price"].iloc[-3:].mean(),
    }
    
    X = pd.DataFrame([temp])[art.features]
    
    pred_price = float(art.model.predict(X)[0])
    
    route = f"{origin}_{destination}"
    airline = art.airline_lookup.get(route, "Unknown")
    
    return {
        "origin": origin,
        "destination": destination,
        "date": str(travel_date.date()),
        "predicted_price": round(pred_price, 2),
        "airline": airline
    }