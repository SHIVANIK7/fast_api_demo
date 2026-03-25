from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ticket_reco_api.recommender import (
    ModelNotReadyError,
    RouteNotFoundError,
    load_artifacts,
    predict_price_for_week,
)
from ticket_reco_api.schemas import TicketPriceRequest, TicketPriceResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="Flight Ticket Price Prediction API",
        version="0.1.0",
        description="Predict flight ticket price from origin, destination, and travel date.",
    )

    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "scripts" / "artifacts"
    data_path = project_root / "scripts" / "data.csv"

    state = {
        "art": None,
        "data": None,
    }
    @app.get("/")
    def root() -> dict:
        return {
            "message": "Flight Ticket Price Prediction API is running",
            "docs": "/docs",
            "health": "/health",
            "sources": "/sources",
            "destinations": "/destinations",
        }
    @app.on_event("startup")
    def _startup() -> None:
        try:
            state["art"] = load_artifacts(artifacts_dir)
            print("Model artifacts loaded successfully.")
        except ModelNotReadyError:
            state["art"] = None
            print("Model artifacts not found.")

        try:
            state["data"] = pd.read_csv(data_path)
            state["data"].columns = state["data"].columns.str.strip().str.lower()
            print(f"CSV loaded successfully from: {data_path}")
            print("CSV columns:", state["data"].columns.tolist())
        except Exception as e:
            state["data"] = None
            print(f"Failed to load CSV: {e}")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_ready": state["art"] is not None,
            "data_loaded": state["data"] is not None,
        }

    @app.get("/sources")
    def get_sources() -> list[str]:
        df = state["data"]
        if df is None:
            return []

        if "origin" in df.columns:
            source_col = "origin"
        elif "source" in df.columns:
            source_col = "source"
        else:
            return []

        return sorted(df[source_col].dropna().astype(str).str.strip().unique().tolist())

    @app.get("/destinations")
    
    def get_destinations(origin: str | None = None) -> list[str]:
        df = state["data"]
        if df is None:
            return []

        if "destination" not in df.columns:
            return []

        temp = df.copy()

        if origin:
            if "origin" not in temp.columns:
                return []
            temp = temp[temp["origin"].astype(str).str.strip() == origin]

        return sorted(temp["destination"].dropna().astype(str).str.strip().unique().tolist())

    @app.post("/predict", response_model=TicketPriceResponse)

    def predict(req: TicketPriceRequest) -> TicketPriceResponse:
        if state["art"] is None:
            raise HTTPException(
                status_code=503,
                detail="Model not ready. Run training pipeline first.",
            )

        try:
            pred = predict_price_for_week(
                art=state["art"],
                origin=req.origin,
                destination=req.destination,
                travel_date=req.travel_date,
            )
            return TicketPriceResponse(**pred)

        except RouteNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.exception_handler(Exception)
    def unhandled_exception_handler(_, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error", "type": type(exc).__name__},
        )

    return app


app = create_app()