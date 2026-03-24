from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from movie_reco_api.recommender import (
    ModelNotReadyError,
    RouteNotFoundError,
    load_artifacts,
    predict_ticket,
)
from movie_reco_api.schemas import TicketPriceRequest, TicketPriceResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="Flight Ticket Price Prediction API",
        version="0.1.0",
        description="Predict ticket price from source, destination, travel date, and passenger count.",
    )

    artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
    state = {"art": None}

    @app.on_event("startup")
    def _startup() -> None:
        try:
            state["art"] = load_artifacts(artifacts_dir)
        except ModelNotReadyError:
            state["art"] = None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_ready": state["art"] is not None}

    @app.post("/predict", response_model=TicketPriceResponse)
    def predict(req: TicketPriceRequest) -> TicketPriceResponse:
        if state["art"] is None:
            raise HTTPException(status_code=503, detail="Model not ready. Train it first.")
        try:
            pred = predict_ticket(
                art=state["art"],
                source=req.source,
                destination=req.destination,
                date_of_travel=req.date_of_travel,
                passengers=req.passengers,
            )
        except RouteNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return TicketPriceResponse(**pred)

    @app.exception_handler(Exception)
    def unhandled_exception_handler(_, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error", "type": type(exc).__name__},
        )

    return app


app = create_app()
