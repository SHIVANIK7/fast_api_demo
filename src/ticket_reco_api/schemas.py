from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class TicketPriceRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=100)
    destination: str = Field(..., min_length=1, max_length=100)
    date_of_travel: date
    passengers: int = Field(1, ge=1, le=1, description="Demo supports exactly 1 passenger for now.")


class TicketPriceResponse(BaseModel):
    source: str
    destination: str
    predicted_price: float
    airline_name: str
    date_of_travel: date
