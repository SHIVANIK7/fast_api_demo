from pydantic import BaseModel, Field


class TicketPriceRequest(BaseModel):
    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    travel_date: str = Field(..., min_length=1)


class TicketPriceResponse(BaseModel):
    origin: str
    destination: str
    date: str
    predicted_price: float
    airline: str