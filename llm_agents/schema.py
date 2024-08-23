from pydantic import BaseModel, Field
from typing import Literal, Optional

class RandomNumberGenerator(BaseModel):
    num: int

class TestSchema(BaseModel):
    test: str = "schema"

class BidSchema(BaseModel):
    bid_action: Literal["buy", "sell"] = Field(..., description="The type of action: 'buy' for purchasing or 'sell' for offering goods")
    price: float = Field(..., description="The proposed price for the bid or ask")
    quantity: int = Field(default=1, description="The quantity of goods to buy or sell")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level in the proposed bid (0-1)")
    reasoning: str = Field(..., description="Brief explanation for the proposed bid")

class MarketActionSchema(BaseModel):
    action: Literal["bid", "ask", "hold"] = Field(..., description="The market action to take: 'bid' for buying, 'ask' for selling, or 'hold' for no action")
    bid: BidSchema = Field(None, description="Bid details if action is 'bid' or 'ask'")
    reasoning: str = Field(..., description="Explanation for the chosen action")