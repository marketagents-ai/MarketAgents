from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

class ACLContent(BaseModel):
    action: Literal["bid", "ask"]
    price: float = Field(..., description="The proposed price for the bid or ask")
    quantity: int = Field(default=1, description="The quantity of goods to buy or sell")

class DoubleAuctionBid(BaseModel):
    reasoning: str = Field(..., description="Detailed reasoning behind the decision")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level in the proposed action (0-1)")
    acl_message: ACLContent = Field(...)

class MarketActionSchema(BaseModel):
    thought: str = Field(..., description="Explanation for the chosen action")
    action: Literal["bid", "ask", "hold"] = Field(..., description="The market action to take")
    bid: Optional[DoubleAuctionBid] = Field(None, description="Bid details if action is 'bid' or 'ask'")

class DoubleAuctionMessage(BaseModel):
    action: Literal["bid", "ask"]
    price: float
    quantity: int

class ReflectionSchema(BaseModel):
    reflection: str = Field(..., description="Reflection on the observation and surplus based on the last action")
    strategy_update: str = Field(..., description="Updated strategy based on the reflection, surplus, and previous strategy")

class PerceptionSchema(BaseModel):
    monologue: str = Field(..., description="Agent's internal monologue about the perceived market situation")
    strategy: str= Field(..., description="Agent's strategy given the current market situation")