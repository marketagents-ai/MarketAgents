from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

class ReflectionSchema(BaseModel):
    reflection: str = Field(..., description="Reflection on the observation and actions")
    strategy_update: List[str] = Field(..., description="Updated strategies based on the reflection and previous strategy")
    self_reward: float = Field(..., description="Self-assigned reward between 0.0 and 1.0")

class PerceptionSchema(BaseModel):
    monologue: str = Field(..., description="Agent's internal monologue about the perceived market situation")
    strategy: List[str] = Field(..., description="Agent's strategies given the current market situation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")