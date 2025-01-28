from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

class PerceptionSchema(BaseModel):
    monologue: str = Field(..., description="Agent's internal monologue about the perceived environment")
    key_observations: List[str] = Field(..., description="Agent's key observations from the environment")
    strategy: List[str] = Field(..., description="Agent's strategies given the current environment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

class ReflectionSchema(BaseModel):
    reflection: str = Field(..., description="Reflection on the observation and actions")
    self_critique: List[str]= Field(..., description="Self-critique of agent's strategies and actions on the environment")
    self_reward: float = Field(..., description="Self-assigned reward between 0.0 and 1.0")
    strategy_update: List[str] = Field(..., description="Updated strategies based on the reflection and previous strategy")
