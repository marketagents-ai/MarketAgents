from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

from pydantic import BaseModel, Field
from typing import List, Optional

class ThoughtStep(BaseModel):
    reasoning: str = Field(description="The agent's reasoning for this step")

class ChainOfThoughtSchema(BaseModel):
    thoughts: List[ThoughtStep] = Field(description="The agent's step-by-step reasoning process")
    final_answer: str = Field(description="The final answer after the reasoning process")

class Action(BaseModel):
    name: str = Field(description="The name of the action to take")
    input: Optional[str] = Field(description="The input for the specified action, if any")

class ReActSchema(BaseModel):
    thought: str = Field(description="The agent's reasoning about the current state and what to do next")
    action: Optional[Action] = Field(description="The action to take, if any")
    observation: Optional[str] = Field(description="The result of the action taken")
    final_answer: Optional[str] = Field(description="The final answer, if the task is complete")

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
