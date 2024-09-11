from pydantic import BaseModel, Field
from typing import List, Optional

class RandomNumberGenerator(BaseModel):
    num: int

class TestSchema(BaseModel):
    test: str = "schema"

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