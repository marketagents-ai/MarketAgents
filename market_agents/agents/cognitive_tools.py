from minference.lite.models import StructuredTool
from market_agents.agents.cognitive_schemas import (
    PerceptionSchema, 
    ChainOfThoughtSchema,
    ReflectionSchema,
    ReActSchema,
)

perception_tool = StructuredTool.from_pydantic(
    model=PerceptionSchema,
    name="perception_tool",
    description="Analyze environment and generate perceptions"
)

reflection_tool = StructuredTool.from_pydantic(
    model=ReflectionSchema,
    name="reflection_tool",
    description="Reflect on previous steps and strategy"
)

chain_of_thought_tool = StructuredTool.from_pydantic(
    model=ChainOfThoughtSchema,
    name="chain_of_thought",
    description="Generate step-by-step reasoning with final answer"
)

react_tool = StructuredTool.from_pydantic(
    model=ReActSchema,
    name="react_reasoning",
    description="Generate thought-action-observation cycle"
)