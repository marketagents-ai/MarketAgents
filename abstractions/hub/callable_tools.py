from typing import List
from pydantic import BaseModel

# Example BaseModel for inputs
class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

# Example BaseModel for outputs
class Stats(BaseModel):
    mean: float
    std: float

class MinimumAnalysis(BaseModel):
    minimum: float

class MaximumAnalysis(BaseModel):
    maximum: float

# Example functions with different input/output types
def analyze_numbers_basemodel(input_data: NumbersInput) -> Stats:
    """Calculate statistical measures using BaseModel input and output."""
    import statistics
    return Stats(
        mean=round(statistics.mean(input_data.numbers), input_data.round_to),
        std=round(statistics.stdev(input_data.numbers), input_data.round_to)
    )

def analyze_minimum_basemodel(input_data: NumbersInput) -> MinimumAnalysis:
    """Find the minimum value in a list of numbers."""
    return MinimumAnalysis(minimum=min(input_data.numbers))

def analyze_maximum_basemodel(input_data: NumbersInput) -> MaximumAnalysis:
    """Find the maximum value in a list of numbers."""
    return MaximumAnalysis(maximum=max(input_data.numbers))

# Dictionary of default callable tools
DEFAULT_CALLABLE_TOOLS = {
    "analyze_stats": {
        "function": analyze_numbers_basemodel,
        "description": "Calculate statistical measures (mean and standard deviation) for a list of numbers",
        "allow_literal_eval": True
    },
    "find_minimum": {
        "function": analyze_minimum_basemodel,
        "description": "Find the minimum value in a list of numbers",
        "allow_literal_eval": True
    },
    "find_maximum": {
        "function": analyze_maximum_basemodel,
        "description": "Find the maximum value in a list of numbers",
        "allow_literal_eval": True
    }
} 