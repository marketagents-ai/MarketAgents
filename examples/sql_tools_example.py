from typing import List, Dict, Tuple, Optional, Any, get_type_hints
from pydantic import BaseModel, create_model
import json
from market_agents.inference.sql_models import Tool, MessageRole, ChatMessage

# Example BaseModel for inputs
class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

# Example BaseModel for outputs
class Stats(BaseModel):
    mean: float
    std: float

# Example functions with different input/output types

# Case 1: BaseModel -> BaseModel
def analyze_numbers_basemodel(input_data: NumbersInput) -> Stats:
    """Calculate statistical measures using BaseModel input and output."""
    import statistics
    return Stats(
        mean=round(statistics.mean(input_data.numbers), input_data.round_to),
        std=round(statistics.stdev(input_data.numbers), input_data.round_to)
    )

# Case 2: Regular args -> BaseModel
def analyze_numbers(numbers: List[float], round_to: int = 2) -> Stats:
    """Calculate statistical measures with regular args returning BaseModel."""
    import statistics
    return Stats(
        mean=round(statistics.mean(numbers), round_to),
        std=round(statistics.stdev(numbers), round_to)
    )

# Case 3: BaseModel -> Dict
def analyze_numbers_dict(input_data: NumbersInput) -> Dict[str, float]:
    """Calculate statistical measures using BaseModel input, returning dict."""
    import statistics
    return {
        "mean": round(statistics.mean(input_data.numbers), input_data.round_to),
        "std": round(statistics.stdev(input_data.numbers), input_data.round_to)
    }

# Case 4: Regular args -> Tuple
def analyze_numbers_tuple(numbers: List[float], round_to: int = 2) -> Tuple[float, float]:
    """Calculate statistical measures returning tuple (mean, std)."""
    import statistics
    return (
        round(statistics.mean(numbers), round_to),
        round(statistics.stdev(numbers), round_to)
    )

def test_all_cases():
    test_numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
    test_round_to = 3
    
    print("\n=== Testing all input/output combinations ===\n")
    
    # Case 1: BaseModel -> BaseModel
    print("Case 1: BaseModel -> BaseModel")
    tool1 = Tool.from_callable(analyze_numbers_basemodel)
    print(f"Input schema: {json.dumps(tool1.json_schema, indent=2)}")
    print(f"Output schema: {json.dumps(tool1.callable_output_schema, indent=2)}")
    
    input_model = NumbersInput(numbers=test_numbers, round_to=test_round_to)
    result1 = tool1.execute(input_model.model_dump())
    print(f"Result content: {result1.content}")
    parsed1 = Stats.model_validate_json(result1.content)
    print(f"Parsed result: {parsed1}\n")
    
    # Case 2: Regular args -> BaseModel
    print("Case 2: Regular args -> BaseModel")
    tool2 = Tool.from_callable(analyze_numbers)
    print(f"Input schema: {json.dumps(tool2.json_schema, indent=2)}")
    print(f"Output schema: {json.dumps(tool2.callable_output_schema, indent=2)}")
    
    result2 = tool2.execute({"numbers": test_numbers, "round_to": test_round_to})
    print(f"Result content: {result2.content}")
    parsed2 = Stats.model_validate_json(result2.content)
    print(f"Parsed result: {parsed2}\n")
    
    # Case 3: BaseModel -> Dict
    print("Case 3: BaseModel -> Dict")
    tool3 = Tool.from_callable(analyze_numbers_dict)
    print(f"Input schema: {json.dumps(tool3.json_schema, indent=2)}")
    print(f"Output schema: {json.dumps(tool3.callable_output_schema, indent=2)}")
    
    input_model = NumbersInput(numbers=test_numbers, round_to=test_round_to)
    result3 = tool3.execute(input_model.model_dump())
    print(f"Result content: {result3.content}")
    result_dict = json.loads(result3.content)["result"]
    parsed3 = Stats(**result_dict)
    print(f"Converted to Stats: {parsed3}\n")
    
    # Case 4: Regular args -> Tuple
    print("Case 4: Regular args -> Tuple")
    tool4 = Tool.from_callable(analyze_numbers_tuple)
    print(f"Input schema: {json.dumps(tool4.json_schema, indent=2)}")
    print(f"Output schema: {json.dumps(tool4.callable_output_schema, indent=2)}")
    
    result4 = tool4.execute({"numbers": test_numbers, "round_to": test_round_to})
    print(f"Result content: {result4.content}")
    tuple_data = json.loads(result4.content)["result"]
    parsed4 = Stats(mean=tuple_data[0], std=tuple_data[1])
    print(f"Converted to Stats: {parsed4}\n")
    
    # Test schema validation
    print("Testing schema validation")
    bad_schema = {
        "type": "object",
        "properties": {
            "values": {"type": "array"}  # wrong parameter name
        }
    }
    try:
        tool_bad = Tool.from_callable(analyze_numbers, json_schema=bad_schema)
    except ValueError as e:
        print(f"Validation error (as expected): {e}")
    return result1,result2,result3,result4

if __name__ == "__main__":
    result1,result2,result3,result4 = test_all_cases()