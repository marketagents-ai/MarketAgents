from typing import List, Dict, Tuple, Optional, Any, get_type_hints
from pydantic import BaseModel, create_model
import json
from market_agents.inference.sql_models import Tool, MessageRole, ChatMessage, CallableRegistry
import statistics

# Example Models
class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

class Stats(BaseModel):
    mean: float
    std: float

# Test Functions with different input/output patterns
def analyze_numbers_basemodel(input_data: NumbersInput) -> Stats:
    """Calculate statistical measures using BaseModel input and output."""
    return Stats(
        mean=round(statistics.mean(input_data.numbers), input_data.round_to),
        std=round(statistics.stdev(input_data.numbers), input_data.round_to)
    )

def analyze_numbers(numbers: List[float], round_to: int = 2) -> Stats:
    """Calculate statistical measures with regular args returning BaseModel."""
    return Stats(
        mean=round(statistics.mean(numbers), round_to),
        std=round(statistics.stdev(numbers), round_to)
    )

def analyze_numbers_dict(input_data: NumbersInput) -> Dict[str, float]:
    """Calculate statistical measures using BaseModel input, returning dict."""
    return {
        "mean": round(statistics.mean(input_data.numbers), input_data.round_to),
        "std": round(statistics.stdev(input_data.numbers), input_data.round_to)
    }

def analyze_numbers_tuple(numbers: List[float], round_to: int = 2) -> Tuple[float, float]:
    """Calculate statistical measures returning tuple (mean, std)."""
    return (
        round(statistics.mean(numbers), round_to),
        round(statistics.stdev(numbers), round_to)
    )

def test_type_patterns():
    """Test different input/output type patterns"""
    test_numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
    test_round_to = 3
    input_model = NumbersInput(numbers=test_numbers, round_to=test_round_to)
    
    print("\n=== Testing Type Patterns ===")
    
    # BaseModel -> BaseModel
    tool1 = Tool.from_callable(analyze_numbers_basemodel)
    result1 = tool1.execute(input_model.model_dump())
    parsed1 = Stats.model_validate_json(result1.content)
    print(f"BaseModel->BaseModel result: {parsed1}")
    verify_tool_integrity(tool1)
    
    # Regular args -> BaseModel
    tool2 = Tool.from_callable(analyze_numbers)
    result2 = tool2.execute({"numbers": test_numbers, "round_to": test_round_to})
    parsed2 = Stats.model_validate_json(result2.content)
    print(f"Regular->BaseModel result: {parsed2}")
    verify_tool_integrity(tool2)
    
    # BaseModel -> Dict
    tool3 = Tool.from_callable(analyze_numbers_dict)
    result3 = tool3.execute(input_model.model_dump())
    dict_result = json.loads(result3.content)
    print(f"BaseModel->Dict result: {dict_result}")
    verify_tool_integrity(tool3)
    
    # Regular args -> Tuple
    tool4 = Tool.from_callable(analyze_numbers_tuple)
    result4 = tool4.execute({"numbers": test_numbers, "round_to": test_round_to})
    tuple_result = json.loads(result4.content)
    print(f"Regular->Tuple result: {tuple_result}")
    verify_tool_integrity(tool4)
    
    return tool1, tool2, tool3, tool4

def test_registry_operations():
    """Test registry functionality"""
    print("\n=== Testing Registry Operations ===")
    registry = CallableRegistry()
    test_numbers = [1.0, 2.0, 3.0]
    input_model = NumbersInput(numbers=test_numbers)
    
    # Initial registration with custom name
    tool = Tool.from_callable(
        analyze_numbers_basemodel,
        schema_name="custom_analyzer"
    )
    assert registry.get("custom_analyzer") is not None, "Function not registered"
    result = tool.execute(input_model.model_dump())
    print(f"Initial execution: {result.content}")
    
    # Test duplicate registration
    try:
        Tool.from_callable(analyze_numbers_basemodel, schema_name="custom_analyzer")
        raise AssertionError("Should have raised ValueError for duplicate registration")
    except ValueError as e:
        print(f"Caught duplicate registration: {e}")
    
    # Test update via registry
    def updated_analyzer(input_data: NumbersInput) -> Stats:
        """Updated version with doubled mean."""
        return Stats(
            mean=round(statistics.mean(input_data.numbers) * 2, input_data.round_to),
            std=round(statistics.stdev(input_data.numbers), input_data.round_to)
        )
    
    registry.update("custom_analyzer", updated_analyzer)
    result_updated = tool.execute(input_model.model_dump())
    print(f"After update: {result_updated.content}")
    
    # Test delete
    registry.delete("custom_analyzer")
    assert registry.get("custom_analyzer") is None, "Function still in registry after deletion"
    try:
        tool.execute(input_model.model_dump())
        raise AssertionError("Should have raised ValueError for deleted function")
    except ValueError as e:
        print(f"Caught deleted function execution: {e}")

def test_literal_eval_and_validation():
    """Test literal eval functionality and schema validation"""
    print("\n=== Testing Lambda Functions ===")
    
    # Test typed lambda wrapper
    lambda_tool = Tool(
        schema_name="multiplier",
        callable=True,
        callable_function="""
def typed_lambda(x: float) -> float:
    \"\"\"Multiply input by 2\"\"\"
    return x * 2
""",
        allow_literal_eval=True,
        json_schema={
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"]
        }
    )
    result = lambda_tool.execute({"x": 5})
    print(f"Typed lambda result: {result.content}")
    
    # Test multi-argument typed function
    multi_arg_tool = Tool(
        schema_name="adder",
        callable=True,
        callable_function="""
def typed_adder(x: float, y: float) -> float:
    \"\"\"Add two numbers\"\"\"
    return x + y
""",
        allow_literal_eval=True,
        json_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"}
            },
            "required": ["x", "y"]
        }
    )
    result2 = multi_arg_tool.execute({"x": 5, "y": 3})
    print(f"Multi-argument typed result: {result2.content}")
    
    # Test list operation with type hints
    list_tool = Tool(
        schema_name="list_op",
        callable=True,
        callable_function="""
def typed_list_op(numbers: List[float]) -> List[float]:
    \"\"\"Double all numbers in list\"\"\"
    return [x * 2 for x in numbers]
""",
        allow_literal_eval=True,
        json_schema={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            },
            "required": ["numbers"]
        }
    )
    result3 = list_tool.execute({"numbers": [1, 2, 3]})
    print(f"Typed list operation result: {result3.content}")
    
    print("\n=== Testing Error Cases ===")
    
    # Test untyped function
    try:
        Tool(
            schema_name="untyped",
            callable=True,
            callable_function="def untyped(x): return x * 2",
            allow_literal_eval=True
        )
        raise AssertionError("Should have raised ValueError for untyped function")
    except ValueError as e:
        print(f"Caught untyped function: {e}")
    
    # Test function missing return type
    try:
        Tool(
            schema_name="missing_return",
            callable=True,
            callable_function="def missing_return(x: float): return x * 2",
            allow_literal_eval=True
        )
        raise AssertionError("Should have raised ValueError for missing return type")
    except ValueError as e:
        print(f"Caught missing return type: {e}")
    
    # Test duplicate registration
    try:
        Tool(
            schema_name="multiplier",
            callable=True,
            callable_function="def another(x: float) -> float: return x * 3",
            allow_literal_eval=True
        )
        raise AssertionError("Should have raised ValueError for duplicate name")
    except ValueError as e:
        print(f"Caught duplicate name: {e}")

    print("\nTyped function tests completed successfully")
def verify_tool_integrity(tool: Tool):
    """Verify tool properties and schema structure"""
    assert tool.schema_name is not None, "Missing schema name"
    assert tool.json_schema is not None, "Missing JSON schema"
    assert "type" in tool.json_schema, "Schema missing type"
    assert "properties" in tool.json_schema, "Schema missing properties"
    
    if tool.callable:
        assert tool.callable_function is not None, "Missing callable name"
        assert tool.callable_output_schema is not None, "Missing output schema"
        assert "type" in tool.callable_output_schema, "Output schema missing type"
        assert "properties" in tool.callable_output_schema, "Output schema missing properties"
        assert CallableRegistry().get(tool.schema_name) is not None, "Function not in registry"
        
    print(f"Tool {tool.schema_name} passed integrity check")

if __name__ == "__main__":
    # Run all tests
    test_type_patterns()
    test_registry_operations()
    test_literal_eval_and_validation()
    print("\n=== All tests completed successfully ===")