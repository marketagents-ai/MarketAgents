from typing import Dict, Optional, List, TypedDict, Literal
from pydantic import BaseModel, Field, ValidationError, validator
import re

# Define the schema classes
class FunctionParameters(BaseModel):
    type: str = Field(..., description="The type of the parameters (should be 'object').")
    properties: Optional[Dict[str, Dict]] = Field(default_factory=dict, description="The properties of the parameters.")
    required: Optional[List[str]] = Field(default_factory=list, description="The required properties.")

    @validator("type")
    def type_must_be_object(cls, v):
        if v != "object":
            raise ValueError("The type must be 'object'.")
        return v

class FunctionDefinition(BaseModel):
    name: str = Field(..., max_length=64, description="The name of the function to be called.")
    description: Optional[str] = Field(None, description="A description of what the function does.")
    parameters: Optional[FunctionParameters] = Field(None, description="The parameters the function accepts as a JSON Schema object.")

    @validator("name")
    def validate_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", v):
            raise ValueError("Name must be 1-64 characters long and can contain letters, numbers, underscores, and dashes.")
        return v

class ChatCompletionToolParam(TypedDict, total=False):
    function: FunctionDefinition
    type: Literal["function"]

# Helper function to convert complex types to valid JSON Schema
def clean_type(type_str: str) -> str:
    """Convert and clean the type string."""
    type_str = type_str.split(",")[0].strip().lower()  # Remove 'optional' and any extra commas or spaces
    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean"
    }
    return type_map.get(type_str, type_str)  # Default to itself if not in type_map

# Helper function to convert and validate tools
def fix_tools_format_oai(tool: Dict[str, object]) -> Dict[str, object]:
    def convert_enum_to_list(prop_data: Dict[str, object]) -> None:
        if "enum" in prop_data and isinstance(prop_data["enum"], list):
            prop_data["enum"] = list(prop_data["enum"])

    def handle_complex_type(type_str: str) -> Dict[str, object]:
        """Handle complex types and convert to JSON Schema format."""
        if type_str.startswith("list["):
            inner_type = type_str[5:-1].strip()  # Extract type inside list[]
            return {"type": "array", "items": handle_complex_type(inner_type)}
        elif type_str.startswith("tuple["):
            inner_types = type_str[6:-1].strip().split(",")
            return {"type": "array", "items": [handle_complex_type(t) for t in inner_types]}
        elif type_str.startswith("union["):
            inner_types = type_str[6:-1].strip().split(",")
            return {"oneOf": [handle_complex_type(t) for t in inner_types]}
        else:
            return {"type": clean_type(type_str)}

    if "type" not in tool or tool["type"] != "function":
        parameters = tool.get("parameters", {})
        properties = {}
        required = []

        for key, value in parameters.items():
            type_value = clean_type(value["type"])

            # Check for complex types
            prop_data = handle_complex_type(value["type"])

            prop_data["description"] = value.get("description", "")

            if "enum" in value:
                prop_data["enum"] = value["enum"]

            properties[key] = prop_data

            # Add to required list only if not optional
            if "optional" not in value["type"]:
                required.append(key)

        fixed_parameters = {"type": "object"}
        if properties:
            fixed_parameters["properties"] = properties
        if required:
            fixed_parameters["required"] = required

        function_definition = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": fixed_parameters if fixed_parameters else None,
        }

        try:
            validated_function = FunctionDefinition(**function_definition)
            return {
                "type": "function",
                "function": validated_function.dict(),
            }
        except ValidationError as e:
            print(f"Validation error: {e}")
            raise ValueError(f"Invalid tool definition: {e}")

    else:
        parameters = tool.get("function", {}).get("parameters", {})
        properties = parameters.get("properties", {})
        required = []

        for key, value in properties.items():
            type_value = clean_type(value["type"])

            # Check for complex types
            prop_data = handle_complex_type(value["type"])

            prop_data["description"] = value.get("description", "")

            if "enum" in value:
                prop_data["enum"] = value["enum"]

            properties[key] = prop_data

            if "optional" not in value["type"]:
                required.append(key)

        function_definition = tool["function"]
        function_definition["parameters"]["required"] = required
        
        try:
            validated_function = FunctionDefinition(**function_definition)
            tool["function"] = validated_function.dict()
            return tool
        except ValidationError as e:
            print(f"Validation error: {e}")
            raise ValueError(f"Invalid tool definition: {e}")