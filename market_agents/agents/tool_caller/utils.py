import inspect
import json
from typing import Callable

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam
)

from openai.types.shared_params import (
    ResponseFormatText,
    ResponseFormatJSONObject,
    FunctionDefinition
)

def function_to_json(func) -> ChatCompletionToolParam:
    """
    Converts a Python function into a ChatCompletionToolParam
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A ChatCompletionToolParam representing the function's signature.
    """
    type_map = {
        str: "string",
        int: "integer", 
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name=func.__name__,
            description=func.__doc__ or "",
            parameters={
                "type": "object",
                "properties": parameters,
                "required": required,
            }
        )
    )