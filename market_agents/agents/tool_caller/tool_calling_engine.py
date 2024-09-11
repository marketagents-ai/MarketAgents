import importlib.util
import os
from typing import Dict, Any, Callable, List, Optional, Union
import json

from pydantic import BaseModel

# Declare type aliases
ValidParameter = Union[str, int, float, bool, dict, list, BaseModel]
ValidOutput = ValidParameter

# Declare constants
INVALID_FUNCTION_CALL_ERROR = """
The function call is invalid.
"""

class Parameter(BaseModel):
    """
    Schema for the parameters of a function
    schema and function call. Used in the fields
    parameters and returns.

    name: str
        The name of the parameter.
    type: str
        The type of the parameter.
    """
    name: str
    type: str

class FunctionCall(BaseModel):
    """
    Schema for the function call.

    name: str
        The name of the function.
    parameters: dict[str, ValidParameter]
        The parameters of the function.
    returns: Optional[list[Parameter]]
        The return values of the function. 
        
        If the used function call format does not 
        support the "returns" field, this schema 
        will still be valid.

    """
    name: str
    parameters: dict[str, ValidParameter]
    returns: Optional[list[Parameter]] = None

class OpenAIFunction(BaseModel):
    """
    Schema for the OpenAI function format.

    arguments: Dict[str, ValidParameter]
        The arguments of the function.
    name: str
        The name of the function.
    """
    arguments: dict[str, ValidParameter]
    name: str

class OpenAIToolCall(BaseModel):
    """
    Schema for the OpenAI tool call format.

    id: str
        The ID of the tool call.
    function: OpenAIFunction
        The function to be called.
    type: str
        The type of the tool call.
    """
    id: str
    function: OpenAIFunction
    type: str
    
class ToolCallingEngine:
    """
    Wrapper class for the FunctionCallingEngine. 
    Processes the LLM output and executes the tool calls.
    """
    def __init__(self):
        self.engine = FunctionCallingEngine()
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, function: Callable):
        """
        Register a tool with its schema and function.

        Args:
            name (str): The name of the tool.
            function (Callable): The function to be called when the tool is used.
        """
        self.tools[name] = function
        self.engine.add_functions([function])

    def register_tools(self, tools: List[Callable]):
        """
        Register a list of tools with their schemas and functions.

        Args:
            tools (List[Callable]): A list of tools to be registered.
        """
        for tool in tools:
            self.register_tool(tool.__name__, tool)

    def parse_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[FunctionCall]:
        """
        Parse tool calls in OpenAIToolCall format.

        Args:
            tool_calls (List[Dict[str, Any]]): A list of tool calls in OpenAIToolCall format.

        Returns:
            List[FunctionCall]: A list of parsed FunctionCall objects.
        """
        function_calls = []
        try:
            for tool_call in tool_calls:
                try:
                    parameters = json.loads(tool_call['function']['arguments'])
                    function_call = FunctionCall(
                        name=tool_call['function']['name'],
                        parameters=parameters,
                        returns=None
                    )
                    function_calls.append(function_call)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in function arguments: {str(e)}")
            return function_calls
        except KeyError as e:
            raise ValueError(f"Invalid tool call format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing tool calls: {str(e)}")

    def process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, ValidOutput]]:
        """
        Process tool calls and execute them.

        Args:
            tool_calls (List[Dict[str, Any]]): A list of tool calls in OpenAIToolCall format.

        Returns:
            List[Dict[str, ValidOutput]]: A list of tool calls and their results.
        """
        try:
            function_calls = self.parse_tool_calls(tool_calls)
            results: List[Dict[str, ValidOutput]] = []
            # Enumerate through the function calls
            for i, function_call in enumerate(function_calls):
                result = self.execute_tool_call(function_call)
                results.append({
                    "tool_name": function_call.name,
                    "tool_call_id": tool_calls[i]['id'],    
                    "result": result
                })
            return results
        except ValueError as e:
            raise ValueError(f"Error processing tool calls: {str(e)}")

    def execute_tool_call(self, tool_call: FunctionCall) -> ValidOutput:
        """
        Execute a tool call and return the result.

        Args:
            tool_call (FunctionCall): The tool call to execute.

        Returns:
            ValidOutput: The result of the tool call.
        """
        tool_name = tool_call.name
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        return self.engine.call_function(tool_call)

# Sourced from AtakanTekparmak/tiny_fnc_engine
# https://github.com/AtakanTekparmak/tiny_fnc_engine/blob/main/tiny_fnc_engine/engine.py
class FunctionCallingEngine:
    """
    Engine to call functions extracted 
    from LLM outputs in JSON format in
    an isolated environment. The engine
    will store the functions and their
    outputs in memory. 
    """
    def __init__(self):
        self.functions: dict[str, callable] = {}
        self.outputs: dict[str, ValidOutput] = {}

    def reset_session(self) -> None:
        """
        Reset the session of the engine.
        """
        self.outputs = {}
    
    def add_functions(self, functions: list[callable]) -> None:
        """
        Add functions to the engine.

        functions: list[callable]
            List of functions to be added to the engine.
        """
        for function in functions:
            self.functions[function.__name__] = function

    def add_functions_from_file(self, file_path: str) -> None:
        """
        Add functions to the engine from a specified .py file.

        file_path: str
            The path to the .py file containing the functions to be added.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        # Use importlib.util to load the module
        module_name = os.path.basename(file_path).split('.')[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the user defined functions
        for name, obj in module.__dict__.items():
            if callable(obj) and not name.startswith("__") and name != "add_functions_from_file":
                self.functions[name] = obj

    def call_function(self, function_call: FunctionCall) -> ValidOutput:
        """
        Call a function from the engine.

        function_call: FunctionCall
            The function call to be executed.
        """
        # Get the function and its parameters
        function = self.functions[function_call.name]
        parameters = function_call.parameters  # This is already a dict, no need to process it

        # Check if any of the parameters are outputs from previous functions
        for key, value in parameters.items():
            if isinstance(value, str) and value in self.outputs:
                parameters[key] = self.outputs[value]

        # Call the function
        output = function(**parameters)  # Use ** to unpack the dictionary as keyword arguments

        # Store the output
        if function_call.returns:
            if len(function_call.returns) == 1:
                self.outputs[function_call.returns[0].name] = output
            else:
                for i, return_value in enumerate(function_call.returns):
                    self.outputs[return_value.name] = output[i]
        
        return output
    
    def call_functions(self, function_calls: list[FunctionCall]) -> list[ValidOutput]:
        """
        Call multiple functions from the engine.

        function_calls: list[FunctionCall]
            The function calls to be executed.
        """
        outputs = []
        for function_call in function_calls:
            output = self.call_function(function_call)
            outputs.append(output)
        return outputs
    
    def _convert_openai_tool_call(self, tool_call: dict) -> FunctionCall:
        """
        Convert an OpenAI tool call to a FunctionCall.

        tool_call: dict
            The OpenAI tool call to be converted.
        """
        function = tool_call['function']
        arguments = json.loads(function['arguments'])
        return FunctionCall(
            name=function['name'],
            parameters=arguments,
            returns=None  # OpenAI tool calls do not specify returns
        )

    def parse_function_calls(self, function_calls: Union[dict, list[dict]]) -> list[FunctionCall]:
        """
        Parse either a single function call or
        a list of function calls.

        function_calls: Union[dict, list[dict]]
            The function call(s) to be parsed.
        """
        if isinstance(function_calls, dict):
            function_calls = [function_calls]
        elif not isinstance(function_calls, list):
            raise TypeError("Input must be a dictionary or a list of dictionaries")

        parsed_calls = []
        for call in function_calls:
            if 'id' in call and 'function' in call and 'type' in call:
                parsed_calls.append(self._convert_openai_tool_call(call))
            else:
                try:
                    parsed_calls.append(FunctionCall(**call))
                except Exception:
                    raise ValueError(INVALID_FUNCTION_CALL_ERROR)
        
        return parsed_calls

    def parse_and_call_functions(
            self, 
            function_calls: Union[dict, list[dict], str],
            verbose: bool = False
        ) -> list[ValidOutput]:
        """
        Parse and call either a single function call or
        a list of function calls.

        function_calls: Union[dict, list[dict]]
            The function call(s) to be parsed and called.
        """
        if isinstance(function_calls, str): 
            function_calls = json.loads(function_calls)

        function_calls = self.parse_function_calls(function_calls)

        if verbose:
            for function_call in function_calls:
                print(f"Calling function: {function_call.name}")
                print(f"Parameters: {function_call.parameters}")
                print(f"Returns: {function_call.returns}")

        return self.call_functions(function_calls)