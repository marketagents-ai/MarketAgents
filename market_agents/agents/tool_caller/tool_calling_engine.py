import importlib.util
import os
from typing import Dict, Any, Callable, List, Union

from market_agents.inference.message_models import GeneratedJsonObject

# Declare type aliases
ValidParameter = Union[str, int, float, bool, dict, list]
ValidOutput = ValidParameter 

class ToolCallingEngine:
    """
    Engine to call functions extracted from LLM outputs in JSON format.
    Supports single-turn non-chained function calling in the OpenAI format.
    """
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.trace: List[Dict[str, Any]] = []

    def add_functions(self, functions: List[Callable]) -> None:
        """
        Add functions to the engine.

        Args:
            functions (List[Callable]): List of functions to be added to the engine.
        """
        for function in functions:
            self.functions[function.__name__] = function

    def add_functions_from_file(self, file_path: str) -> None:
        """
        Add functions to the engine from a specified .py file.

        Args:
            file_path (str): The path to the .py file containing the functions to be added.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        module_name = os.path.basename(file_path).split('.')[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name, obj in module.__dict__.items():
            if callable(obj) and not name.startswith("__"):
                self.functions[name] = obj

    def call_function(self, function_call: GeneratedJsonObject) -> ValidOutput:
        """
        Call a function from the engine.

        Args:
            function_call (GeneratedJsonObject): The function call to be executed.

        Returns:
            ValidOutput: The result of the function call.

        Raises:
            ValueError: If the function is not found or if there's an error in execution.
        """
        function_name = function_call.name
        if function_name not in self.functions:
            raise ValueError(f"Function '{function_name}' not found")

        try:
            result = self.functions[function_name](**function_call.object)
            self.trace.append({
                "function": function_name,
                "arguments": function_call.object,
                "result": result
            })
            return result
        except Exception as e:
            raise ValueError(f"Error executing function '{function_name}': {str(e)}")

    def parse_and_call_functions(self, function_calls: Union[GeneratedJsonObject, List[GeneratedJsonObject]]) -> List[ValidOutput]:
        """
        Parse and call either a single function call or a list of function calls.

        Args:
            function_calls (Union[GeneratedJsonObject, List[GeneratedJsonObject]]): The function call(s) to be parsed and called.

        Returns:
            List[ValidOutput]: A list of results from the function calls.
        """
        if isinstance(function_calls, GeneratedJsonObject):
            function_calls = [function_calls]

        results = []
        for call in function_calls:
            result = self.call_function(call)
            results.append(result)

        return results
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get the trace of function calls.

        Returns:
            List[Dict[str, Any]]: The trace of function calls.
        """
        return self.trace