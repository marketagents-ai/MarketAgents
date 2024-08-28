from tiny_fnc_engine import FunctionCallingEngine, FunctionCall, ValidOutput
from typing import Dict, Any, Callable, List
import json

class ToolCallingEngine:
    """
    A class that processes LLM output, identifies tool calls, and executes them.
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

    def extract_tool_calls(self, llm_output: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM output."""
        return llm_output # TODO: Implement this

    def parse_llm_output(self, llm_output: str) -> List[FunctionCall]:
        """
        Parse LLM output and identify tool calls.

        Args:
            llm_output (str): The output from the LLM.

        Returns:
            List[FunctionCall]: A list of tool calls.
        """
        try:
            tool_calls_str = self.extract_tool_calls(llm_output)
            return self.engine.parse_function_calls(tool_calls_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in LLM output")

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

    def process_llm_output(self, llm_output: str) -> List[Dict[str, ValidOutput]]:
        """
        Process LLM output, identify tool calls, and execute them.

        Args:
            llm_output (str): The output from the LLM.

        Returns:
            List[Dict[str, ValidOutput]]: A list of tool calls and their results.
        """
        try:
            tool_calls = self.parse_llm_output(llm_output)
            results: List[Dict[str, ValidOutput]] = []
            for tool_call in tool_calls:
                result = self.execute_tool_call(tool_call)
                results.append({
                    "tool_name": tool_call.name,
                    "result": result
                })
            return results
        except ValueError as e:
            raise ValueError(f"Error processing LLM output: {str(e)}")

