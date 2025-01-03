from typing import Callable, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from market_agents.agents.tool_caller.utils import function_to_json
from market_agents.inference.message_models import GeneratedJsonObject

class Engine:
    """
    Tool calling engine.

    Args:
        tools: List of tools to call.
    """
    def __init__(self, tools: List[Callable] = []):
        self.tools = tools
        self.tools_map = {tool.__name__: tool for tool in tools}
        self.tools_json = self.convert_tools_to_json(tools)

    @classmethod
    def convert_tools_to_json(cls, tools: List[Callable]):
        """
        Convert the tools to a JSON-serializable format.

        Args:
            tools: List of tools to convert.
        """
        return [function_to_json(tool) for tool in tools]

    def add_tools(self, tools: List[Callable]):
        """
        Add tools to the engine.

        Args:
            tools: List of tools to add.
        """
        self.tools.extend(tools)
        self.tools_map.update({tool.__name__: tool for tool in tools})
        self.tools_json.extend(self.convert_tools_to_json(tools))

    def call_tool(self, tool_call: GeneratedJsonObject) -> Any:
        """
        Call a tool with the given name and arguments.

        Args:
            tool_call: GeneratedJsonObject containing tool name and arguments.
        """
        tool_name = tool_call.name
        arguments = tool_call.object
        print(f"Executing tool '{tool_name}' with arguments: {json.dumps(arguments, indent=2)}")
        try:
            tool = self.tools_map.get(tool_name)
            if tool:
                result = tool(**arguments)
                print(f"Tool '{tool_name}' execution completed successfully")
                return result
            else:
                raise ValueError(f"Tool '{tool_name}' not found")
        except Exception as e:
            print(f"Error executing tool '{tool_name}': {e}")
            raise ValueError(f"Error calling tool '{tool_name}': {e}")

    def execute_tool_calls(self, tool_calls: List[GeneratedJsonObject]) -> List[Any]:
        """
        Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of GeneratedJsonObject instances representing tool calls.

        Returns:
            List of results from the tool executions.
        """
        print(f"Executing {len(tool_calls)} tool calls in parallel")
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_tool_call = {
                executor.submit(self.call_tool, tool_call): tool_call for tool_call in tool_calls
            }
            for future in as_completed(future_to_tool_call):
                tool_call = future_to_tool_call[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error in parallel execution for tool '{tool_call.name}': {e}")
                    result = f"Error executing tool '{tool_call.name}': {e}"
                results.append(result)
        print("All tool calls completed")
        return results
