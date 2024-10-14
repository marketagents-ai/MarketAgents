from typing import Callable
from openai.types.chat import ChatCompletionToolParam

from market_agents.agents.tool_caller.utils import function_to_json

class Engine:
    """
    Tool calling engine.

    Args:
        tools: List of tools to call.
    """
    def __init__(self, tools: list[Callable] = []):
        self.tools = tools
        self.tools_json = self.convert_tools_to_json(tools)

    @classmethod
    def convert_tools_to_json(cls, tools: list[Callable]):
        """
        Convert the tools to a JSON-serializable format.

        Args:
            tools: List of tools to convert.
        """
        return [function_to_json(tool) for tool in tools]

    def add_tools(self, tools: list[Callable]):
        """
        Add a tool to the engine.

        Args:
            tool: Tool to add.
        """
        self.tools.extend(tools)
        self.tools_json.extend(self.convert_tools_to_json(tools))

    def call_tool(self, tool: ChatCompletionToolParam):
        """
        Call a tool with the given name and arguments.

        Args:
            tool_name: Name of the tool to call.
            args: Arguments to pass to the tool.
        """
        try:
            tool = self.tools.get(tool.function.name)
            if tool:
                return tool(**tool.function.parameters)
            else:
                raise ValueError(f"Tool {tool.function.name} not found")
        except Exception as e:
            raise ValueError(f"Error calling tool {tool.function.name}: {e}")
