import json
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import yaml
import os

def json_to_markdown(data: Union[Dict, List, Any], indent: int = 0) -> str:
    """Convert JSON/dict data to a markdown formatted string."""
    if data is None:
        return "None"
    
    if isinstance(data, str):
        # Clean up newlines and escape characters for markdown compatibility
        return data.replace('\\n', '\n').replace('\\', '')
    
    if isinstance(data, (int, float, bool)):
        return str(data)
    
    indent_str = "  " * indent
    if isinstance(data, list):
        if not data:
            return "None"
        markdown = ""
        for item in data:
            item_str = json_to_markdown(item, indent + 1)
            markdown += f"{indent_str}- {item_str}\n"
        return markdown.rstrip()
    
    if isinstance(data, dict):
        if not data:
            return "None"
        markdown = ""
        for key, value in data.items():
            value_str = json_to_markdown(value, indent + 1)
            if isinstance(value, (dict, list)) and value:
                markdown += f"{indent_str}{key}:\n{value_str}\n"
            else:
                markdown += f"{indent_str}{key}: {value_str}\n"
        return markdown.rstrip()
    
    # For any other types, convert to string
    return str(data)


class AgentPromptVariables(BaseModel):
    environment_name: str
    environment_info: Any
    short_term_memory: Optional[List[Dict[str, Any]]] = None
    long_term_memory: Optional[List[Dict[str, Any]]] = None
    perception: Optional[Any] = None
    observation: Optional[Any] = None
    action_space: Dict[str, Any] = Field(default_factory=dict)
    last_action: Optional[Any] = None
    reward: Optional[float] = None
    previous_strategy: Optional[str] = None

class MarketAgentPromptManager(BaseModel):
    prompts: Dict[str, str] = Field(default_factory=dict)
    prompt_file: str = Field(default="market_agents/agents/configs/prompts/market_agent_prompt.yaml")

    def __init__(self, **data: Any):
        super().__init__(**data)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        full_path = os.path.join(project_root, self.prompt_file)
        
        try:
            with open(full_path, 'r') as file:
                self.prompts = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        
    def format_prompt(self, prompt_type: str, variables: Dict[str, Any]) -> str:
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Convert empty values to N/A and format JSON/dict values as markdown
        formatted_vars = {}
        for key, value in variables.items():
            if value is None or (isinstance(value, (list, dict)) and not value):
                formatted_vars[key] = "N/A"
            elif isinstance(value, (dict, list)):
                # Convert dict/list values to markdown format
                formatted_vars[key] = json_to_markdown(value).strip()
            else:
                formatted_vars[key] = str(value) if value else "N/A"
        
        try:
            return self.prompts[prompt_type].format(**formatted_vars)
        except KeyError as e:
            raise KeyError(f"Missing required variable in prompt: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt: {e}")

    def get_perception_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('perception', variables)

    def get_action_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('action', variables)

    def get_reflection_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('reflection', variables)