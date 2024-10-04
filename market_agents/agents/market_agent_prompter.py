from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import yaml
import os

class AgentPromptVariables(BaseModel):
    environment_name: str
    environment_info: Any
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)
    perception: Optional[Any] = None
    observation: Optional[Any] = None
    action_space: Dict[str, Any] = {}
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
        
        prompt_template = self.prompts[prompt_type]
        return prompt_template.format(**variables)

    def get_perception_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('perception', variables)

    def get_action_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('action', variables)

    def get_reflection_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('reflection', variables)