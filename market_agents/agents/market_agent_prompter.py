from typing import Dict, Any, Optional, List
import importlib.resources as ires
from pydantic import Field
from pathlib import Path

from market_agents.agents.base_agent.prompter import (
    PromptVariables,
    PromptTemplate,
    PromptManager
)

class MarketAgentPromptVariables(PromptVariables):
    """Variables specific to market agent prompts."""
    environment_name: str = Field(
        ...,
        description="Name of the market environment")
    environment_info: Any = Field(
        ...,
        description="Information about the market environment")
    short_term_memory: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Recent observations and actions"
    )
    long_term_memory: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Historical market data and patterns"
    )
    documents: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Reference documents and market analysis"
    )
    perception: Optional[Any] = Field(
        default=None,
        description="Current market perception"
    )
    observation: Optional[Any] = Field(
        default=None,
        description="Latest market observation"
    )
    action_space: Dict[str, Any] = Field(
        default_factory=dict,
        description="Available market actions"
    )
    last_action: Optional[Any] = Field(
        default=None,
        description="Previously taken action"
    )
    reward: Optional[float] = Field(
        default=None,
        description="Reward from last action"
    )
    previous_strategy: Optional[str] = Field(
        default=None,
        description="Previously used trading strategy"
    )

    class Config:
        arbitrary_types_allowed = True

class MarketAgentPromptTemplate(PromptTemplate):
    """Template loader for market agent prompts."""
    template_path: Path = Field(
        default_factory=lambda: Path(ires.files("market_agents.agents.configs.prompts") / "market_agent_prompt.yaml"),
        description="Path to market agent prompt templates"
    )

class MarketAgentPromptManager(PromptManager):
    """Manages prompts for market agents with specialized template handling."""
    
    def __init__(self, template_paths: Optional[List[Path]] = None):
        default_paths = [
            Path(ires.files("market_agents.agents.configs.prompts") / "market_agent_prompt.yaml")
        ]
        super().__init__(template_paths=template_paths if template_paths else default_paths)

    def get_perception_prompt(self, variables: Dict[str, Any]) -> str:
        """Generate perception analysis prompt."""
        vars_model = MarketAgentPromptVariables(**variables)
        template_vars = vars_model.get_template_vars()
        
        # Get perception template
        perception_template = self.template.templates.get('perception', '')
        
        # Format template with variables
        return perception_template.format(**template_vars)

    def get_action_prompt(self, variables: Dict[str, Any]) -> str:
        """Generate action selection prompt."""
        vars_model = MarketAgentPromptVariables(**variables)
        template_vars = vars_model.get_template_vars()
        
        # Get action template
        action_template = self.template.templates.get('action', '')
        
        # Format template with variables
        return action_template.format(**template_vars)

    def get_reflection_prompt(self, variables: Dict[str, Any]) -> str:
        """Generate strategy reflection prompt."""
        vars_model = MarketAgentPromptVariables(**variables)
        template_vars = vars_model.get_template_vars()
        
        # Get reflection template
        reflection_template = self.template.templates.get('reflection', '')
        
        # Format template with variables
        return reflection_template.format(**template_vars)