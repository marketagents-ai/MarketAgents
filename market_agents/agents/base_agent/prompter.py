from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

class BasePromptVariables(BaseModel):
    """Base class for prompt variables with common utility methods."""
    
    def format_value(self, value: Any) -> str:
        """Format a value for template insertion."""
        if value is None:
            return "N/A"
        elif isinstance(value, (dict, list)):
            return yaml.dump(value, default_flow_style=False).strip() or "No entries"
        return str(value)

    def get_template_vars(self) -> Dict[str, str]:
        """Get formatted variables for template substitution."""
        return {
            key: self.format_value(value)
            for key, value in self.model_dump().items()
        }

class SystemPromptVariables(BasePromptVariables):
    """Variables for system prompt template."""
    role: str = Field(
        ...,
        description="Functional role of the agent")
    persona: Optional[str] = Field(
        default=None,
        description="Agent's persona")
    objectives: Optional[str] = Field(
        default=None,
        description="Agent's objectives")
    datetime: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Current timestamp"
    )

class TaskPromptVariables(BasePromptVariables):
    """Variables for task prompt template."""
    task: Union[str, List[str]] = Field(
        ...,
        description="Task instructions")
    output_schema: Optional[str] = Field(
        default=None,
        description="Output schema")
    output_format: str = Field(
        default="text",
        description="Expected output format ('text' or 'json_object')"
    )

class BasePromptTemplate(BaseSettings):
    """Base class for loading and managing prompt templates."""
    
    template_path: Path = Field(
        default=Path("configs/prompts/default_prompt.yaml"),
        description="Path to the YAML template file."
    )
    templates: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of loaded prompt templates."
    )
    
    model_config = SettingsConfigDict(
        env_prefix="PROMPT_",
        arbitrary_types_allowed=True
    )
    
    def model_post_init(self, _) -> None:
        """Load templates after initialization."""
        try:
            with open(self.template_path) as f:
                self.templates = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template file not found: {self.template_path}")

    def format_prompt(self, prompt_type: str, variables: BasePromptVariables) -> str:
        """Format a specific prompt type with variables."""
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        template = self.templates[prompt_type]
        return template.format(**variables.get_template_vars())

class PromptTemplate(BasePromptTemplate):
    """Template loader for default agent prompts."""
    template_path: Path = Field(
        default=Path("configs/prompts/default_prompt.yaml"),
        description="Default path to prompt template file."
    )
    
class PromptManager:
    """Manages prompts for agents with template-based generation."""
    
    def __init__(self, template_path: Optional[Path] = None):
        self.template = PromptTemplate(
            template_path=template_path if template_path else Path("market_agents/agents/configs/prompts/default_prompt.yaml")
        )

    def get_system_prompt(self, variables: Dict[str, Any]) -> str:
        """Generate system prompt."""
        vars_model = SystemPromptVariables(**variables)
        return self.template.format_prompt('system', vars_model)

    def get_task_prompt(self, variables: Dict[str, Any]) -> str:
        """Generate task prompt."""
        vars_model = TaskPromptVariables(**variables)
        return self.template.format_prompt('task', vars_model)