from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
import importlib.resources as ires

class PromptVariables(BaseModel):
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

class SystemPromptVariables(PromptVariables):
    """Variables for system prompt template."""
    role: str = Field(
        ...,
        description="Functional role of the agent")
    persona: Optional[str] = Field(
        default=None,
        description="Agent's persona")
    objectives: Optional[List[str]] = Field(
        default=None,
        description="Agent's objectives")
    skills: Optional[Union[List[str], List[Dict[str, str]]]] = Field(
        default_factory=list,
        description="Agent's skills as either list of strings or list of dicts")
    datetime: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Current timestamp")

class TaskPromptVariables(PromptVariables):
    """Variables for task prompt template."""
    task: Union[str, List[str]] = Field(
        ...,
        description="Task instructions")
    output_format: Optional[str] = Field(
        default="text",
        description="Expected output format")
    output_schema: Optional[str] = Field(
        default=None,
        description="Output schema")

class PromptTemplate(BaseSettings):
    """Template loader for agent prompts."""
    template_paths: List[Path] = Field(
        default_factory=lambda: [
            Path(ires.files("market_agents.agents.configs.prompts") / "default_prompt.yaml")
        ],
        description="Paths to the YAML template files."
    )
    templates: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Dictionary of loaded prompt templates."
    )
    
    model_config = SettingsConfigDict(
        env_prefix="PROMPT_",
        arbitrary_types_allowed=True
    )
    
    def model_post_init(self, _) -> None:
        """Load and merge templates from all template paths."""
        self.templates = {}
        for path in self.template_paths:
            try:
                with open(path) as f:
                    new_templates = yaml.safe_load(f)
                    if new_templates:
                        self.templates.update(new_templates)
            except FileNotFoundError:
                raise FileNotFoundError(f"Prompt template file not found: {path}")

class PromptManager:
    """Manages prompts for agents with template-based generation."""
    
    def __init__(self, template_paths: Optional[List[Path]] = None):
        self.template = PromptTemplate(
            template_paths=template_paths if template_paths else [
                Path(ires.files("market_agents.agents.configs.prompts") / "default_prompt.yaml")
            ]
        )

    def get_system_prompt(self, variables: Dict[str, Any]) -> str:
        """Generate system prompt from persona template sections."""
        vars_model = SystemPromptVariables(**variables)
        template_vars = vars_model.get_template_vars()
        
        # Get persona template sections
        persona_template = self.template.templates.get('persona', {})
        
        # Format each section
        sections = []
        for section, template in persona_template.items():
            formatted = template.format(**template_vars)
            sections.append(formatted)
            
        # Join all sections with newlines
        return "\n".join(sections)

    def get_task_prompt(self, variables: Dict[str, Any]) -> str:
        """Generate task prompt from task template sections."""
        vars_model = TaskPromptVariables(**variables)
        template_vars = vars_model.get_template_vars()
        
        # Get task template sections
        task_template = self.template.templates.get('task', {})
        
        # Format each section
        sections = []
        for section, template in task_template.items():
            # Only include sections that have corresponding variables
            if any(var in template for var in template_vars):
                formatted = template.format(**template_vars)
                sections.append(formatted)
            
        # Join all sections with newlines
        return "\n".join(sections)