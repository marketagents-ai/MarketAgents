import uuid
import logging
from typing import Optional, List, Union, Dict, Any
from pydantic import Field
from minference.lite.models import Entity

from minference.lite.models import (
    EntityRegistry,
    ChatThread,
    SystemPrompt,
    LLMConfig,
    ProcessedOutput,
    CallableTool,
    StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.prompter import PromptManager
from market_agents.agents.personas.persona import Persona

EntityRegistry()
logger = EntityRegistry._logger

class Agent(Entity):
    """
    Base LLM-driven agent using ChatThread-based inference.
    """

    name: str = Field(
        ...,
        description="Unique alphanumeric identifier for the agent (no spaces)"
    )
    persona: Union[str, Persona] = Field(
        default="You are a helpful AI agent",
        description="Agent's persona prompt as string or Persona object"
    )
    task: Optional[str] = Field(
        default=None,
        description="Primary tasks or instructions for the agent."
    )
    tools: List[Union[CallableTool, StructuredTool]] = Field(
        default_factory=list,
        description="List of callable or structured tools the agent can invoke."
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM configuration (model, client, response format, etc.)."
    )
    llm_orchestrator: InferenceOrchestrator = Field(
        ...,
        description="Inference orchestrator for parallel LLM requests."
    )
    prompt_manager: Optional[PromptManager] = Field(
        default=None,
        description="Manages YAML-based prompt assembly for system/user messages."
    )
    chat_thread: Optional[ChatThread] = Field(
        default=None,
        description="Holds conversation state, system prompt, and message history."
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def persona_prompt(self) -> str:
        """Get formatted persona as system prompt."""
        if isinstance(self.persona, str):
            return self.persona
        elif isinstance(self.persona, Persona):
            return self.prompt_manager.get_system_prompt({
                "role": self.persona.role,
                "persona": self.persona.persona,
                "objectives": self.persona.objectives,
                "skills": self.persona.skills
            })
        else:
            return str(self.persona)
            
    @property
    def task_prompt(self) -> str:
        """Get formatted task text."""
        if not self.task:
            return "observe environment"
            
        if self.prompt_manager:
            return self.prompt_manager.get_task_prompt({
                "task": self.task,
                "output_schema": None,
                "output_format": "text"
            })
        else:
            return self.task

    def __init__(self, **data: Any) -> None:
        """Initialize the agent and set up ChatThread from the prompt manager."""
        super().__init__(**data)

        if 'name' not in data or data['name'] is None:
            data['name'] = f"agent-{uuid.uuid4().hex[:8]}"

        if not self.llm_config:
            raise ValueError(
                "Agent requires an `llm_config` to specify which model/client to use."
            )

        if not self.prompt_manager:
            self.prompt_manager = PromptManager()

        system_prompt = SystemPrompt(
            name=self.name,
            content=self.persona_prompt
        )

        self.chat_thread = ChatThread(
            name=self.name,
            system_prompt=system_prompt,
            llm_config=self.llm_config,
            tools=self.tools,
            new_message=self.task_prompt
        )

    def _refresh_prompts(self) -> None:
        """
        Re-generate persona and task prompts,
        and update the chat thread with the latest versions.
        """
        logger.debug(f"Agent {self.name}: Refreshing prompts")

        if not self.chat_thread:
            logger.warning(f"Agent {self.name}: ChatThread not properly initialized")
            return

        # Update system prompt with latest persona_prompt
        if self.chat_thread.system_prompt:
            self.chat_thread.system_prompt.content = self.persona_prompt
            logger.debug(f"Agent {self.name}: Updated system prompt")

        # Update task message if task is set
        if self.task:
            self.chat_thread.new_message = self.task_prompt
            logger.debug(f"Agent {self.name}: Updated task prompt")

    async def execute(self) -> Union[str, Dict[str, Any]]:
        """
        Generate a new user prompt from tasks/persona and run an LLM completion.
        Returns either plain text (if LLM not forced to structured output)
        or a JSON object if the LLM yields a parsed structure.
        """
        if not self.chat_thread:
            raise RuntimeError("No ChatThread is available to run inference.")

        results = await self.llm_orchestrator.run_parallel_ai_completion([self.chat_thread])
        if not results:
            raise RuntimeError("No LLM outputs returned from orchestrator.")

        last_output: ProcessedOutput = results[-1]
        logger.info(f"Agent {self.id} received LLM output from {self.llm_config.client}.")

        if last_output.json_object:
            return last_output.json_object.object
        else:
            return last_output.content or ""