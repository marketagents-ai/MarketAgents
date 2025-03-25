import uuid
import logging
from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field

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

EntityRegistry()
agent_logger = logging.getLogger(__name__)

class Agent(BaseModel):
    """
    Base LLM-driven agent using ChatThread-based inference.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique string identifier for the agent instance."
    )
    role: str = Field(
        ...,
        description="Functional role of the agent (e.g., 'financial analyst')."
    )
    persona: Optional[str] = Field(
        default=None,
        description="Additional persona or background info for the agent."
    )
    objectives: Optional[List[str]] = Field(
        default=None,
        description="High-level goals or objectives for the agent."
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

    def __init__(self, **data: Any) -> None:
        """Initialize the agent and set up ChatThread from the prompt manager."""
        super().__init__(**data)

        if not self.llm_config:
            raise ValueError(
                "Agent requires an `llm_config` to specify which model/client to use."
            )

        if not self.prompt_manager:
            self.prompt_manager = PromptManager()

        system_str = self.prompt_manager.get_system_prompt({
            "role": self.role,
            "persona": self.persona,
            "objectives": self.objectives
        })
        system_prompt = SystemPrompt(
            name=f"SystemPrompt_{self.id}",
            content=system_str
        )

        initial_message = self.prompt_manager.get_task_prompt({
            "task": self.task or "observe environment",
            "output_schema": None,
            "output_format": "text"
        })

        self.chat_thread = ChatThread(
            name=f"ChatThread_{self.id}",
            system_prompt=system_prompt,
            llm_config=self.llm_config,
            tools=self.tools,
            new_message=initial_message
        )

    def _refresh_prompts(self) -> None:
        """
        Re-generate system and user prompts from PromptManager,
        and place the user prompt into `chat_thread.new_message`.
        """
        agent_logger.debug("Refreshing prompts via PromptManager.")

        if not self.prompt_manager or not self.chat_thread:
            agent_logger.warning("PromptManager or ChatThread not properly initialized.")
            return

        system_str = self.prompt_manager.get_system_prompt({
            "role": self.role,
            "persona": self.persona,
            "objectives": self.objectives
        })
        if self.chat_thread.system_prompt:
            self.chat_thread.system_prompt.content = system_str

        if self.task:
            task_str = self.prompt_manager.get_task_prompt({
                "task": self.task,
                "output_schema": None,
                "output_format": "text"
            })
            self.chat_thread.new_message = task_str

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
        agent_logger.info(f"Agent {self.id} received LLM output from {self.llm_config.client}.")

        if last_output.json_object:
            return last_output.json_object.object
        else:
            return last_output.content or ""