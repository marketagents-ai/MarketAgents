import importlib
import uuid
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

from base_agent.aiutilities import AIUtilities, LLMConfig
from base_agent.prompter import PromptManager
from base_agent.schemas import BaseSchema

agent_logger = logging.getLogger(__name__)


class Agent(BaseModel):
    """Base class for all agents in the multi-agent system.

    Attributes:
        id (str): Unique identifier for the agent.
        role (str): Role of the agent in the system.
        system (Optional[str]): System instructions for the agent.
        task (Optional[str]): Current task assigned to the agent.
        tools (Optional[Dict[str, Any]]): Tools available to the agent.
        output_format (Optional[Union[Dict[str, Any], str]]): Expected output format.
        llm_config (Dict[str, Any]): Configuration for the language model.
        max_retries (int): Maximum number of retry attempts for AI inference.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the agent.
        interactions (List[Dict[str, Any]]): History of agent interactions.

    Methods:
        execute(task: Optional[str] = None) -> str:
            Execute a task and return the result.
        _load_output_schema() -> None:
            Load the output schema based on the output_format.
        _prepare_messages(task: Optional[str]) -> List[Dict[str, str]]:
            Prepare messages for AI inference.
        _run_ai_inference(messages: List[Dict[str, str]]) -> str:
            Run AI inference with retry logic.
        _log_interaction(prompt: List[Dict[str, str]], response: str) -> None:
            Log an interaction between the agent and the AI.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    system: Optional[str] = None
    task: Optional[str] = None
    tools: Optional[Dict[str, Any]] = None
    output_format: Optional[Union[Dict[str, Any], str]] = None
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int = Field(default=2, ge=1)
    metadata: Optional[Dict[str, Any]] = None
    interactions: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.ai_utilities = AIUtilities()
        self._load_output_schema()

    def execute(self, task: Optional[str] = None) -> str:
        """Execute a task and return the result."""
        messages = self._prepare_prompt_messages(task)
        agent_logger.debug(f"Prepared messages:\n{json.dumps(messages, indent=2)}")

        return self._run_ai_inference(messages)

    def _load_output_schema(self) -> None:
        """Load the output schema based on the output_format."""
        if isinstance(self.output_format, str):
            try:
                schema_class = globals().get(self.output_format)
                if schema_class and issubclass(schema_class, BaseSchema):
                    self.output_format = schema_class.model_json_schema()
                else:
                    raise ValueError(f"Invalid schema: {self.output_format}")
            except ImportError:
                agent_logger.warning(f"Could not import schema: {self.output_format}")
        elif not isinstance(self.output_format, dict):
            self.output_format = None

    def _prepare_prompt_messages(self, task: Optional[str]) -> List[Dict[str, str]]:
        """Prepare messages for AI inference."""
        execution_task = task if task is not None else self.task
        prompt_manager = PromptManager(
            role=self.role,
            task=execution_task,
            resources=None,
            output_schema=self.output_format,
            char_limit=1000
        )

        messages = [
            {"role": "system", "content": prompt_manager.generate_system_prompt()}
        ]

        if execution_task:
            messages.append({
                "role": "user",
                "content": prompt_manager.generate_task_prompt()
            })

        return messages

    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(max_retries)
    )
    def _run_ai_inference(self, messages: List[Dict[str, str]]) -> str:
        """Run AI inference with retry logic."""
        try:
            agent_logger.info(f"Running inference with {self.llm_config.get('client')}")
            llm_config = LLMConfig(**self.llm_config)
            
            if self.tools:
                completion = self.ai_utilities.run_ai_tool_completion(messages, self.tools, llm_config)
            else:
                completion = self.ai_utilities.run_ai_completion(messages, llm_config)
            
            agent_logger.debug(f"Assistant Message:\n{completion}")
            messages.append({"role": "assistant", "content": completion})
            self._log_interaction(messages, completion)
        except Exception as e:
            agent_logger.error(f"Error during AI inference: {e}")
            raise

        return completion

    def _log_interaction(self, prompt: List[Dict[str, str]], response: str) -> None:
        """Log an interaction between the agent and the AI."""
        interaction = {
            "id": self.id,
            "name": self.role,
            "messages": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.interactions.append(interaction)
        agent_logger.debug(f"Agent Interaction logged:\n{json.dumps(interaction, indent=2)}")