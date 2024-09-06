import uuid
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

from base_agent.aiutilities import AIUtilities, LLMConfig, LLMPromptContext, StructuredTool, LLMOutput
from base_agent.prompter import PromptManager
from base_agent.utils import extract_json_from_response
from base_agent.schemas import *

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
        llm_config (LLMConfig): Configuration for the language model.
        max_retries (int): Maximum number of retry attempts for AI inference.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the agent.
        interactions (List[Dict[str, Any]]): History of agent interactions.

    Methods:
        execute(task: Optional[str] = None, output_format: Optional[Union[Dict[str, Any], str]] = None) -> Union[str, Dict[str, Any]]:
            Execute a task and return the result.
        _load_output_schema(output_format: Optional[Union[Dict[str, Any], str]]) -> Optional[Dict[str, Any]]:
            Load the output schema based on the output_format.
        _prepare_prompt_context(task: Optional[str], output_format: Optional[Dict[str, Any]]) -> LLMPromptContext:
            Prepare LLMPromptContext for AI inference.
        _run_ai_inference(prompt_context: LLMPromptContext) -> Union[str, Dict[str, Any]]:
            Run AI inference with retry logic.
        _log_interaction(prompt: LLMPromptContext, response: Union[str, Dict[str, Any]]) -> None:
            Log an interaction between the agent and the AI.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    system: Optional[str] = None
    task: Optional[str] = None
    tools: Optional[Dict[str, Any]] = None
    output_format: Optional[Union[Dict[str, Any], str]] = None
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    max_retries: int = 2
    metadata: Optional[Dict[str, Any]] = None
    interactions: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        extra = "allow"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.ai_utilities = AIUtilities()

    def execute(self, task: Optional[str] = None, output_format: Optional[Union[Dict[str, Any], str]] = None) -> Union[str, Dict[str, Any]]:
        """Execute a task and return the result."""
        execution_task = task if task is not None else self.task
        execution_output_format = output_format if output_format is not None else (self.output_format or "plain_text")
        execution_output_format = self._load_output_schema(execution_output_format)
        
        # Update llm_config based on output_format
        if execution_output_format is None or execution_output_format == "text":
            self.llm_config.response_format = "text"
        else:
            self.llm_config.response_format = "json_object"
        
        prompt_context = self._prepare_prompt_context(execution_task, execution_output_format)
        agent_logger.debug(f"Prepared LLMPromptContext:\n{json.dumps(prompt_context.model_dump(), indent=2)}")

        return self._run_ai_inference(prompt_context)

    def _load_output_schema(self, output_format: Optional[Union[Dict[str, Any], str]] = None) -> Optional[Dict[str, Any]]:
        """Load the output schema based on the output_format."""
        if output_format is None:
            output_format = self.output_format

        if isinstance(output_format, str):
            try:
                schema_class = globals().get(output_format)
                if schema_class and issubclass(schema_class, BaseModel):
                    return schema_class.model_json_schema()
                else:
                    raise ValueError(f"Invalid schema: {output_format}")
            except (ImportError, AttributeError, ValueError) as e:
                agent_logger.warning(f"Could not load schema: {output_format}. Error: {str(e)}")
                return None
        elif isinstance(output_format, dict):
            return output_format
        else:
            return None

    def _prepare_prompt_context(self, task: Optional[str], output_format: Optional[Dict[str, Any]] = None) -> LLMPromptContext:
        """Prepare LLMPromptContext for AI inference."""
        prompt_manager = PromptManager(
            role=self.role,
            task=task,
            resources=None,
            output_schema=output_format,
            char_limit=1000
        )

        prompt_messages = prompt_manager.generate_prompt_messages()
        system_message = prompt_messages["messages"][0]["content"]
        user_message = prompt_messages["messages"][1]["content"]

        structured_output = None
        if output_format:
            structured_output = StructuredTool(json_schema=output_format)

        return LLMPromptContext(
            system_string=system_message,
            new_message=user_message,
            llm_config=self.llm_config,
            structured_output=structured_output
        )
    
    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(max_retries)
    )
    def _run_ai_inference(self, prompt_context: LLMPromptContext) -> Union[str, Dict[str, Any]]:
        """Run AI inference with retry logic."""
        try:
            agent_logger.info(f"Running inference with {prompt_context.llm_config.client}")
            
            if self.tools:
                completion = self.ai_utilities.run_ai_tool_completion(prompt_context)
            else:
                completion = self.ai_utilities.run_ai_completion(prompt_context)
            
            llm_output = LLMOutput(raw_result=completion.raw_result)
            agent_logger.debug(f"Assistant Message:\n{llm_output}")
            
            if prompt_context.structured_output:
                if llm_output.json_object:
                    result = llm_output.json_object.object
                elif llm_output.str_content:
                    try:
                        result = json.loads(llm_output.str_content)
                    except json.JSONDecodeError:
                        result = extract_json_from_response(llm_output.str_content)
                else:
                    result = {}
            else:
                result = llm_output.str_content or str(llm_output.raw_result)
            
            self._log_interaction(prompt_context, result)
            return result
        except Exception as e:
            agent_logger.error(f"Error during AI inference: {e}")
            raise

    def _log_interaction(self, prompt: LLMPromptContext, response: Union[str, Dict[str, Any]]) -> None:
        """Log an interaction between the agent and the AI."""
        interaction = {
            "id": self.id,
            "name": self.role,
            "prompt_context": prompt.model_dump(),
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.interactions.append(interaction)
        agent_logger.debug(f"Agent Interaction logged:\n{json.dumps(interaction, indent=2)}")