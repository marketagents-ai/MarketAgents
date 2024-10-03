import uuid
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import StructuredTool, LLMConfig, LLMPromptContext, LLMOutput
from market_agents.agents.base_agent.prompter import PromptManager
from market_agents.agents.base_agent.utils import extract_json_from_response
from market_agents.agents.base_agent.schemas import *

agent_logger = logging.getLogger(__name__)


class Agent(BaseModel):
    """Base class for all agents in the multi-agent system.

    Attributes:
        id (str): Unique identifier for the agent.
        role (str): Role of the agent in the system.
        persona(str): Personal characteristics of the agent.
        system (Optional[str]): System instructions for the agent.
        task (Optional[str]): Current task assigned to the agent.
        tools (Optional[Dict[str, Any]]): Tools available to the agent.
        output_format (Optional[Union[Dict[str, Any], str]]): Expected output format.
        llm_config (LLMConfig): Configuration for the language model.
        max_retries (int): Maximum number of retry attempts for AI inference.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the agent.
        interactions (List[Dict[str, Any]]): History of agent interactions.

    Methods:
        execute(task: Optional[str] = None, output_format: Optional[Union[Dict[str, Any], str]] = None, return_prompt: bool = False) -> Union[str, Dict[str, Any], LLMPromptContext]:
            Execute a task and return the result or the prompt context.
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
    persona: Optional[str] = None
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
        self.ai_utilities = ParallelAIUtilities()

    async def execute(self, task: Optional[str] = None, output_format: Optional[Union[Dict[str, Any], str, Type[BaseModel]]] = None, return_prompt: bool = False) -> Union[str, Dict[str, Any], LLMPromptContext]:
        """Execute a task and return the result or the prompt context."""
        execution_task = task if task is not None else self.task
        if execution_task is None:
            raise ValueError("No task provided. Agent needs a task to execute.")
        
        execution_output_format = output_format if output_format is not None else self.output_format
        
        # Update llm_config based on output_format
        if execution_output_format == "plain_text":
            self.llm_config.response_format = "text"
        else:
            self.llm_config.response_format = "structured_output"
            execution_output_format = self._load_output_schema(execution_output_format)

        prompt_context = self._prepare_prompt_context(execution_task, execution_output_format)
        agent_logger.debug(f"Prepared LLMPromptContext:\n{json.dumps(prompt_context.model_dump(), indent=2)}")

        if return_prompt:
            return prompt_context

        result = await self._run_ai_inference(prompt_context)
        self._log_interaction(prompt_context, result)
        return result

    def _load_output_schema(self, output_format: Optional[Union[Dict[str, Any], str, Type[BaseModel]]] = None) -> Optional[Dict[str, Any]]:
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
            except (AttributeError, ValueError) as e:
                agent_logger.warning(f"Could not load schema: {output_format}. Error: {str(e)}")
                return None
        elif isinstance(output_format, dict):
            return output_format
        elif isinstance(output_format, type) and issubclass(output_format, BaseModel):
            return output_format.model_json_schema()
        else:
            return None

    def _prepare_prompt_context(self, task: Optional[str], output_format: Optional[Dict[str, Any]] = None) -> LLMPromptContext:
        """Prepare LLMPromptContext for AI inference."""
        prompt_manager = PromptManager(
            role=self.role,
            persona=self.persona,
            task=task,
            resources=None,
            output_schema=output_format,
            char_limit=1000
        )

        prompt_messages = prompt_manager.generate_prompt_messages()
        system_message = prompt_messages["messages"][0]["content"]
        if self.system:
            system_message += f"\n{self.system}"
        user_message = prompt_messages["messages"][1]["content"]
       
        structured_output = None
        if output_format and isinstance(output_format, dict):
            structured_output = StructuredTool(json_schema=output_format, strict_schema=False)

        return LLMPromptContext(
            id=self.id,
            system_string=system_message,
            new_message=user_message,
            llm_config=self.llm_config,
            structured_output=structured_output
        )
    
    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(2),  # Limit to 2 attempts
        reraise=True  # Reraise the last exception
    )
    async def _run_ai_inference(self, prompt_context: LLMPromptContext) -> Any:
        try:
            llm_output = await self.ai_utilities.run_parallel_ai_completion([prompt_context])
            
            if not llm_output:
                raise ValueError("No output received from AI inference")
            
            llm_output = llm_output[0]  # Get the first (and only) output
            
            if prompt_context.llm_config.response_format == "text":
                return llm_output.str_content or str(llm_output.raw_result)
            elif prompt_context.llm_config.response_format in ["json_beg", "json_object", "structured_output"]:
                if llm_output.json_object:
                    return llm_output.json_object.object
                elif llm_output.str_content:
                    try:
                        return json.loads(llm_output.str_content)
                    except json.JSONDecodeError:
                        return extract_json_from_response(llm_output.str_content)
            elif prompt_context.llm_config.response_format == "tool":
                # Handle tool response format if needed
                pass
            
            # If no specific handling or parsing failed, return the raw output
            agent_logger.warning(f"No parsing logic for response format '{prompt_context.llm_config.response_format}'. Returning raw output.")
            return llm_output.raw_result
        
        except Exception as e:
            agent_logger.error(f"Error during AI inference: {e}")
            raise

    def _log_interaction(self, prompt: LLMPromptContext, response: Union[str, Dict[str, Any]]) -> None:
        """Log an interaction between the agent and the AI."""
        interaction = {
            "id": self.id,
            "name": self.role,
            "system": prompt.system_message,
            "task": prompt.new_message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.interactions.append(interaction)
        agent_logger.debug(f"Agent Interaction logged:\n{json.dumps(interaction, indent=2)}")