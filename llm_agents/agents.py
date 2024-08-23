import importlib
import uuid
import json
from datetime import datetime
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

from aiutilities import AIUtilities
from prompter import PromptManager
from schema import *
from utils import agent_logger

class Agent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    system: Optional[str] = None
    task: Optional[str] = None
    tools: Optional[Dict[str, Any]] = None
    output_format: Optional[Union[Dict[str, Any], str]] = None
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    max_iter: int = Field(default=2, ge=1)
    metadata: Optional[Dict[str, Any]] = None
    interactions: list = Field(default_factory=list)

    class Config:
        extra = "allow"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.ai_utilities = AIUtilities()
        if isinstance(self.output_format, str):
            try:
                self.output_format = self.load_output_schema()
            except ImportError:
                pass

    def execute(self, task: Optional[str] = None) -> str:
        messages = []

        execution_task = task if task is not None else self.task
        prompt_manager = PromptManager(
            role=self.role,
            task=execution_task,
            resources=None,
            output_schema=self.output_format,
            char_limit=1000
        )

        system_prompt = prompt_manager.generate_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        if execution_task:
            user_message = prompt_manager.generate_task_prompt()
            messages.append({"role": "user", "content": user_message})
        
        agent_logger.info(f"Logging prompt text\n{messages}")

        @retry(
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(self.max_iter)
        )
        def run_ai_inference() -> str:
            try:
                agent_logger.info(f"Running inference with {self.llm_config.get('client')}")
                if self.tools:
                    # TODO: @interstellarninja Implement tool calling with a dedicated tool calling engine
                    raise NotImplementedError("Tool calling is not implemented yet.")
                else:
                    completion = self.ai_utilities.run_ai_completion(messages, self.llm_config)
                    agent_logger.info(f"Assistant Message:\n{completion}")
                    messages.append({"role": "assistant", "content": completion})
                    self.log_interaction(messages, completion)
            except Exception as e:
                agent_logger.error(e)
                raise

            return completion

        return run_ai_inference()
    
    def load_output_schema(self) -> Dict[str, Any]:
            if not isinstance(self.output_format, str):
                raise ValueError("output_format must be a string")
            
            schema_class = globals().get(self.output_format)
            if not schema_class or not issubclass(schema_class, BaseModel):
                raise ValueError(f"Invalid schema: {self.output_format}")
            
            return schema_class.schema_json()

    def log_interaction(self, prompt, response):
        self.interactions.append({
            "id": self.id,
            "name": self.role,
            "messages": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        agent_logger.info(f"Agent Interaction logged:\n{json.dumps(self.interactions[-1], indent=2)}")