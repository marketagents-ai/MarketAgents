from typing import Optional, Dict, Any, List, Union
from enum import StrEnum
from pydantic import BaseModel, Field, ValidationError
import logging
import json
from datetime import datetime
from typing import Literal
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os
import requests
import time
from abstractions.inference.sql_models import Tool
import httpx
import asyncio

class ForgeError(Exception):
    """Custom exception for Forge API errors"""
    pass

class ReasoningSpeed(StrEnum):
    """Speed/depth of reasoning for the Forge pipeline"""
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"

class ReasoningType(StrEnum):
    """Type of reasoning pipeline"""
    PLAIN = "plain"  # Simple sequential reasoning
    MOCOCOA = "mococoa"  # Multi-model collaborative reasoning
    MCTS_MOA_COC = "mcts_moa_coc"  # Monte Carlo Tree Search with MOA/COC

class ForgeInput(BaseModel):
    """Input model for Forge API calls"""
    prompt: str = Field(..., description="The prompt to send to Forge")
    reasoning_speed: ReasoningSpeed = Field(
        default=ReasoningSpeed.MEDIUM, 
        description="Speed/depth of reasoning"
    )
    track: bool = Field(
        default=True, 
        description="Whether to return detailed trace information"
    )

async def call_forge_api(input_data: ForgeInput) -> 'ForgeResponse':
    """Makes a call to the Forge API with response validation. 
    Only returns when the task is complete."""
    load_dotenv()
    api_key = os.getenv("FORGE_API_KEY")
    if not api_key:
        logger.error("Please set env var FORGE_API_KEY to your personal API key.")
        raise ForgeError("FORGE_API_KEY environment variable not set")

    base_url = "https://forge-api.nousresearch.com/v1/asyncplanner/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async with httpx.AsyncClient() as client:
        try:
            # Convert to raw dict and manually set enum values to strings
            payload = {
                "prompt": input_data.prompt,
                "reasoning_speed": input_data.reasoning_speed.value,
                "track": input_data.track
            }
            
            logger.info("Initiating completion with settings:")
            logger.info(f"  Reasoning Speed: {payload['reasoning_speed']}")
            logger.info(f"  Trace Enabled: {payload['track']}")
            logger.info(f"  Prompt: {payload['prompt'][:100]}...")
            
            # Initial request to start the task
            post_response = await client.post(base_url, json=payload, headers=headers, timeout=10)
            post_response.raise_for_status()
            post_data = post_response.json()
            
            if 'task_id' not in post_data:
                raise ForgeError(f"No task_id in response: {post_data}")
            
            task_id = post_data['task_id']
            poll_url = f"{base_url}/{task_id}"
            start_time = time.time()
            
            # Keep polling until we get a final result
            while time.time() - start_time < 300:  # 5 minute timeout
                await asyncio.sleep(5)  # Non-blocking sleep
                
                poll_response = await client.get(poll_url, headers=headers, timeout=10)
                poll_response.raise_for_status()
                poll_data = poll_response.json()
                
                # Get status before validation
                polled_status = poll_data.get('metadata', {}).get('status', 'unknown')
                
                if polled_status == "succeeded":
                    try:
                        # Only return when we have a successful result
                        return poll_data
                    except ValidationError as e:
                        logger.error("Response validation failed:")
                        for error in e.errors():
                            logger.error(f"  - {'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}")
                        raise ForgeError("Failed to validate Forge response")
                
                elif polled_status in ["failed", "cancelled"]:
                    error_msg = f"Forge API task {polled_status}: {json.dumps(poll_data, indent=2)}"
                    logger.error(error_msg)
                    raise ForgeError(error_msg)
                
                logger.info(f"Status: {polled_status} (waited: {time.time() - start_time:.2f}s)")
            
            # If we exit the loop without returning, we timed out
            raise ForgeError("Forge API request timed out after 5 minutes")
                
        except Exception as e:
            logger.exception("Failed to complete Forge request")
            raise ForgeError(f"Failed to complete Forge request: {str(e)}")

class TaskStatus(StrEnum):
    """Status of a Forge task"""
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    IN_PROGRESS = "in_progress"

class Role(StrEnum):
    """Message role in the conversation"""
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"

class ModelConfig(BaseModel):
    """Configuration for an individual model in the pipeline"""
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    logprobs: bool = Field(default=False)
    n: int = Field(default=1)
    stop: Optional[str] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    logit_bias: Optional[Dict[str, float]] = None
    modelID: str

class ForgeConfig(BaseModel):
    """Configuration for the Forge reasoning pipeline"""
    reasoning_speed: ReasoningSpeed
    reasoning_type: ReasoningType
    orchestrator_model: ModelConfig
    reasoning_models: List[ModelConfig]
    num_simulations: int
    max_depth: int
    track: bool
    logprobs: Optional[bool] = None
    top_logprobs: Optional[Dict[str, float]] = None

class ForgeMetadata(BaseModel):
    """Metadata about the Forge API request and response"""
    status: TaskStatus
    id: str
    config: ForgeConfig
    error: Optional[str] = None
    updated_at: datetime
    owner: str
    task_id: str
    progress_message: Optional[str] = None
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime

class Message(BaseModel):
    """Message in the completion response"""
    content: str
    role: Role = Field(default=Role.ASSISTANT)

class Choice(BaseModel):
    """Individual completion choice"""
    message: Message
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None
    index: int = 0

class UsageMetrics(BaseModel):
    """Token usage metrics"""
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    time: int = Field(ge=0)

class TraceMetrics(BaseModel):
    """Metrics for trace data"""
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)
    total_duration: float = Field(default=0.0, ge=0.0)
    num_interactions: int = Field(default=0, ge=0)
    num_errors: int = Field(default=0, ge=0)
    models_used: List[str] = Field(default_factory=list)
    highest_input_tokens: int = Field(default=0, ge=0)
    highest_output_tokens: int = Field(default=0, ge=0)
    lowest_input_tokens: int = Field(default=0, ge=0)
    lowest_output_tokens: int = Field(default=0, ge=0)
    mean_input_tokens: float = Field(default=0.0, ge=0.0)
    mean_output_tokens: float = Field(default=0.0, ge=0.0)

class TraceStatus(StrEnum):
    """Status of trace execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TraceMetadata(BaseModel):
    """Metadata for trace information"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: TraceStatus = TraceStatus.IN_PROGRESS
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0"

class Traces(BaseModel):
    """Trace information for debugging and analysis"""
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: TraceMetadata
    metrics: TraceMetrics
    current_node_id: Optional[Union[str, int]] = None

class ForgeResult(BaseModel):
    """The main result object from the Forge API"""
    id: str
    object: str = Field(default="planner.chat.completion")
    created: int
    model: str
    usage: UsageMetrics
    choices: List[Choice]
    traces: Optional[Traces] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ForgeResponse(BaseModel):
    """Complete Forge API response"""
    metadata: ForgeMetadata
    result: ForgeResult
    test_config: Optional[Dict[str, Any]] = None

def parse_forge_response(response_json: str) -> ForgeResponse:
    """
    Parse and validate a Forge API response.
    
    Args:
        response_json: JSON string containing the Forge API response
        
    Returns:
        ForgeResponse: Validated response object
        
    Raises:
        ValidationError: If the response doesn't match the expected schema
        json.JSONDecodeError: If the input isn't valid JSON
    """
    try:
        # First try to parse the JSON
        try:
            data = json.loads(response_json) if isinstance(response_json, str) else response_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise

        # Then validate with Pydantic
        response = ForgeResponse(**data)
        logger.info(f"Successfully parsed Forge response (task_id: {response.metadata.task_id})")
        
        # Log some key metrics
        logger.debug(f"Status: {response.metadata.status}")
        logger.debug(f"Model: {response.result.model}")
        logger.debug(f"Total tokens: {response.result.usage.total_tokens}")
        
        return response

    except ValidationError as e:
        logger.error("Validation failed for Forge response")
        for error in e.errors():
            logger.error(f"Error at {'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}")
        raise

forge_tool = Tool.from_callable(call_forge_api)
