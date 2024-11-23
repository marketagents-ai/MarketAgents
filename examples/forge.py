import os
import asyncio
import aiohttp
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ForgeInput(BaseModel):
    """Input model for Forge API calls"""
    prompt: str = Field(..., description="The prompt to send to Forge")
    reasoning_speed: Literal["fast", "medium", "slow"] = Field(
        default="medium", 
        description="Speed/depth of reasoning"
    )
    track: bool = Field(
        default=True, 
        description="Whether to return detailed trace information"
    )
    timeout: Optional[int] = Field(
        default=600,
        description="Maximum time to wait for completion in seconds"
    )
    max_poll_interval: Optional[int] = Field(
        default=30,
        description="Maximum time between polling attempts in seconds"
    )
    """Input model for Forge API calls"""
    prompt: str = Field(..., description="The prompt to send to Forge")
    reasoning_speed: Literal["fast", "medium", "slow"] = Field(
        default="medium", 
        description="Speed/depth of reasoning"
    )
    track: bool = Field(
        default=True, 
        description="Whether to return detailed trace information"
    )

class ForgeMetadata(BaseModel):
    """Metadata about the Forge API response"""
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    reasoning_steps: Optional[int] = None
    total_time: Optional[float] = None

class ForgeOutput(BaseModel):
    """Output model for Forge API responses"""
    completion: str
    metadata: ForgeMetadata
    trace: Optional[Dict[str, Any]] = None

class ForgeError(Exception):
    """Custom exception for Forge API errors"""
    pass

async def call_forge_api(input_data: ForgeInput) -> ForgeOutput:
    """
    Makes an async call to the Forge API with retry logic and timeout handling
    
    Args:
        input_data: ForgeInput model containing the request parameters
        
    Returns:
        ForgeOutput model containing the response and metadata
        
    Raises:
        ForgeError: If the API call fails after retries or times out
    """
    load_dotenv()
    api_key = os.getenv("FORGE_API_KEY")
    if not api_key:
        raise ForgeError("FORGE_API_KEY environment variable not set")

    base_url = "https://forge-api.nousresearch.com/v1/asyncplanner/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async def poll_completion(session: aiohttp.ClientSession, task_id: str) -> Dict[str, Any]:
        """Polls the completion endpoint until done or timeout"""
        poll_url = f"{base_url}/{task_id}"
        last_status_time = 0
        status_interval = 10  # Print detailed status every 10 seconds
        start_time = time.time()
        poll_failures = 0
        
        while time.time() - start_time < 600:  # 10 minute timeout
            try:
                async with session.get(poll_url, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    status = data.get("metadata", {}).get("status", "unknown")
                    if status == "succeeded":
                        return data
                    elif status in ["failed", "cancelled"]:
                        raise ForgeError(f"Forge API task {status}: {json.dumps(data)}")
                    
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Print detailed status at intervals
                    if current_time - last_status_time >= status_interval:
                        detailed_status = {
                            "status": status,
                            "elapsed_time": f"{elapsed:.1f}s",
                            "reasoning_steps": data.get("metadata", {}).get("reasoning_steps", "unknown"),
                            "processing_rate": data.get("metadata", {}).get("processing_rate", "unknown")
                        }
                        logger.info(f"Detailed Status: {json.dumps(detailed_status, indent=2)}")
                        last_status_time = current_time
                    else:
                        logger.debug(f"Status: {status}, waiting... ({elapsed:.1f}s elapsed)")
                    poll_failures = 0
                    
            except Exception as e:
                poll_failures += 1
                if poll_failures > 5:
                    raise ForgeError(f"Failed to poll completion after {poll_failures} attempts")
                logger.warning(f"Polling attempt {poll_failures} failed: {str(e)}")
                
            # Progressive backoff for polling
            poll_interval = min(30, 5 + (time.time() - start_time) / 10)  # Gradually increase up to 30s
            logger.info(f"Waiting {poll_interval:.1f}s before next poll...")
            await asyncio.sleep(poll_interval)
            
        raise ForgeError("Timeout waiting for Forge API completion")

    try:
        async with aiohttp.ClientSession() as session:
            # Initial request to start the task
            payload = input_data.model_dump(exclude_none=True)
            async with session.post(base_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                post_data = await response.json()
                
                if "task_id" not in post_data:
                    raise ForgeError(f"No task_id in response: {post_data}")
                
                task_id = post_data["task_id"]
                logger.info(f"Task initiated: {task_id}")
                
                # Poll until completion
                result = await poll_completion(session, task_id)
                
                return ForgeOutput(
                    completion=result.get("completion", ""),
                    metadata=ForgeMetadata(**result.get("metadata", {})),
                    trace=result.get("trace") if input_data.track else None
                )
                
    except Exception as e:
        raise ForgeError(f"Forge API call failed: {str(e)}")

async def test_forge():
    """Test function to demonstrate usage"""
    # Configure longer timeout for test
    test_input = ForgeInput(
        prompt="What are three creative ways to use a paperclip? Explain your reasoning for each.",
        reasoning_speed="fast",
        track=True
    )
    
    try:
        result = await call_forge_api(test_input)
        print("\nForge API Response:")
        print(f"Status: {result.metadata.status}")
        print(f"Time taken: {result.metadata.total_time:.1f}s")
        print(f"\nCompletion:\n{result.completion}")
        
        if result.trace:
            print("\nReasoning Trace:")
            print(json.dumps(result.trace, indent=2))
            
    except ForgeError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_forge())