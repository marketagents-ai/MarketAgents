import os
import requests
from typing import Optional, List, cast
from dotenv import load_dotenv
import json
import time
import logging
import sys
from datetime import datetime
from itertools import product
from abstractions.hub.forge import (
    ForgeResponse, ForgeInput, ReasoningSpeed, 
    ForgeError, TaskStatus, TraceStatus, call_forge_api
)
from pydantic import ValidationError

# Set up logging
def setup_logging(output_dir: str = "forge_responses") -> None:
    """Configure logging to both file and console"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"forge_test_{timestamp}.log")
    
    # Create formatters and handlers
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging setup complete. Log file: {log_file}")

# Initialize logger
logger = logging.getLogger(__name__)



TEST_PROMPTS = [
    # Simple creative task
    {
        "prompt": "Give me three creative uses for a rubber band.",
        "description": "Simple creative task"
    },
    # Analytical task
    {
        "prompt": "Explain the concept of supply and demand in economics.",
        "description": "Basic analytical task"
    },
    # Step-by-step task
    {
        "prompt": "Describe how to make a peanut butter and jelly sandwich.",
        "description": "Simple procedural task"
    }
]

def test_prompts():
    """Run test prompts systematically with all combinations"""
    timestamp = int(time.time())
    output_dir = "forge_responses"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track failures
    failures = []
    
    # Test all combinations
    reasoning_speeds = [speed for speed in ReasoningSpeed]
    trace_options = [True, False]
    
    for prompt_info, speed, trace in product(TEST_PROMPTS, reasoning_speeds, trace_options):
        output_file = f"{output_dir}/forge_response_{timestamp}_{speed}_{trace}.json"
        
        try:
            logger.info(f"\n=== Testing Configuration ===")
            logger.info(f"Description: {prompt_info['description']}")
            logger.info(f"Speed: {speed}")
            logger.info(f"Trace: {trace}")
            
            input_data = ForgeInput(
                prompt=prompt_info["prompt"],
                reasoning_speed=speed,
                track=trace
            )
            
            # Add delay between requests to respect rate limits
            if len(failures) > 0 or os.path.exists(output_file):
                logger.info("Waiting 30 seconds before next request...")
                time.sleep(30)
            
            result = call_forge_api(input_data)
            
            # Save model as JSON
            with open(output_file, 'w') as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            
            logger.info(f"Response saved to {output_file}")
            
            # Log completion
            completion = result.result.choices[0].message.content
            logger.info("\nCompletion:")
            logger.info(completion)
            
            # Log trace information if available
            if result.result.traces:
                logger.info("\nTrace Summary:")
                metrics = result.result.traces.metrics
                logger.info(f"Total tokens: {metrics.total_tokens}")
                logger.info(f"Duration: {metrics.total_duration:.2f}s")
                logger.info(f"Models used: {', '.join(metrics.models_used)}")

            
            
        except Exception as e:
            failure_info = {
                "description": prompt_info["description"],
                "speed": speed,
                "trace": trace,
                "error": str(e)
            }
            failures.append(failure_info)
            logger.error(f"Error in test case: {json.dumps(failure_info, indent=2)}")
            
            # Save failure information
            failure_file = f"{output_dir}/failures_{timestamp}.json"
            with open(failure_file, 'w') as f:
                json.dump(failures, f, indent=2)
    
    return failures

if __name__ == "__main__":
    try:
        # Setup output directory and logging
        output_dir = "forge_responses"
        os.makedirs(output_dir, exist_ok=True)
        setup_logging(output_dir)
        
        logger.info("Starting Forge API systematic testing...")
        failures = test_prompts()
        
        if failures:
            logger.warning(f"Completed with {len(failures)} failures")
            for failure in failures:
                logger.error(f"Failure in test: {json.dumps(failure, indent=2)}")
        else:
            logger.info("All tests completed successfully!")
            
    except KeyboardInterrupt:
        logger.warning("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("\nTest run failed")
        sys.exit(1)