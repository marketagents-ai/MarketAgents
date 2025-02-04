from pathlib import Path
from market_agents.agents.base_agent.prompter import PromptManager
import logging

logger = logging.getLogger("test_agent")
logger.setLevel(logging.DEBUG)

def test_prompt_formatting():
    """Test that prompts are formatted correctly with all components."""
    
    # Test variables
    system_variables = {
        "role": "financial analyst",
        "persona": "You are an experienced financial analyst with expertise in market analysis.",
        "objectives": "Analyze market trends and provide actionable insights."
    }
    
    task_variables = {
        "task": "Review recent market data and generate analysis report",
        "output_schema": """
{
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "confidence": {"type": "number"}
    }
}
""",
        "output_format": "json_object"
    }
    
    # Initialize prompt manager with default template path
    prompt_manager = PromptManager()
    
    # Generate prompts
    system_prompt = prompt_manager.get_system_prompt(system_variables)
    task_prompt = prompt_manager.get_task_prompt(task_variables)
    
    # Print prompts for debugging
    print("\n=== System Prompt ===")
    print(system_prompt)
    print("\n=== Task Prompt ===")
    print(task_prompt)
    print("\n=== Task Variables ===")
    print(task_variables)
    
    # Basic assertions to verify prompt generation
    assert "financial analyst" in system_prompt, "Role not found in system prompt"
    assert "Review recent market data" in task_prompt, "Task not found in task prompt"
    assert isinstance(task_prompt, str), f"Task prompt is not a string, got {type(task_prompt)}"
    
    # Check if the schema content is in the prompt (not the variable name)
    assert '"type": "object"' in task_prompt, "JSON schema not found in task prompt"
    assert '"properties"' in task_prompt, "JSON schema not found in task prompt"
    assert '"analysis"' in task_prompt, "Expected output fields not found in task prompt"


if __name__ == "__main__":
    test_prompt_formatting()