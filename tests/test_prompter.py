from pathlib import Path
from datetime import datetime
from market_agents.agents.base_agent.prompter import (
    PromptManager, 
    SystemPromptVariables,
    TaskPromptVariables
)
import logging

logger = logging.getLogger("test_agent")
logger.setLevel(logging.DEBUG)

# In test_prompter.py
def test_prompt_formatting():
    """Test that prompts are formatted correctly with nested YAML structure."""
    
    # Test variables with dict-style skills
    system_variables_dict = SystemPromptVariables(
        role="financial analyst",
        persona="You are an experienced financial analyst with expertise in market analysis.",
        objectives=["Analyze market trends", "Provide actionable insights"],
        skills=[
            {
                "name": "Market Analysis",
                "description": "Expert in technical and fundamental analysis"
            },
            {
                "name": "Data Visualization",
                "description": "Creating clear visual representations of market data"
            }
        ],
        datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Test variables with string-style skills
    system_variables_str = SystemPromptVariables(
        role="market researcher",
        persona="You are a skilled market researcher specializing in trend analysis.",
        objectives=["Research market patterns", "Generate insights"],
        skills=["Technical Analysis", "Data Mining", "Pattern Recognition"],
        datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    task_variables = TaskPromptVariables(
        task="Review recent market data and generate analysis report",
        output_format="json_object",
        output_schema="""
{
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "confidence": {"type": "number"}
    }
}
"""
    )
    
    # Initialize prompt manager
    prompt_manager = PromptManager(
        template_paths=[Path("market_agents/agents/configs/prompts/default_prompt.yaml")]
    )
    
    # Test dict-style skills
    system_prompt_dict = prompt_manager.get_system_prompt(system_variables_dict.model_dump())
    print("\n=== System Prompt (Dict Skills) ===")
    print(system_prompt_dict)
    
    # Test string-style skills
    system_prompt_str = prompt_manager.get_system_prompt(system_variables_str.model_dump())
    print("\n=== System Prompt (String Skills) ===")
    print(system_prompt_str)
    
    # Test task prompt
    task_prompt = prompt_manager.get_task_prompt(task_variables.model_dump())
    print("\n=== Task Prompt ===")
    print(task_prompt)
    
    # Test dict-style skills prompt
    assert "financial analyst" in system_prompt_dict
    assert "Market Analysis" in system_prompt_dict
    assert "Expert in technical and fundamental analysis" in system_prompt_dict
    
    # Test string-style skills prompt
    assert "market researcher" in system_prompt_str
    assert "Technical Analysis" in system_prompt_str
    assert "Pattern Recognition" in system_prompt_str
    
    # Test common elements
    for prompt in [system_prompt_dict, system_prompt_str]:
        assert isinstance(prompt, str)
        assert datetime.now().strftime("%Y-%m-%d") in prompt
        assert any(obj in prompt for obj in ["Analyze market trends", "Research market patterns"])

def test_empty_optional_fields():
    """Test that prompts handle missing optional fields gracefully."""
    
    # Test with minimal required fields
    system_variables = SystemPromptVariables(
        role="basic agent",
        datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    task_variables = TaskPromptVariables(
        task="simple task"
    )
    
    prompt_manager = PromptManager(
        template_paths=[Path("market_agents/agents/configs/prompts/default_prompt.yaml")]
    )
    
    system_prompt = prompt_manager.get_system_prompt(system_variables.model_dump())
    task_prompt = prompt_manager.get_task_prompt(task_variables.model_dump())
    
    assert "basic agent" in system_prompt
    assert "simple task" in task_prompt
    assert "N/A" in system_prompt
    assert isinstance(system_prompt, str)
    assert isinstance(task_prompt, str)

if __name__ == "__main__":
    test_prompt_formatting()
    test_empty_optional_fields()