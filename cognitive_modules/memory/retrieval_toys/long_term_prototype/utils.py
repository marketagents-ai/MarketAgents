import json
from typing import Dict, Any

def format_memory_for_prompt(memory: Dict[str, Any]) -> str:
    """Format a single memory for inclusion in an LLM prompt."""
    content = memory['content']
    metadata = memory.get('metadata', {})
    relevance = memory.get('relevance_score', 0)
    
    formatted = f"Memory (relevance: {relevance:.2f}):\n"
    formatted += f"Content: {content}\n"
    
    if metadata:
        formatted += "Metadata:\n"
        for key, value in metadata.items():
            formatted += f"- {key}: {value}\n"
    
    return formatted

def format_memories_for_prompt(memories: List[Dict[str, Any]], max_memories: int = 5) -> str:
    """Format multiple memories for inclusion in an LLM prompt."""
    formatted_memories = [format_memory_for_prompt(mem) for mem in memories[:max_memories]]
    return "\n".join(formatted_memories)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: str):
    """Save a dictionary as a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)