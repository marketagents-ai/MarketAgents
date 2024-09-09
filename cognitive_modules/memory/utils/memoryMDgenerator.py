import os
import sys
import json
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
import yaml
import argparse
from string import Template
from collections import deque
import tempfile
from datetime import timedelta
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from cognitive_modules.memory.retrieval_toys.bm25_cosine import SearchableCollection

class Conversation(BaseModel):
    role: str
    content: str
    entities: List[str] = Field(default_factory=list)

class ShortTermMemory(BaseModel):
    observation: str
    timestamp: datetime

class LongTermMemory(BaseModel):
    content: str
    relevance: float = Field(default=0.0)

class MemoryModule:
    def __init__(self, max_conversation_history: int = 5, max_short_term_memory: int = 10, yaml_template: dict = None):
        self.conversation_history = deque(maxlen=max_conversation_history)
        self.short_term_memory = deque(maxlen=max_short_term_memory)
        self.long_term_memory: List[LongTermMemory] = []
        self.external_references: List[Dict[str, str]] = []
        self.max_conversation_history = max_conversation_history
        self.max_short_term_memory = max_short_term_memory
        self.yaml_template = yaml_template or self.default_yaml_template()
        self.search_collection = SearchableCollection([], lambda x: x)

    @staticmethod
    def default_yaml_template():
        return {
            'title': '# Memory Module for Agent Context',
            'sections': [
                {'name': 'Conversation History', 'content': '${conversation_history}'},
                {'name': 'Short-term Memory (Environmental Observations)', 'content': '${short_term_memory}'},
                {'name': 'Long-term Memory', 'content': '${long_term_memory}'},
                {'name': 'External References', 'content': '${external_references}'},
                {'name': 'Context Metadata', 'content': '${context_metadata}'}
            ]
        }

    def load_conversation_history(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.conversation_history.clear()
        for item in tqdm(data[-self.max_conversation_history:], desc="Loading conversation history"):
            self.conversation_history.append(Conversation(**item))
        self._update_search_collection()

    def load_short_term_memory(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.short_term_memory.clear()
        for item in tqdm(data[-self.max_short_term_memory:], desc="Loading short-term memory"):
            self.short_term_memory.append(ShortTermMemory(**item))
        self._update_search_collection()

    def ingest_long_term_memory(self, folder_path: str):
        self.long_term_memory.clear()
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        for file in tqdm(files, desc="Ingesting long-term memory"):
            with open(os.path.join(folder_path, file), 'r') as f:
                content = f.read()
            self.long_term_memory.append(LongTermMemory(content=content))
        self._update_search_collection()

    def _update_search_collection(self):
        all_texts = (
            [f"{conv.role}: {conv.content}" for conv in self.conversation_history] +
            [item.observation for item in self.short_term_memory] +
            [item.content for item in self.long_term_memory]
        )
        self.search_collection = SearchableCollection(all_texts, lambda x: x)
        print("Building search index...")
        self.search_collection.build_index()
        print("Search index built.")

    def search(self, query: str, top_k: int = 5):
        results = self.search_collection.search(query, top_k=top_k)
        for ltm in self.long_term_memory:
            ltm.relevance = next((score for item, score in results if item == ltm.content), 0.0)
        return results

    def get_memory_artifact(self, query: str) -> str:
        conversation_history = "\n".join([f"{item.role}: {item.content}\nEntities: {', '.join(item.entities)}\n" for item in self.conversation_history])
        short_term_memory = "\n".join([f"- {item.timestamp.isoformat()}: {item.observation}" for item in self.short_term_memory])
        
        search_results = self.search(query)
        long_term_memory = "\n".join([f"- {item} (relevance: {score:.4f})" for item, score in search_results])
        
        external_references = "\n".join([f"- {ref['title']}: {ref['url']}" for ref in self.external_references]) if self.external_references else ""
        
        context_metadata = f"- Last Updated: {datetime.now().isoformat()}\n- Indexed Items: {len(self.search_collection.items)}\n- Conversation History Items: {len(self.conversation_history)}\n- Short-term Memory Items: {len(self.short_term_memory)}"
        
        content_map = {
            'conversation_history': conversation_history,
            'short_term_memory': short_term_memory,
            'long_term_memory': long_term_memory,
            'external_references': external_references,
            'context_metadata': context_metadata
        }
        
        output = [self.yaml_template['title']]
        for section in self.yaml_template['sections']:
            output.append(f"\n## {section['name']}")
            content = Template(section['content']).substitute(content_map)
            output.append(content)
        
        return "\n".join(output)

def load_yaml_template(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def run_memory_module(conversation_history: List[Dict], short_term_memory: List[Dict], long_term_memory: List[str], yaml_template: dict = None, max_conversation_history: int = 5, max_short_term_memory: int = 10):
    memory = MemoryModule(max_conversation_history=max_conversation_history, max_short_term_memory=max_short_term_memory, yaml_template=yaml_template)

    # Convert conversation history to Conversation objects
    memory.conversation_history = deque([Conversation(**conv) for conv in conversation_history], maxlen=max_conversation_history)

    # Convert short-term memory to ShortTermMemory objects
    memory.short_term_memory = deque([ShortTermMemory(**stm) for stm in short_term_memory], maxlen=max_short_term_memory)

    # Add long-term memory directly
    memory.long_term_memory = [LongTermMemory(content=ltm) for ltm in long_term_memory]

    memory._update_search_collection()

    # Generate and print the memory artifact
    print(memory.get_memory_artifact("sample query"))

if __name__ == "__main__":
    # Sample data (you can move these to separate variables if you prefer)
    conversation_history = [
        {"role": "user", "content": "What's the capital of France?", "entities": ["France"]},
        {"role": "assistant", "content": "The capital of France is Paris.", "entities": ["France", "Paris"]},
        {"role": "user", "content": "Tell me more about its famous landmarks.", "entities": ["Paris", "landmarks"]}
    ]

    short_term_memory = [
        {"observation": "User seems interested in French culture.", "timestamp": datetime.now().isoformat()},
        {"observation": "Weather in Paris is sunny today.", "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()}
    ]

    long_term_memory = [
        "Paris is known for the Eiffel Tower, a wrought-iron lattice tower on the Champ de Mars.",
        "The Louvre Museum in Paris is the world's largest art museum and a historic monument.",
        "French cuisine is famous for its refinement and is an important part of French culture."
    ]

    run_memory_module(conversation_history, short_term_memory, long_term_memory)

def run_memory_module_demo():
    # Sample conversation history
    conversation_history = [
        {
            "role": "user",
            "content": "What's the capital of France?",
            "entities": ["France"]
        },
        {
            "role": "assistant",
            "content": "The capital of France is Paris.",
            "entities": ["France", "Paris"]
        },
        {
            "role": "user",
            "content": "Tell me more about its famous landmarks.",
            "entities": ["Paris", "landmarks"]
        }
    ]

    # Sample short-term memory
    short_term_memory = [
        {
            "observation": "User seems interested in French culture.",
            "timestamp": datetime.now().isoformat()
        },
        {
            "observation": "Weather in Paris is sunny today.",
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()
        }
    ]

    # Sample long-term memory
    long_term_memory = [
        "Paris is known for the Eiffel Tower, a wrought-iron lattice tower on the Champ de Mars.",
        "The Louvre Museum in Paris is the world's largest art museum and a historic monument.",
        "French cuisine is famous for its refinement and is an important part of French culture."
    ]

    # Create temporary files for the sample data
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as conv_file:
        json.dump(conversation_history, conv_file)
        conversation_file = conv_file.name

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as stm_file:
        json.dump(short_term_memory, stm_file)
        short_term_file = stm_file.name

    # Create a temporary directory for long-term memory
    ltm_folder = tempfile.mkdtemp()
    
    try:
        # Write long-term memory to files
        for i, memory in enumerate(long_term_memory):
            with open(os.path.join(ltm_folder, f"memory_{i}.txt"), 'w') as f:
                f.write(memory)

        # Run the memory module with the sample data
        memory = MemoryModule()
        memory.load_conversation_history(conversation_file)
        memory.load_short_term_memory(short_term_file)
        memory.ingest_long_term_memory(ltm_folder)

        # Generate and print the memory artifact
        print(memory.get_memory_artifact("Paris landmarks"))

    finally:
        # Clean up temporary files and directory
        os.unlink(conversation_file)
        os.unlink(short_term_file)
        for file in os.listdir(ltm_folder):
            os.unlink(os.path.join(ltm_folder, file))
        os.rmdir(ltm_folder)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Generate memory artifact from external files")
        parser.add_argument("--conversation", required=True, help="Path to conversation history JSON file")
        parser.add_argument("--short_term", required=True, help="Path to short-term memory JSON file")
        parser.add_argument("--long_term", required=True, help="Path to long-term memory folder")
        parser.add_argument("--yaml", help="Path to YAML template file", default=None)
        parser.add_argument("--max_conv", type=int, help="Maximum conversation history items", default=5)
        parser.add_argument("--max_short", type=int, help="Maximum short-term memory items", default=10)
        args = parser.parse_args()

        run_memory_module(args.conversation, args.short_term, args.long_term, args.yaml, args.max_conv, args.max_short)
    else:
        run_memory_module_demo()