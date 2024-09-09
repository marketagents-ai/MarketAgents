import os
import sys
import tiktoken
from typing import List, Dict
from datetime import datetime, timedelta
import random
from nltk.corpus import wordnet as wn
from pydantic import BaseModel, Field
import nltk
import yaml
import argparse
from string import Template
from collections import deque

nltk.download('wordnet', quiet=True)

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

try:
    from cognitive_modules.memory.retrieval_toys.bm25_cosine import SearchableCollection
except ImportError:
    print("Warning: Unable to import SearchableCollection. Using a placeholder implementation.")
    class SearchableCollection:
        def __init__(self, items, preprocess_func):
            self.items = items
        def build_index(self):
            pass
        def search(self, query, top_k=5):
            return [(item, random.random()) for item in self.items[:top_k]]

class Conversation(BaseModel):
    question: str
    answer: str

class ShortTermMemory(BaseModel):
    observation: str
    timestamp: datetime

class LongTermMemory(BaseModel):
    content: str
    relevance: float = Field(default=0.0)

class MemoryModule:
    def __init__(self, max_conversation_history: int = 5, max_short_term_memory: int = 10, corpus: dict = None, yaml_template: dict = None):
        self.conversation_history = deque(maxlen=max_conversation_history)
        self.short_term_memory = deque(maxlen=max_short_term_memory)
        self.long_term_memory: List[LongTermMemory] = []
        self.external_references: List[Dict[str, str]] = []
        self.max_conversation_history = max_conversation_history
        self.max_short_term_memory = max_short_term_memory
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.search_collection = SearchableCollection([], lambda x: x)
        self.yaml_template = yaml_template or self.default_yaml_template()
        
        if corpus:
            self.ingest_corpus(corpus)

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

    def ingest_corpus(self, corpus: dict):
        self.conversation_history.extend([Conversation(**conv) for conv in corpus["conversations"][-self.max_conversation_history:]])
        self.short_term_memory.extend([ShortTermMemory(**mem) for mem in corpus["short_term_memory"][-self.max_short_term_memory:]])
        self.long_term_memory = [LongTermMemory(content=mem) for mem in corpus["long_term_memory"]]
        self._update_search_collection()

    def add_conversation(self, question: str, answer: str):
        self.conversation_history.append(Conversation(question=question, answer=answer))
        self._update_search_collection()

    def add_short_term_memory(self, observation: str, timestamp: datetime):
        self.short_term_memory.append(ShortTermMemory(observation=observation, timestamp=timestamp))
        self._update_search_collection()

    def add_long_term_memory(self, memory: str):
        self.long_term_memory.append(LongTermMemory(content=memory))
        self._update_search_collection()

    def add_external_reference(self, title: str, url: str):
        self.external_references.append({"title": title, "url": url})

    def _update_search_collection(self):
        all_texts = (
            [f"{conv.question} {conv.answer}" for conv in self.conversation_history] +
            [item.observation for item in self.short_term_memory] +
            [item.content for item in self.long_term_memory]
        )
        self.search_collection = SearchableCollection(all_texts, lambda x: x)
        self.search_collection.build_index()

    def search(self, query: str, top_k: int = 5):
        results = self.search_collection.search(query, top_k=top_k)
        for ltm in self.long_term_memory:
            ltm.relevance = next((score for item, score in results if item == ltm.content), 0.0)
        return results

    def get_memory_artifact(self, query: str) -> str:
        conversation_history = "\n".join([f"Q: {item.question}\nA: {item.answer}\n" for item in self.conversation_history])
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

def generate_random_sentence():
    noun = random.choice(list(wn.all_synsets(pos=wn.NOUN))).name().split('.')[0]
    verb = random.choice(list(wn.all_synsets(pos=wn.VERB))).name().split('.')[0]
    adj = random.choice(list(wn.all_synsets(pos=wn.ADJ))).name().split('.')[0]
    return f"The {adj} {noun} {verb}s."

def generate_mock_corpus(num_conversations: int = 10, num_short_term: int = 20, num_long_term: int = 50):
    corpus = {
        "conversations": [],
        "short_term_memory": [],
        "long_term_memory": []
    }
    
    # Generate conversation history
    for _ in range(num_conversations):
        question = generate_random_sentence()
        answer = generate_random_sentence()
        corpus["conversations"].append({"question": question, "answer": answer})
    
    # Generate short-term memory (environmental observations)
    current_time = datetime.now()
    for i in range(num_short_term):
        observation = generate_random_sentence()
        timestamp = current_time - timedelta(minutes=i*5)  # 5-minute intervals
        corpus["short_term_memory"].append({"observation": observation, "timestamp": timestamp})
    
    # Generate long-term memory
    for _ in range(num_long_term):
        memory = generate_random_sentence()
        corpus["long_term_memory"].append(memory)
    
    return corpus

def extract_query(corpus: dict) -> str:
    all_text = " ".join([
        " ".join([conv["question"], conv["answer"]]) for conv in corpus["conversations"]
    ] + [
        mem["observation"] for mem in corpus["short_term_memory"]
    ] + corpus["long_term_memory"])
    
    words = all_text.split()
    return random.choice([word for word in words if len(word) > 3])

def run_memory_module(corpus: dict = None, yaml_template_path: str = None, max_conversation_history: int = 5, max_short_term_memory: int = 10):
    if corpus is None:
        corpus = generate_mock_corpus()

    yaml_template = load_yaml_template(yaml_template_path) if yaml_template_path else None
    query = extract_query(corpus)
    memory = MemoryModule(max_conversation_history=max_conversation_history, max_short_term_memory=max_short_term_memory, corpus=corpus, yaml_template=yaml_template)

    # Add some additional content
    for _ in range(7):  # This will test the max_conversation_history limit
        memory.add_conversation(generate_random_sentence(), generate_random_sentence())
    
    for _ in range(12):  # This will test the max_short_term_memory limit
        memory.add_short_term_memory(generate_random_sentence(), datetime.now())

    memory.add_long_term_memory(generate_random_sentence())
    memory.add_external_reference("Random Reference", "https://example.com")

    # Generate and print the memory artifact
    print(memory.get_memory_artifact(query))

def load_yaml_template(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate memory artifact from YAML template")
    parser.add_argument("--yaml", help="Path to YAML template file", default=None)
    parser.add_argument("--max_conv", type=int, help="Maximum conversation history items", default=5)
    parser.add_argument("--max_short", type=int, help="Maximum short-term memory items", default=10)
    args = parser.parse_args()

    run_memory_module(yaml_template_path=args.yaml, max_conversation_history=args.max_conv, max_short_term_memory=args.max_short)