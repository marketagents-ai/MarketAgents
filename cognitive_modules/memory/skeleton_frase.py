from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from bm25wcosine.bm25_cosine import BM25CosineEnsemble, SearchableCollection

class MemoryType(Enum):
    REFLECTION = "reflection"
    ACTION = "action"
    PERCEPTION = "perception"

class RetrievalMode(Enum):
    BM25_COSINE = "bm25_cosine"

class MemoryModule:
    def __init__(self, retrieval_mode: RetrievalMode = RetrievalMode.BM25_COSINE):
        self.memory: List[Dict[str, Any]] = []
        self.last_action: Optional[Dict[str, Any]] = None
        self.retrieval_mode = retrieval_mode
        self.dummy_dataset = []
        self.searchable_collection = SearchableCollection(self.dummy_dataset, lambda x: x['content'])

    def add_memory(self, entry: Dict[str, Any]) -> None:
        entry['timestamp'] = datetime.now()
        self.memory.append(entry)
        self.dummy_dataset.append(entry)
        self._update_indexes()

    def get_recent_memories(self, n: int) -> List[Dict[str, Any]]:
        return sorted(self.memory, key=lambda x: x['timestamp'], reverse=True)[:n]

    def update_last_action(self, action: Dict[str, Any]) -> None:
        self.last_action = action

    def get_last_action(self) -> Optional[Dict[str, Any]]:
        return self.last_action

    def get_memory_summary(self) -> str:
        recent_memories = self.get_recent_memories(5)
        summary = "Summary of recent memories:\n"
        for memory in recent_memories:
            summary += f"- {memory['type'].value}: {memory['content'][:50]}...\n"
        return summary

    def clear_memory(self) -> None:
        self.memory.clear()
        self.last_action = None
        self._clear_indexes()

    def _update_indexes(self) -> None:
        self.searchable_collection = SearchableCollection(self.dummy_dataset, lambda x: x['content'])

    def _clear_indexes(self) -> None:
        self.dummy_dataset.clear()
        self.searchable_collection = SearchableCollection(self.dummy_dataset, lambda x: x['content'])

    def retrieve_relevant_memories(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.searchable_collection.search(query, top_k)
        return [item for item, score in results]

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk_data(self, data: Any) -> List[Any]:
        pass

class TextChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

    def chunk_data(self, data: str) -> List[str]:
        words = data.split()
        return [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

class JsonChunkingStrategy(ChunkingStrategy):
    def __init__(self, max_keys: int = 5):
        self.max_keys = max_keys

    def chunk_data(self, data: Dict) -> List[Dict]:
        chunks = []
        keys = list(data.keys())
        for i in range(0, len(keys), self.max_keys):
            chunk = {k: data[k] for k in keys[i:i+self.max_keys]}
            chunks.append(chunk)
        return chunks

class Environment:
    def __init__(self, name: str):
        self.name = name
        self.state = {}
        self.last_observation = None
        self.last_reward = 0

    def get_state(self):
        return self.state

    def get_action_space(self):
        return ["action1", "action2", "action3"]

    def step(self, action):
        # Simulate environment step
        self.last_observation = f"Observation after {action}"
        self.last_reward = 1  # Simple reward
        return self.last_observation, self.last_reward

    def get_last_observation(self):
        return self.last_observation

    def get_last_reward(self):
        return self.last_reward

class Agent:
    def perceive(self, environment):
        return f"Perceived state: {environment.get_state()}"

    def decide_action(self, perception, recent_memories):
        return {"action": "action1", "reason": "Based on perception and memories"}

    def reflect(self, observation, reward):
        return {
            "reflection": f"Reflected on observation: {observation}",
            "strategy_update": "Updated strategy based on reward"
        }

def generate_actions(agent: Agent, environment: Environment, memory_module: MemoryModule):
    perception = agent.perceive(environment)
    memory_module.add_memory({
        'type': MemoryType.PERCEPTION,
        'content': perception,
        'environment': environment.name
    })

    recent_memories = memory_module.get_recent_memories(5)
    action = agent.decide_action(perception, recent_memories)
    memory_module.update_last_action(action)

    memory_module.add_memory({
        'type': MemoryType.ACTION,
        'content': str(action),
        'environment': environment.name
    })

    observation, reward = environment.step(action['action'])
    reflection = agent.reflect(observation, reward)
    memory_module.add_memory({
        'type': MemoryType.REFLECTION,
        'content': reflection['reflection'],
        'environment': environment.name,
        'reward': reward,
        'strategy_update': reflection['strategy_update']
    })

    return action

def generate_prompts(memory_module: MemoryModule, environment: Environment):
    recent_memories = memory_module.get_recent_memories(5)
    last_action = memory_module.get_last_action()

    perception_prompt = f"""
    Perceive the current state of the {environment.name} environment:

    Environment State: {environment.get_state()}
    Recent Memories: {recent_memories}

    Generate a brief monologue about your current perception of this environment.
    """

    action_prompt = f"""
    Generate an action for the {environment.name} environment based on the following:

    Perception: {perception_prompt}
    Environment State: {environment.get_state()}
    Recent Memories: {recent_memories}
    Available Actions: {environment.get_action_space()}

    Choose an appropriate action for this environment.
    """

    reflection_prompt = f"""
    Reflect on this observation from the {environment.name} environment:

    Observation: {environment.get_last_observation()}
    Environment State: {environment.get_state()}
    Last Action: {last_action}
    Reward: {environment.get_last_reward()}

    Actions:
    1. Reflect on the observation and surplus based on your last action
    2. Update strategy based on this reflection, the surplus, and your previous strategy

    Previous strategy: {memory_module.get_memory_summary()}
    """

    return perception_prompt, action_prompt, reflection_prompt

if __name__ == "__main__":
    # Example usage
    env = Environment("TestEnvironment")
    agent = Agent()
    memory = MemoryModule()

    for _ in range(5):
        action = generate_actions(agent, env, memory)
        print(f"Performed action: {action}")

    query = "action1"
    relevant_memories = memory.retrieve_relevant_memories(query)
    print(f"Relevant memories for query '{query}':")
    for mem in relevant_memories:
        print(f"- {mem['type'].value}: {mem['content']}")

    perception_prompt, action_prompt, reflection_prompt = generate_prompts(memory, env)
    print("\nPerception Prompt:")
    print(perception_prompt)
    print("\nAction Prompt:")
    print(action_prompt)
    print("\nReflection Prompt:")
    print(reflection_prompt)