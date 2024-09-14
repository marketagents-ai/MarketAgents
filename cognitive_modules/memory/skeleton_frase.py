from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from retrieval_toys.bm25wcosine.bm25_cosine import BM25CosineEnsemble, SearchableCollection
from llm_agents.market_agent.market_schemas import PerceptionSchema, ReflectionSchema

class MemoryType(Enum):
    PERCEPTION = "perception"
    REFLECTION = "reflection"

class RetrievalMode(Enum):
    BM25_COSINE = "bm25_cosine"

class MemoryModule:
    def __init__(self, retrieval_mode: RetrievalMode = RetrievalMode.BM25_COSINE):
        self.memory: List[Dict[str, Any]] = []
        self.retrieval_mode = retrieval_mode
        self.dummy_dataset = []
        self.searchable_collection = SearchableCollection(self.dummy_dataset, lambda x: x['content'])

    def add_memory(self, entry: Dict[str, Any]) -> None:
        entry['timestamp'] = datetime.now()
        if entry['type'] == MemoryType.PERCEPTION:
            entry['content'] = PerceptionSchema(**entry['content']).dict()
        elif entry['type'] == MemoryType.REFLECTION:
            entry['content'] = ReflectionSchema(**entry['content']).dict()
        self.memory.append(entry)
        self.dummy_dataset.append(entry)
        self._update_indexes()

    def get_recent_memories(self, n: int) -> List[Dict[str, Any]]:
        return sorted(self.memory, key=lambda x: x['timestamp'], reverse=True)[:n]

    def get_memory_summary(self) -> str:
        recent_memories = self.get_recent_memories(5)
        summary = "Summary of recent memories:\n"
        for memory in recent_memories:
            content = memory['content']
            if memory['type'] == MemoryType.PERCEPTION:
                summary += f"- Perception: {content['monologue'][:50]}...\n"
                summary += f"  Strategy: {content['strategy'][:50]}...\n"
            elif memory['type'] == MemoryType.REFLECTION:
                summary += f"- Reflection: {content['reflection'][:50]}...\n"
                summary += f"  Strategy Update: {content['strategy_update'][:50]}...\n"
        return summary

    def clear_memory(self) -> None:
        self.memory.clear()
        self._clear_indexes()

    def _update_indexes(self) -> None:
        self.searchable_collection = SearchableCollection(self.dummy_dataset, lambda x: self._get_indexable_content(x))

    def _clear_indexes(self) -> None:
        self.dummy_dataset.clear()
        self.searchable_collection = SearchableCollection(self.dummy_dataset, lambda x: self._get_indexable_content(x))

    def _get_indexable_content(self, x):
        if x['type'] == MemoryType.PERCEPTION:
            return f"{x['content']['monologue']} {x['content']['strategy']}"
        elif x['type'] == MemoryType.REFLECTION:
            return f"{x['content']['reflection']} {x['content']['strategy_update']}"

    def retrieve_relevant_memories(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        results = self.searchable_collection.search(query, top_k)
        return [item for item, score in results]

class Environment:
    def __init__(self, name: str):
        self.name = name
        self.state = {}
        self.last_observation = None
        self.last_reward = 0

    def get_state(self):
        return self.state

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
        return PerceptionSchema(
            monologue=f"Perceived state: {environment.get_state()}",
            strategy="Current market strategy"
        )

    def reflect(self, observation, reward):
        return ReflectionSchema(
            reflection=f"Reflected on observation: {observation}",
            strategy_update="Updated strategy based on reward"
        )

def generate_perception(agent: Agent, environment: Environment, memory_module: MemoryModule):
    perception = agent.perceive(environment)
    memory_module.add_memory({
        'type': MemoryType.PERCEPTION,
        'content': perception,
        'environment': environment.name
    })
    return perception

def generate_reflection(agent: Agent, environment: Environment, memory_module: MemoryModule):
    observation = environment.get_last_observation()
    reward = environment.get_last_reward()
    reflection = agent.reflect(observation, reward)
    memory_module.add_memory({
        'type': MemoryType.REFLECTION,
        'content': reflection,
        'environment': environment.name
    })
    return reflection

def generate_prompts(memory_module: MemoryModule, environment: Environment):
    recent_memories = memory_module.get_recent_memories(5)

    perception_prompt = f"""
    Perceive the current state of the {environment.name} environment:

    Environment State: {environment.get_state()}
    Recent Memories: {recent_memories}

    Generate a brief monologue about your current perception of this environment and your strategy.
    """

    reflection_prompt = f"""
    Reflect on this observation from the {environment.name} environment:

    Observation: {environment.get_last_observation()}
    Environment State: {environment.get_state()}
    Reward: {environment.get_last_reward()}

    Actions:
    1. Reflect on the observation and surplus based on your last action
    2. Update strategy based on this reflection, the surplus, and your previous strategy

    Previous memories: {memory_module.get_memory_summary()}
    """

    return perception_prompt, reflection_prompt

if __name__ == "__main__":
    # Example usage
    env = Environment("TestEnvironment")
    agent = Agent()
    memory = MemoryModule()

    for _ in range(5):
        perception = generate_perception(agent, env, memory)
        print(f"Generated perception: {perception}")
        
        # Simulate an action and environment step
        env.step("some_action")
        
        reflection = generate_reflection(agent, env, memory)
        print(f"Generated reflection: {reflection}")

    query = "market strategy"
    relevant_memories = memory.retrieve_relevant_memories(query)
    print(f"\nRelevant memories for query '{query}':")
    for mem in relevant_memories:
        if mem['type'] == MemoryType.PERCEPTION:
            print(f"- Perception: {mem['content']['monologue']}")
            print(f"  Strategy: {mem['content']['strategy']}")
        elif mem['type'] == MemoryType.REFLECTION:
            print(f"- Reflection: {mem['content']['reflection']}")
            print(f"  Strategy Update: {mem['content']['strategy_update']}")

    perception_prompt, reflection_prompt = generate_prompts(memory, env)
    print("\nPerception Prompt:")
    print(perception_prompt)
    print("\nReflection Prompt:")
    print(reflection_prompt)