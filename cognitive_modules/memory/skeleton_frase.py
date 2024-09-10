from enum import Enum
from typing import Any, Dict, List
from abc import ABC, abstractmethod

class MemoryType(Enum):
    INNER_MONOLOGUE = "inner_monologue"
    FINANCE_HISTORY = "finance_history"
    SOCIAL_HISTORY = "social_history"
    ACTIVITY_LOG = "activity_log"
    MARKET_HISTORY = "market_history"
    OBSERVATION_HISTORY = "observation_history"

class BaseMemory(ABC):
    @abstractmethod
    def add(self, data: Any):
        pass

    @abstractmethod
    def get(self, query: Dict) -> List[Any]:
        pass

    @abstractmethod
    def update(self, memory_id: str, data: Any):
        pass

    @abstractmethod
    def delete(self, memory_id: str):
        pass

class SimpleMemory(BaseMemory):
    def __init__(self):
        self.data: List[Any] = []

    def add(self, data: Any):
        self.data.append(data)

    def get(self, query: Dict) -> List[Any]:
        return [item for item in self.data if all(item.get(k) == v for k, v in query.items())]

    def update(self, memory_id: str, data: Any):
        for item in self.data:
            if item.get('id') == memory_id:
                item.update(data)
                break

    def delete(self, memory_id: str):
        self.data = [item for item in self.data if item.get('id') != memory_id]

class MemoryFactory:
    @staticmethod
    def create_memory(memory_type: MemoryType) -> BaseMemory:
        return SimpleMemory()

class MemoryIndex(ABC):
    @abstractmethod
    def add_to_index(self, memory: Any):
        pass

    @abstractmethod
    def search(self, query: str) -> List[Any]:
        pass

class SimpleMemoryIndex(MemoryIndex):
    def __init__(self):
        self.index: Dict[str, List[Any]] = {}

    def add_to_index(self, memory: Any):
        keywords = str(memory).lower().split()
        for keyword in keywords:
            if keyword not in self.index:
                self.index[keyword] = []
            self.index[keyword].append(memory)

    def search(self, query: str) -> List[Any]:
        keywords = query.lower().split()
        results = []
        for keyword in keywords:
            results.extend(self.index.get(keyword, []))
        return list(set(results))

class MemoryModule:
    def __init__(self):
        self.memories: Dict[MemoryType, BaseMemory] = {
            memory_type: MemoryFactory.create_memory(memory_type)
            for memory_type in MemoryType
        }
        self.index = SimpleMemoryIndex()

    def add_memory(self, memory_type: MemoryType, data: Any):
        self.memories[memory_type].add(data)
        self.index.add_to_index(data)

    def get_memory(self, memory_type: MemoryType, query: Dict) -> List[Any]:
        return self.memories[memory_type].get(query)

    def update_memory(self, memory_type: MemoryType, memory_id: str, data: Any):
        self.memories[memory_type].update(memory_id, data)
        self.index.add_to_index(data)

    def delete_memory(self, memory_type: MemoryType, memory_id: str):
        self.memories[memory_type].delete(memory_id)

    def search_memories(self, query: str) -> List[Any]:
        return self.index.search(query)

if __name__ == "__main__":
    memory_module = MemoryModule()
    memory_module.add_memory(MemoryType.FINANCE_HISTORY, {"id": "trade_1", "type": "buy", "asset": "AAPL", "amount": 100, "price": 150.00})
    trades = memory_module.get_memory(MemoryType.FINANCE_HISTORY, {"asset": "AAPL"})
    print("Retrieved trades:", trades)
    memory_module.update_memory(MemoryType.FINANCE_HISTORY, "trade_1", {"price": 151.00})
    search_results = memory_module.search_memories("AAPL buy")
    print("Search results:", search_results)
    memory_module.delete_memory(MemoryType.FINANCE_HISTORY, "trade_1")
