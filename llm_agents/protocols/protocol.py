from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel

class Protocol(BaseModel, ABC):
    @abstractmethod
    def parse_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_message(self, *args, **kwargs) -> 'Protocol':
        pass

    @classmethod
    @abstractmethod
    def create_message(cls, *args, **kwargs) -> 'Protocol':
        pass