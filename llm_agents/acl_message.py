from enum import Enum
from typing import Optional, Dict, Any, List, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')

class Performative(Enum):
    ACCEPT_PROPOSAL = "accept-proposal"
    CALL_FOR_PROPOSAL = "cfp"
    PROPOSE = "propose"
    REJECT_PROPOSAL = "reject-proposal"
    INFORM = "inform"
    REQUEST = "request"
    QUERY_IF = "query-if"
    NOT_UNDERSTOOD = "not-understood"

class AgentID(BaseModel):
    name: str
    address: Optional[str] = None

class ACLMessage(BaseModel, Generic[T]):
    performative: Performative
    sender: AgentID
    receivers: List[AgentID]
    content: T
    reply_with: Optional[str] = None
    in_reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    protocol: str = "double-auction"
    language: str = "JSON"
    ontology: str = "market-ontology"
    reply_by: Optional[datetime] = None

    class Config:
        use_enum_values = True

    @classmethod
    def create_bid(cls, sender: AgentID, receiver: AgentID, bid_price: float, bid_quantity: int) -> 'ACLMessage':
        content = {
            "type": "bid",
            "price": bid_price,
            "quantity": bid_quantity
        }
        return cls(
            performative=Performative.PROPOSE,
            sender=sender,
            receivers=[receiver],
            content=content,
            conversation_id=f"bid-{datetime.now().isoformat()}"
        )

    @classmethod
    def create_ask(cls, sender: AgentID, receiver: AgentID, ask_price: float, ask_quantity: int) -> 'ACLMessage':
        content = {
            "type": "ask",
            "price": ask_price,
            "quantity": ask_quantity
        }
        return cls(
            performative=Performative.PROPOSE,
            sender=sender,
            receivers=[receiver],
            content=content,
            conversation_id=f"ask-{datetime.now().isoformat()}"
        )

    @classmethod
    def create_accept(cls, sender: AgentID, receiver: AgentID, original_message_id: str) -> 'ACLMessage':
        return cls(
            performative=Performative.ACCEPT_PROPOSAL,
            sender=sender,
            receivers=[receiver],
            content={"accepted": True},
            in_reply_to=original_message_id
        )

    @classmethod
    def create_reject(cls, sender: AgentID, receiver: AgentID, original_message_id: str, reason: str) -> 'ACLMessage':
        return cls(
            performative=Performative.REJECT_PROPOSAL,
            sender=sender,
            receivers=[receiver],
            content={"accepted": False, "reason": reason},
            in_reply_to=original_message_id
        )

    @classmethod
    def create_inform(cls, sender: AgentID, receiver: AgentID, info_type: str, info_content: Any) -> 'ACLMessage':
        return cls(
            performative=Performative.INFORM,
            sender=sender,
            receivers=[receiver],
            content={"type": info_type, "content": info_content}
        )
    @classmethod
    def create_message(cls, performative: Performative, sender: str, receiver: str, content: Any, **kwargs) -> 'ACLMessage':
        return cls(
            performative=performative,
            sender=AgentID(name=sender),
            receivers=[AgentID(name=receiver)],
            content=content,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACLMessage':
        return cls(**data)