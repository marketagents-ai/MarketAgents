from enum import Enum
from typing import Optional, Dict, Any, List, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class Performative(Enum):
    """Enumeration of ACL message performatives."""

    ACCEPT_PROPOSAL = "accept-proposal"
    CALL_FOR_PROPOSAL = "cfp"
    PROPOSE = "propose"
    REJECT_PROPOSAL = "reject-proposal"
    INFORM = "inform"
    REQUEST = "request"
    QUERY_IF = "query-if"
    NOT_UNDERSTOOD = "not-understood"


class AgentID(BaseModel):
    """
    Represents the identity of an agent.

    Attributes:
        name (str): The name of the agent.
        address (Optional[str]): The address of the agent, if applicable.
    """

    name: str
    address: Optional[str] = None


class ACLMessage(BaseModel, Generic[T]):
    """
    Represents an ACL message with various attributes and creation methods.

    Attributes:
        performative (Performative): The type of performative for the message.
        sender (AgentID): The sender of the message.
        receivers (List[AgentID]): The list of receivers for the message.
        content (T): The content of the message.
        reply_with (Optional[str]): An identifier for the message to be used in replies.
        in_reply_to (Optional[str]): The identifier of the message this is replying to.
        conversation_id (Optional[str]): An identifier for the conversation this message is part of.
        protocol (str): The protocol being used (default is "double-auction").
        language (str): The language of the message content (default is "JSON").
        ontology (str): The ontology used for the message content (default is "market-ontology").
        reply_by (Optional[datetime]): The deadline for replying to this message.
    """

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
        """
        Create a bid message.

        Args:
            sender (AgentID): The agent sending the bid.
            receiver (AgentID): The agent receiving the bid.
            bid_price (float): The price of the bid.
            bid_quantity (int): The quantity of the bid.

        Returns:
            ACLMessage: A new ACLMessage instance representing a bid.
        """
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
        """
        Create an ask message.

        Args:
            sender (AgentID): The agent sending the ask.
            receiver (AgentID): The agent receiving the ask.
            ask_price (float): The price of the ask.
            ask_quantity (int): The quantity of the ask.

        Returns:
            ACLMessage: A new ACLMessage instance representing an ask.
        """
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
        """
        Create an accept message in response to a proposal.

        Args:
            sender (AgentID): The agent sending the accept message.
            receiver (AgentID): The agent receiving the accept message.
            original_message_id (str): The ID of the original proposal being accepted.

        Returns:
            ACLMessage: A new ACLMessage instance representing an acceptance.
        """
        return cls(
            performative=Performative.ACCEPT_PROPOSAL,
            sender=sender,
            receivers=[receiver],
            content={"accepted": True},
            in_reply_to=original_message_id
        )

    @classmethod
    def create_reject(cls, sender: AgentID, receiver: AgentID, original_message_id: str, reason: str) -> 'ACLMessage':
        """
        Create a reject message in response to a proposal.

        Args:
            sender (AgentID): The agent sending the reject message.
            receiver (AgentID): The agent receiving the reject message.
            original_message_id (str): The ID of the original proposal being rejected.
            reason (str): The reason for rejection.

        Returns:
            ACLMessage: A new ACLMessage instance representing a rejection.
        """
        return cls(
            performative=Performative.REJECT_PROPOSAL,
            sender=sender,
            receivers=[receiver],
            content={"accepted": False, "reason": reason},
            in_reply_to=original_message_id
        )

    @classmethod
    def create_inform(cls, sender: AgentID, receiver: AgentID, info_type: str, info_content: Any) -> 'ACLMessage':
        """
        Create an inform message.

        Args:
            sender (AgentID): The agent sending the inform message.
            receiver (AgentID): The agent receiving the inform message.
            info_type (str): The type of information being sent.
            info_content (Any): The content of the information.

        Returns:
            ACLMessage: A new ACLMessage instance representing an inform message.
        """
        return cls(
            performative=Performative.INFORM,
            sender=sender,
            receivers=[receiver],
            content={"type": info_type, "content": info_content}
        )

    @classmethod
    def create_message(cls, performative: Performative, sender: str, receiver: str, content: Any, **kwargs) -> 'ACLMessage':
        """
        Create a generic ACL message.

        Args:
            performative (Performative): The performative of the message.
            sender (str): The name of the sender agent.
            receiver (str): The name of the receiver agent.
            content (Any): The content of the message.
            **kwargs: Additional keyword arguments for the message.

        Returns:
            ACLMessage: A new ACLMessage instance.
        """
        return cls(
            performative=performative,
            sender=AgentID(name=sender),
            receivers=[AgentID(name=receiver)],
            content=content,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ACLMessage to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the ACLMessage.
        """
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACLMessage':
        """
        Create an ACLMessage instance from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing ACLMessage data.

        Returns:
            ACLMessage: A new ACLMessage instance created from the dictionary data.
        """
        return cls(**data)