from enum import Enum
from typing import Optional, Dict, Any, List, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime
from protocols.protocol import Protocol

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


class ACLMessage(Protocol, Generic[T]):
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

    performative: Optional[Performative] = None
    sender: Optional[AgentID] = None
    receivers: Optional[List[AgentID]] = None
    content: Optional[T] = None
    reply_with: Optional[str] = None
    in_reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    protocol: str = "double-auction"
    language: str = "JSON"
    ontology: str = "market-ontology"
    reply_by: Optional[datetime] = None

    class Config:
        use_enum_values = True

    def parse_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse an action into a format the environment can use.

        Args:
            action (Dict[str, Any]): The action to parse.

        Returns:
            Dict[str, Any]: The parsed action.
        """
        # Implement parsing logic here
        return action

    def generate_message(self, *args, **kwargs) -> 'ACLMessage':
        """
        Generate a new ACLMessage based on the provided arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ACLMessage: A new ACLMessage instance.
        """
        return self.create_message(*args, **kwargs)

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
    
    def parse_to_market_action(self) -> Dict[str, Any]:
        """
        Parse the ACL message content into a market action.
        
        Returns:
            Dict[str, Any]: A dictionary containing the parsed market action.
        """
        if self.performative != Performative.PROPOSE:
            raise ValueError(f"Unexpected performative: {self.performative}")
        
        content = self.content
        if not isinstance(content, dict):
            raise ValueError(f"Unexpected content type: {type(content)}")
        
        action_type = content.get("type")
        if action_type not in ["bid", "ask"]:
            raise ValueError(f"Invalid action type: {action_type}")
        
        return {
            "type": action_type,
            "price": content.get("price"),
            "quantity": content.get("quantity")
        }