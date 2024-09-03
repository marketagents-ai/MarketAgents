from datetime import datetime
import json
import logging
import traceback
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from acl_message.acl_message import ACLMessage, AgentID, Performative
from zi_agent.ziagents import MarketAction, ZIAgent, Order, Trade, create_zi_agent, MarketInfo
from base_agent.agent import Agent as LLMAgent
from market_agent.market_schemas import DoubleAuctionMessage, MarketActionSchema
from base_agent.aiutilities import LLMConfig

logger = logging.getLogger(__name__)

MARKET_ADDRESS = "market_orchestrator_node"


class MarketAgent(BaseModel):
    """
    Represents a market agent that can use either a ZI (Zero Intelligence) agent or an LLM-based agent.

    Attributes:
        zi_agent (ZIAgent): The Zero Intelligence agent.
        llm_agent (LLMAgent): The LLM-based agent.
        memory (List[Dict[str, Any]]): List to store agent's memory of interactions.
        use_llm (bool): Flag to determine whether to use LLM for decision making.
        address (str): Agent's address in the market.
    """

    zi_agent: ZIAgent
    llm_agent: LLMAgent
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    use_llm: bool = Field(default=False, description="Whether to use LLM for decision making")
    address: str = Field(default="", description="Agent's address")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, agent_id: int, is_buyer: bool, num_units: int, base_value: float, use_llm: bool,
               initial_cash: float, initial_goods: int, noise_factor: float = 0.1,
               max_relative_spread: float = 0.2, llm_config: Optional[LLMConfig] = None) -> 'MarketAgent':
        """
        Create a new MarketAgent instance.

        Args:
            agent_id (int): Unique identifier for the agent.
            is_buyer (bool): True if the agent is a buyer, False if seller.
            num_units (int): Number of units the agent can trade.
            base_value (float): Base value for the agent's goods.
            use_llm (bool): Whether to use LLM for decision making.
            initial_cash (float): Initial cash holdings of the agent.
            initial_goods (int): Initial goods holdings of the agent.
            noise_factor (float, optional): Noise factor for ZI agent. Defaults to 0.1.
            max_relative_spread (float, optional): Maximum relative spread for ZI agent. Defaults to 0.2.
            llm_config (Optional[LLMConfig], optional): Configuration for LLM agent. Defaults to None.

        Returns:
            MarketAgent: A new instance of MarketAgent.
        """
        zi_agent = create_zi_agent(agent_id, is_buyer, num_units, base_value, initial_cash, initial_goods,
                                   noise_factor, max_relative_spread)
        
        role = "buyer" if is_buyer else "seller"
        llm_agent = LLMAgent(role=role, llm_config=llm_config.dict() if llm_config else {}, 
                             output_format=MarketActionSchema.model_json_schema())

        return cls(zi_agent=zi_agent, llm_agent=llm_agent, use_llm=use_llm, address=f"agent_{agent_id}")

    def generate_bid(self, market_info: MarketInfo, round_num: int) -> Optional[ACLMessage]:
        """
        Generate a bid based on the current market information.

        Args:
            market_info (MarketInfo): Current market information.
            round_num (int): Current round number.

        Returns:
            Optional[ACLMessage]: The generated bid as an ACL message, or None if no bid is made.
        """
        if self.use_llm:
            return self._generate_llm_bid(market_info, round_num)
        else:
            return self._generate_zi_bid(market_info, round_num)
        
    def _generate_llm_bid(self, market_info: MarketInfo, round_num: int) -> Optional[ACLMessage]:
        """
        Generate a bid using the LLM-based strategy.

        Args:
            market_info (MarketInfo): Current market information.
            round_num (int): Current round number.

        Returns:
            Optional[ACLMessage]: The generated bid as an ACL message, or None if no bid is made.
        """
        try:
            market_info_str = self._get_market_info(market_info)
            recent_memories = self.get_recent_memories(2)
            memory_str = self._format_memories(recent_memories)

            task_prompt = f"Generate a market action based on the following market information: {market_info_str}"
            
            if memory_str:
                task_prompt += f"\n\nRecent market activities:\n{memory_str}"
            
            llm_response = self.llm_agent.execute(task_prompt)

            self.log_interaction(round_num, task_prompt, llm_response)

            logger.info("---LLM JSON RESPONSE---")
            logger.info(json.dumps(llm_response))

            market_action = MarketActionSchema(**llm_response)
            if market_action.action == "hold":
                logger.info("LLM decided to hold, returning None")
                return None
            
            bid = market_action.bid

            if bid is None:
                logger.info("No bid in market action, returning None")
                return None

            return self._create_acl_message(bid.acl_message.action, bid.acl_message.price, bid.acl_message.quantity)
        
        except Exception as e:
            logger.error(f"Error in _generate_llm_bid: {e}")
            logger.error(traceback.format_exc())
            return None

    def _generate_zi_bid(self, market_info: MarketInfo, round_num: int) -> Optional[ACLMessage]:
        """
        Generate a bid using the Zero Intelligence strategy.

        Args:
            market_info (MarketInfo): Current market information.
            round_num (int): Current round number.

        Returns:
            Optional[ACLMessage]: The generated bid as an ACL message, or None if no bid is made.
        """
        zi_bid = self.zi_agent.generate_bid()
        if zi_bid is None:
            return None
        return self._create_acl_message("bid" if zi_bid.is_buy else "ask", zi_bid.price, zi_bid.quantity)

    def _create_acl_message(self, action: str, price: float, quantity: int) -> ACLMessage:
        """
        Create an ACL message for a bid or ask.

        Args:
            action (str): The action, either "bid" or "ask".
            price (float): The price of the bid or ask.
            quantity (int): The quantity of the bid or ask.

        Returns:
            ACLMessage: The created ACL message.
        """
        sender_id = AgentID(name=str(self.zi_agent.id), address=self.address)
        receiver_id = AgentID(name="market", address=MARKET_ADDRESS)

        content = DoubleAuctionMessage(
            action=action,
            price=price,
            quantity=quantity
        )

        acl_message = ACLMessage(
            performative=Performative.PROPOSE,
            sender=sender_id,
            receivers=[receiver_id],
            content=content,
            protocol="double-auction",
            ontology="market-ontology",
            conversation_id=f"{action}-{datetime.now().isoformat()}"
        )

        logger.info(f"Generated ACL message: {acl_message}")
        return acl_message
    def receive_message(self, message: ACLMessage) -> None:
        """
        Process a received ACL message.

        Args:
            message (ACLMessage): The received ACL message.
        """
        logger.info(f"Agent {self.zi_agent.id} received message: {message}")
        if message.performative == Performative.INFORM and message.protocol == "double-auction":
            logger.info(f"Agent {self.zi_agent.id} processing trade information")
            self.log_trade_information(message.content)
            self._process_trade_execution(message.content)
        else:
            logger.warning(f"Agent {self.zi_agent.id} received unexpected message type: "
                           f"{message.performative} with protocol: {message.protocol}")
    
    def _process_trade_execution(self, trade_info: Dict[str, Any]) -> None:
        """
        Process the execution of a trade.

        Args:
            trade_info (Dict[str, Any]): Information about the executed trade.
        """
        trade = Trade(
            trade_id=trade_info['trade_id'],
            bid=Order(agent_id=self.zi_agent.id, price=trade_info['price'], 
                      quantity=trade_info['quantity'], is_buy=True),
            ask=Order(agent_id=self.zi_agent.id, price=trade_info['price'], 
                      quantity=trade_info['quantity'], is_buy=False),
            price=trade_info['price'],
            quantity=trade_info['quantity'],
            round=trade_info['round']
        )
        self.finalize_trade(trade)
    
    def log_interaction(self, round_num: int, task_prompt: str, response: str) -> None:
        """
        Log an interaction with the LLM.

        Args:
            round_num (int): The current round number.
            task_prompt (str): The prompt given to the LLM.
            response (str): The response from the LLM.
        """
        interaction = {
            "round": round_num,
            "task": task_prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.append(interaction)

    def get_recent_memories(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the n most recent memories.

        Args:
            n (int): The number of recent memories to retrieve.

        Returns:
            List[Dict[str, Any]]: The n most recent memories.
        """
        return self.memory[-n:]

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format a list of memories into a string.

        Args:
            memories (List[Dict[str, Any]]): The list of memories to format.

        Returns:
            str: A formatted string representation of the memories.
        """
        if not memories:
            return ""
        formatted_memories = []
        for memory in memories:
            formatted_memories.append(f"Round {memory['round']}:\n"
                                      f"Task: {memory['task'].split('Recent market activities:')[0].strip()}\n"
                                      f"Response: {memory['response']}\n")
        return "\n".join(formatted_memories)  
    
    def _get_market_info(self, market_info: MarketInfo) -> str:
        """
        Get a string representation of the current market information.

        Args:
            market_info (MarketInfo): The current market information.

        Returns:
            str: A formatted string containing the market information.
        """
        return f"""
        Current Cash: {round(self.zi_agent.allocation.cash, 2)}
        Current Goods: {self.zi_agent.allocation.goods}
        Last Trade Price: {round(market_info.last_trade_price, 2) if market_info.last_trade_price is not None else 'N/A'}
        Average Market Price: {round(market_info.average_price, 2) if market_info.average_price is not None else 'N/A'}
        Total Trades: {market_info.total_trades}
        Current Round: {market_info.current_round}
        Base Value/Cost: {round(self.zi_agent.preference_schedule.get_value(self.zi_agent.allocation.goods + 1), 2)}
        """

    def finalize_trade(self, trade: Trade) -> None:
        """
        Finalize a trade by updating the agent's state.

        Args:
            trade (Trade): The trade to finalize.
        """
        self.zi_agent.finalize_trade(trade)

    def log_trade_information(self, trade_info: Dict[str, Any]) -> None:
        """
        Log information about a trade.

        Args:
            trade_info (Dict[str, Any]): Information about the trade.
        """
        logger.info(f"Agent {self.zi_agent.id} logged trade information: {trade_info}")

    def calculate_trade_surplus(self, trade: Trade) -> float:
        """
        Calculate the surplus from a trade.

        Args:
            trade (Trade): The trade to calculate surplus for.

        Returns:
            float: The calculated trade surplus.
        """
        return self.zi_agent.calculate_trade_surplus(trade)

    def calculate_individual_surplus(self) -> float:
        """
        Calculate the individual surplus of the agent.

        Returns:
            float: The calculated individual surplus.
        """
        return self.zi_agent.calculate_individual_surplus()

    def plot_order_history(self) -> None:
        """
        Plot the order history of the agent.
        """
        self.zi_agent.plot_order_history()
