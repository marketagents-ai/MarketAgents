from datetime import datetime
import json
import logging
from typing import List, Optional, Dict, Any
from colorama import Fore, Style

from pydantic import Field, ValidationError

from acl_message.acl_message import ACLMessage, AgentID, Performative
from econ_agents.econ_agent import EconomicAgent, create_economic_agent
from base_agent.agent import Agent as LLMAgent
from market_agent.market_schemas import DoubleAuctionMessage, MarketActionSchema
from base_agent.aiutilities import LLMConfig

logger = logging.getLogger(__name__)

MARKET_ADDRESS = "market_orchestrator_node"


class MarketAgent(LLMAgent, EconomicAgent):
    """
    Represents a market agent that combines LLM-based decision making with economic characteristics.

    Attributes:
        memory (List[Dict[str, Any]]): List to store agent's memory of interactions.
        use_llm (bool): Flag to determine whether to use LLM for decision making.
        address (str): Agent's address in the market.
    """

    memory: List[Dict[str, Any]] = Field(default_factory=list)
    use_llm: bool = Field(default=False, description="Whether to use LLM for decision making")
    address: str = Field(default="", description="Agent's address")

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
            noise_factor (float, optional): Noise factor for preference schedule. Defaults to 0.1.
            max_relative_spread (float, optional): Maximum relative spread. Defaults to 0.2.
            llm_config (Optional[LLMConfig], optional): Configuration for LLM agent. Defaults to None.

        Returns:
            MarketAgent: A new instance of MarketAgent.
        """
        econ_agent = create_economic_agent(agent_id, is_buyer, num_units, base_value, initial_cash, initial_goods,
                                           "step", noise_factor, max_relative_spread)
        
        role = "buyer" if is_buyer else "seller"
        llm_agent = LLMAgent(role=role, llm_config=llm_config or LLMConfig(), 
                             output_format=MarketActionSchema.model_json_schema())

        return cls(is_buyer=is_buyer, preference_schedule=econ_agent.preference_schedule,
                   endowment=econ_agent.endowment, utility_function=econ_agent.utility_function,
                   max_relative_spread=econ_agent.max_relative_spread, use_llm=use_llm, 
                   address=f"agent_{agent_id}_address", **llm_agent.dict())

    def generate_bid(self, market_info: Dict[str, Any], round_num: int) -> Optional[ACLMessage]:
        if self.use_llm:
            return self._generate_llm_bid(market_info, round_num)
        else:
            return self._generate_econ_bid(market_info, round_num)

    def _generate_llm_bid(self, market_info: Dict[str, Any], round_num: int) -> Optional[ACLMessage]:
        market_info_str = self._get_market_info(market_info)
        recent_memories = self.get_recent_memories(2)
        memory_str = self._format_memories(recent_memories)

        task_prompt = f"Generate a market action based on the following market information: {market_info_str}"
        
        if memory_str:
            task_prompt += f"\n\nRecent market activities:\n{memory_str}"
        
        llm_response = self.execute(task_prompt)

        self.log_interaction(round_num, task_prompt, llm_response)

        try:
            market_action = MarketActionSchema(**llm_response)
        except ValidationError as e:
            logger.error(f"{Fore.RED}Validation error in LLM response: {e}{Style.RESET_ALL}")
            return None

        if market_action.action == "hold" or market_action.bid is None:
            return None
        
        return self._create_acl_message(market_action.bid.acl_message.action, 
                                        market_action.bid.acl_message.price, 1)

    def _generate_econ_bid(self, market_info: Dict[str, Any], round_num: int) -> Optional[ACLMessage]:
        econ_bid = self.generate_bid(market_info)
        if econ_bid is None:
            return None
        return self._create_acl_message("bid" if self.is_buyer else "ask", econ_bid["price"], econ_bid["quantity"])

    def _create_acl_message(self, action: str, price: float, quantity: int) -> ACLMessage:
        sender_id = AgentID(name=str(self.id), address=self.address)
        receiver_id = AgentID(name="market", address=MARKET_ADDRESS)

        content = DoubleAuctionMessage(
            action=action,
            price=price,
            quantity=quantity
        )

        return ACLMessage(
            performative=Performative.PROPOSE,
            sender=sender_id,
            receivers=[receiver_id],
            content=content,
            protocol="double-auction",
            ontology="market-ontology",
            conversation_id=f"{action}-{datetime.now().isoformat()}"
        )

    def receive_message(self, message: ACLMessage) -> None:
        logger.info(f"{Fore.BLUE}Agent {self.id} received message: {message}{Style.RESET_ALL}")
        if message.performative == Performative.INFORM and message.protocol == "double-auction":
            logger.info(f"{Fore.GREEN}Agent {self.id} processing trade information{Style.RESET_ALL}")
            self.log_trade_information(message.content)
            self.finalize_trade(message.content)
        else:
            logger.warning(f"{Fore.YELLOW}Agent {self.id} received unexpected message type: "
                           f"{message.performative} with protocol: {message.protocol}{Style.RESET_ALL}")
    
    def log_interaction(self, round_num: int, task_prompt: str, response: str) -> None:
        interaction = {
            "round": round_num,
            "task": task_prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.append(interaction)

    def get_recent_memories(self, n: int) -> List[Dict[str, Any]]:
        return self.memory[-n:]

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        formatted_memories = []
        for memory in memories:
            formatted_memories.append(f"Round {memory['round']}:\n"
                                      f"Task: {memory['task'].split('Recent market activities:')[0].strip()}\n"
                                      f"Response: {memory['response']}\n")
        return "\n".join(formatted_memories)  
    
    def _get_market_info(self, market_info: Dict[str, Any]) -> str:
        return f"""
        Current Cash: {round(self.endowment.cash, 2)}
        Current Goods: {self.endowment.goods}
        Last Trade Price: {round(market_info.get('last_trade_price', 0), 2)}
        Average Market Price: {round(market_info.get('average_price', 0), 2)}
        Total Trades: {market_info.get('total_trades', 0)}
        Current Round: {market_info.get('current_round', 0)}
        Base Value/Cost: {round(self.preference_schedule.get_value(self.endowment.goods + 1), 2)}
        """

    def log_trade_information(self, trade_info: Dict[str, Any]) -> None:
        logger.info(f"{Fore.GREEN}Agent {self.id} logged trade information: {trade_info}{Style.RESET_ALL}")
