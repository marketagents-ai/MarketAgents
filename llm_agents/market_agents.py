from collections import defaultdict
from datetime import datetime
import json
import random
import logging
import traceback
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from ziagents import ZIAgent, Order, Trade
from agents import Agent as LLMAgent
from schema import MarketActionSchema, DoubleAuctionMessage
from acl_message import ACLMessage, AgentID, Performative

logger = logging.getLogger(__name__)

MARKET_ADDRESS = "market_orchestrator_node"

class MarketAgent(BaseModel):
    zi_agent: ZIAgent
    llm_agent: LLMAgent
    memory: Dict[int, Dict[str, Any]] = Field(default_factory=lambda: defaultdict(dict))
    address: Optional[str] = None
    use_llm: bool = Field(default=False, description="Whether to use LLM for decision making")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, agent_id: int, is_buyer: bool, num_units: int, base_value: float, use_llm: bool,
               max_relative_spread: float = 0.2, llm_config: Dict[str, Any] = None, address: Optional[str] = None):
        zi_agent = ZIAgent.generate(agent_id, is_buyer, num_units, base_value, max_relative_spread)
        
        role = "buyer" if is_buyer else "seller"
        #system_prompt = f"You are a {role} agent in a double auction market. Your goal is to maximize your profit."
        llm_agent = LLMAgent(role=role, llm_config=llm_config or {}, output_format="MarketActionSchema")

        return cls(zi_agent=zi_agent, llm_agent=llm_agent, address=address, use_llm=use_llm)

    def generate_bid(self, market_info: dict, round_num: int) -> Optional[Order]:
        try:
            if self.use_llm:
                acl_message = self._generate_llm_bid(market_info, round_num)
                if acl_message:
                    order = self._acl_message_to_order(acl_message)
                    logger.info(f"Generated order from ACL message: {order}")
                    return order
                else:
                    logger.info("No ACL message generated, returning None")
                    return None
            else:
                return self.zi_agent.generate_bid()
        except Exception as e:
            logger.error(f"Error in generate_bid: {e}")
            logger.error(traceback.format_exc())
            return None

    def _generate_llm_bid(self, market_info: dict, round_num: int) -> Optional[ACLMessage[DoubleAuctionMessage]]:
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
            logger.info(json.dumps(json.loads(llm_response), indent=2))

            market_action = MarketActionSchema.parse_raw(llm_response)
            if market_action.action == "hold":
                logger.info("LLM decided to hold, returning None")
                return None
            
            bid = market_action.bid

            if bid is None:
                logger.info("No bid in market action, returning None")
                return None

            # Set quantity to 1 for now
            bid.acl_message.quantity = 1

            sender_id = AgentID(name=str(self.zi_agent.id), address=self.address)
            receiver_id = AgentID(name="market", address=MARKET_ADDRESS)

            content = DoubleAuctionMessage(
                action=bid.acl_message.action,
                price=bid.acl_message.price,
                quantity=bid.acl_message.quantity
            )

            acl_message = ACLMessage(
                performative=Performative.PROPOSE,
                sender=sender_id,
                receivers=[receiver_id],
                content=content,
                protocol="double-auction",
                ontology="market-ontology",
                conversation_id=f"{content.action}-{datetime.now().isoformat()}"
            )

            logger.info(f"Generated ACL message: {acl_message}")
            return acl_message
        
        except Exception as e:
            logger.error(f"Error in _generate_llm_bid: {e}")
            logger.error(traceback.format_exc())
            return None

    def _acl_message_to_order(self, acl_message: ACLMessage) -> Order:
        try:
            is_buy = acl_message.content.action == "bid"
            agent_id = int(acl_message.sender.name)
            price = acl_message.content.price
            quantity = acl_message.content.quantity
            
            if is_buy:
                base_value = self.zi_agent.preference_schedule.get_value(self.zi_agent.allocation.goods + quantity)
                base_cost = None
                # Ensure buy price doesn't exceed base value
                if price > base_value:
                    logger.warning(f"Buy price {price} exceeds base value {base_value}. Adjusting to base value.")
                    price = base_value
            else:
                base_value = None
                base_cost = self.zi_agent.preference_schedule.get_cost(quantity)

            order = Order(
                agent_id=agent_id,
                price=price,
                quantity=quantity,
                is_buy=is_buy,
                base_value=base_value,
                base_cost=base_cost
            )
            return order
        except Exception as e:
            logger.error(f"Error in _acl_message_to_order: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def receive_message(self, message: ACLMessage):
        logger.info(f"Agent {self.zi_agent.id} received message: {message}")
        if message.performative == Performative.INFORM and message.protocol == "double-auction":
            logger.info(f"Agent {self.zi_agent.id} processing trade information")
            self.log_trade_information(message.content)
        else:
            logger.warning(f"Agent {self.zi_agent.id} received unexpected message type: {message.performative}")

    def log_trade_information(self, trade_info: dict):
        round_num = trade_info["round"]
        if "trades" not in self.memory[round_num]:
            self.memory[round_num]["trades"] = []
        
        self.memory[round_num]["trades"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "trade_executed",
            "trade_id": trade_info["trade_id"],
            "price": trade_info["price"],
            "quantity": trade_info["quantity"]
        })

    def log_interaction(self, round_num: int, task_prompt: str, response: str):
        self.memory[round_num] = {
            "round": round_num,
            "task": task_prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

    def get_recent_memories(self, n: int) -> List[Dict[str, Any]]:
        """Retrieve the n most recent rounds of memories."""
        sorted_rounds = sorted(self.memory.keys(), reverse=True)
        recent_rounds = sorted_rounds[:n]
        return [self.memory[round] for round in recent_rounds]


    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        formatted_memories = []
        for memory in memories:
            formatted_memories.append(f"Round {memory['round']}:\nTask: {memory['task'].split('Recent market activities:')[0].strip()}\nResponse: {memory['response']}\n")
        return "\n".join(formatted_memories)  
    
    def _get_market_info(self, market_info: dict) -> str:
        return f"""
        Current Cash: {round(self.zi_agent.allocation.cash, 2)}
        Current Goods: {self.zi_agent.allocation.goods}
        Last Trade Price: {round(market_info['last_trade_price'], 2) if market_info['last_trade_price'] is not None else 'N/A'}
        Average Market Price: {round(market_info['average_price'], 2) if market_info['average_price'] is not None else 'N/A'}
        Total Trades: {market_info['total_trades']}
        Current Round: {market_info['current_round']}
        Base Value/Cost: {round(self.zi_agent.preference_schedule.get_value(self.zi_agent.allocation.goods + 1), 2)}
        """

    def finalize_trade(self, trade: Trade):
        self.zi_agent.finalize_trade(trade)

    def respond_to_order(self, order: Order, accepted: bool):
        self.zi_agent.respond_to_order(order, accepted)

    def calculate_trade_surplus(self, trade: Trade) -> float:
        return self.zi_agent.calculate_trade_surplus(trade)

    def calculate_individual_surplus(self) -> float:
        return self.zi_agent.calculate_individual_surplus()

    def plot_order_history(self):
        self.zi_agent.plot_order_history()