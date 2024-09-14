from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent
from market_agents.economics.econ_models import Basket, Good, Trade, Bid, Ask
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionAction, GlobalAuctionAction, AuctionGlobalObservation

#import llm context and ai utiliteis from 

from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import LLMPromptContext, LLMOutput
from market_agents.simple_agent import SimpleAgent
from typing import List, Tuple, Optional, Dict,Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketOrchestrator():
    def __init__(self, agents: List[Union[SimpleAgent, EconomicAgent]], markets: List[MultiAgentEnvironment], ai_utils: ParallelAIUtilities):
        self.agents = agents
        self.markets = markets
        self.ai_utils = ai_utils
        self.markets_dict = self.create_markets_dict(markets)


    def create_markets_dict(self,markets: List[MultiAgentEnvironment]):
        markets_dict = {}
        for market in markets:
            if not isinstance(market.mechanism, DoubleAuction):
                raise ValueError(f"Market {market.mechanism} is not a DoubleAuction")
            markets_dict[market.mechanism.good_name] = market
        return markets_dict
    
    def get_zero_intelligence_agents(self):
        return [agent for agent in self.agents if not isinstance(agent, SimpleAgent)]
    
    def get_llm_agents(self):
        return [agent for agent in self.agents if isinstance(agent, LLMPromptContext)]
    
    def create_local_actions_zero_intelligence(self, good_name: str) -> Dict[str, AuctionAction]:
        actions = {}
        agents = [agent for agent in self.get_zero_intelligence_agents() if good_name in agent.cost_schedules.keys() or good_name in agent.value_schedules.keys()]
        for agent in agents:
            bid = agent.generate_bid(good_name)
            ask = agent.generate_ask(good_name)
            if bid:
                actions[agent.id] = AuctionAction(agent_id=agent.id, action=bid)
            elif ask:
                actions[agent.id] = AuctionAction(agent_id=agent.id, action=ask)
        return actions
    
    async def run_parallel_ai_completion(self, prompts: List[SimpleAgent]) -> List[LLMOutput]:
        typed_prompts : List[LLMPromptContext] = [p for p in prompts if isinstance(p, LLMPromptContext)]
        return await self.ai_utils.run_parallel_ai_completion(typed_prompts)
    
    def create_local_actions_llm(self, good_name: str) -> Dict[str, AuctionAction]:
        #filter agents for those that have the good name in their cost or sell schedule
        agents = [agent for agent in self.get_llm_agents() if good_name in agent.cost_schedules.keys() or good_name in agent.value_schedules.keys()]
        llm_outputs = self.run_parallel_ai_completion(agents)
        assert isinstance(llm_outputs, list)
        typed_outputs : List[LLMOutput] = [p for p in llm_outputs if isinstance(p, LLMOutput)]
        actions = {}
        for output,agent in zip(typed_outputs,agents):
            market_action = None
            if output.json_object is not None and output.json_object.name == "Bid":
                market_action = Bid.model_validate(output.json_object)
                agent.add_chat_turn_history(output)
            elif output.json_object is not None and output.json_object.name == "Ask":
                market_action = Ask.model_validate(output.json_object)
            if market_action is not None:
                agent.add_chat_turn_history(output)
                actions[agent.id] = AuctionAction(agent_id=agent.id, action=market_action)
        return actions
    
    def execute_trade(self, trade: Trade) -> Optional[float]:
        buyer = next(agent for agent in self.agents if agent.id == trade.buyer_id)
        seller = next(agent for agent in self.agents if agent.id == trade.seller_id)
        if buyer is None or seller is None:
            raise ValueError(f"Trade {trade} has invalid agent IDs")
        if buyer.would_accept_trade(trade) and seller.would_accept_trade(trade):
            buyer_utility_before = buyer.calculate_utility(buyer.endowment.current_basket)
            seller_utility_before = seller.calculate_utility(seller.endowment.current_basket)
            buyer.process_trade(trade)
            seller.process_trade(trade)
            buyer_utility_after = buyer.calculate_utility(buyer.endowment.current_basket)
            seller_utility_after = seller.calculate_utility(seller.endowment.current_basket)
            trade_surplus = buyer_utility_after - buyer_utility_before + seller_utility_after - seller_utility_before
            return trade_surplus
        else:
            raise ValueError(f"Trade {trade} was not accepted by either buyer {buyer.would_accept_trade(trade)} and seller {seller.would_accept_trade(trade)}")
    
    def process_trades(self, global_observation: AuctionGlobalObservation) -> List[float]:
        surplus = []
        new_trades = global_observation.all_trades
        new_trades.sort(key=lambda x: x.trade_id)
        for trade in new_trades:
            trade_surplus = self.execute_trade(trade)
            if trade_surplus is not None:
                surplus.append(trade_surplus)
        return surplus
    
    def update_llm_new_message(self, global_observation: AuctionGlobalObservation):
        for agent in self.get_llm_agents():
            agent.update_local(global_observation.observations[agent.id])

    
    def run_auction(self, good_name: str, max_rounds: int) -> Tuple[List[Trade], Dict[int,List[float]]]:
        environment = self.markets_dict[good_name]    
        all_trades = []
        surplus_by_round : Dict[int,List[float]] = {}

        for round_num in range(max_rounds):
            zero_intelligence_actions = self.create_local_actions_zero_intelligence(good_name)
            llm_actions = self.create_local_actions_llm(good_name)
            actions = {**zero_intelligence_actions, **llm_actions}
            if not actions:
                logger.info(f"No more actions generated. Ending auction at round {round_num}")
                break

            global_action = GlobalAuctionAction(actions=actions)
            environment_step = environment.step(global_action)
            
            assert isinstance(environment_step.global_observation, AuctionGlobalObservation)
            all_trades.extend(environment_step.global_observation.all_trades)
            surplus_by_round[round_num] = self.process_trades(environment_step.global_observation)
            self.update_llm_new_message(environment_step.global_observation)
                

        return all_trades, surplus_by_round


