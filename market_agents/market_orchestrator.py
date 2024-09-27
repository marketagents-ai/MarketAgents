from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent
from market_agents.economics.econ_models import Basket, Good, Trade, Bid, Ask
from market_agents.economics.equilibrium import Equilibrium
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionAction, GlobalAuctionAction, AuctionGlobalObservation

#import llm context and ai utiliteis from 
from market_agents.environments.environment import EnvironmentStep
from market_agents.economics.analysis import analyze_and_plot_market_results

from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import LLMPromptContext, LLMOutput
from market_agents.simple_agent import SimpleAgent
from typing import List, Tuple, Optional, Dict,Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketOrchestrator():
    def __init__(self, agents: Union[List[SimpleAgent], List[EconomicAgent]], markets: List[MultiAgentEnvironment], ai_utils: ParallelAIUtilities):
        self.agents = agents
        self.markets = markets
        self.ai_utils = ai_utils
        self.markets_dict = self.create_markets_dict(markets)
        self.agents_dict = self.creat_agents_dict(agents)
        self.failed_actions : List[LLMOutput] = []
        typed_agents = [agent for agent in agents if isinstance(agent, EconomicAgent)]
        typed_mechanisms = [market.mechanism for market in markets if isinstance(market.mechanism, DoubleAuction)]
        self.equilibrium = Equilibrium(agents=typed_agents, goods=[mechanism.good_name for mechanism in typed_mechanisms])

    def create_markets_dict(self,markets: List[MultiAgentEnvironment]) -> Dict[str, MultiAgentEnvironment]:
        markets_dict = {}
        for market in markets:
            if not isinstance(market.mechanism, DoubleAuction):
                raise ValueError(f"Market {market.mechanism} is not a DoubleAuction")
            markets_dict[market.mechanism.good_name] = market
        return markets_dict
    
    def creat_agents_dict(self,agents: Union[List[SimpleAgent], List[EconomicAgent]]) -> Dict[str, Union[SimpleAgent, EconomicAgent]]:
        agents_dict = {}
        for agent in agents:
            agents_dict[agent.id] = agent
        return agents_dict
    
    def get_zero_intelligence_agents(self):
        return [agent for agent in self.agents if not isinstance(agent, SimpleAgent)]
    
    def get_llm_agents(self):
        return [agent for agent in self.agents if isinstance(agent, LLMPromptContext)]
    
    def get_agent(self, agent_id: str) -> Union[SimpleAgent, EconomicAgent]:
        return self.agents_dict[agent_id]
    
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
    
    async def run_parallel_ai_completion(self, prompts: List[SimpleAgent],update_history:bool=True) -> List[LLMOutput]:
        typed_prompts : List[LLMPromptContext] = [p for p in prompts if isinstance(p, LLMPromptContext)]
        return await self.ai_utils.run_parallel_ai_completion(typed_prompts,update_history)
    
    def validate_output(self, output: LLMOutput, good_name: str) -> Union[Bid, Ask, None]:
        agend_id = output.source_id
        agent = self.get_agent(agend_id)
        if agent is None:
            raise ValueError(f"Agent {agend_id} not found")
        market_action = None
        if output.json_object is not None:
            if output.json_object.name == "Bid":
                bid = Bid.model_validate(output.json_object.object)
                current_value = agent.get_current_value(good_name)
                print(f"bid price: {bid.price}, current_value: {current_value}")
                if current_value is not None and bid.price < current_value:
                    print(f"adding bid to pending orders")
                    market_action = bid
                else:
                    print(f"not adding bid to pending orders")
            elif output.json_object.name == "Ask":
                ask = Ask.model_validate(output.json_object.object)
                current_cost = agent.get_current_cost(good_name)
                print(f"ask price: {ask.price}, current_cost: {current_cost}")
                if current_cost is not None and ask.price > current_cost:
                    print(f"adding ask {ask}  from agent {agent.id} with current_cost {current_cost} to pending orders")
                    market_action = ask
                else:
                    print(f"not adding ask to pending orders")
        if market_action is not None:
            agent.pending_orders.setdefault(good_name, []).append(market_action)
        else:
            self.failed_actions.append(output)
        return market_action
    
    def create_local_actions_llm(self, good_name: str, llm_outputs: List[LLMOutput]) -> Dict[str, AuctionAction]:
        actions = {}
        
        for output in llm_outputs:
            market_action = self.validate_output(output, good_name)
            if market_action is not None:
                actions[output.source_id] = AuctionAction(agent_id=output.source_id, action=market_action)
           
        return actions
    
    def execute_trade(self, trade: Trade) -> float:
        buyer = next(agent for agent in self.agents if agent.id == trade.buyer_id)
        seller = next(agent for agent in self.agents if agent.id == trade.seller_id)
        print(f"buyer_id: {trade.buyer_id}, seller_id: {trade.seller_id}")
        if buyer is None or seller is None:
            raise ValueError(f"Trade {trade} has invalid agent IDs")
        
        buyer_utility_before = buyer.calculate_utility(buyer.endowment.current_basket)
        seller_utility_before = seller.calculate_utility(seller.endowment.current_basket)
        buyer.process_trade(trade)
        seller.process_trade(trade)
        buyer_utility_after = buyer.calculate_utility(buyer.endowment.current_basket)
        seller_utility_after = seller.calculate_utility(seller.endowment.current_basket)
        trade_surplus = buyer_utility_after - buyer_utility_before + seller_utility_after - seller_utility_before
        return trade_surplus
        
    
    def process_trades(self, global_observation: AuctionGlobalObservation) -> List[float]:
        surplus = []
        new_trades = global_observation.all_trades
        new_trades.sort(key=lambda x: x.trade_id)
        for trade in new_trades:
            trade_surplus = self.execute_trade(trade)
            if trade_surplus is not None:
                surplus.append(trade_surplus)
        return surplus
    
    def update_llm_state(self, global_observation: AuctionGlobalObservation):
        for agent in self.get_llm_agents():
            if agent.id in global_observation.observations:
                agent.update_state(global_observation.observations[agent.id])

    
    async def run_auction_step(self, good_name: str) -> Tuple[EnvironmentStep, List[float]]:
        environment = self.markets_dict[good_name]
        llm_agents = self.get_llm_agents()
        
        # Generate actions for LLM agents
        llm_outputs = await self.run_parallel_ai_completion(llm_agents, update_history=True)  # Await here
        llm_actions = self.create_local_actions_llm(good_name, llm_outputs)  # Pass llm_outputs
        print(f"llm_actions: {llm_actions}")
        
        # Generate actions for zero-intelligence agents
        zi_actions = self.create_local_actions_zero_intelligence(good_name)
        
        # Combine all actions
        all_actions = {**llm_actions, **zi_actions}
        
        # Create global action and step the environment
        global_action = GlobalAuctionAction(actions=all_actions)
        step_result = environment.step(global_action)
        assert isinstance(step_result.global_observation, AuctionGlobalObservation)
        
        # Process trades and update agents
        surplus = self.process_trades(step_result.global_observation)
        self.update_llm_state(step_result.global_observation)
        return step_result, surplus
    
    async def simulate_market(self, max_rounds: int, good_name: str):
        relevant_agents = [agent for agent in self.agents if good_name in agent.cost_schedules.keys() or good_name in agent.value_schedules.keys()]
        
        all_trades : List[Trade] = []
        per_trade_surplus : List[float] = []
        per_trade_quantities : List[int] = []

        for round in range(max_rounds):
            logger.info(f"Round {round + 1}")
            
            # Run one step of the orchestrator
            step_result, surplus = await self.run_auction_step(good_name)
            per_trade_surplus.extend(surplus)
            # Process trades
            global_observation  = step_result.global_observation
            assert isinstance(global_observation, AuctionGlobalObservation)
            new_trades = global_observation.all_trades
            all_trades.extend(new_trades)
            quantities = [trade.quantity for trade in new_trades]
            per_trade_quantities.extend(quantities)
            # Update cumulative quantities and surplus
            
           
            
            if new_trades:
                logger.info(f"Trades executed in this round: {len(new_trades)}")
                for trade in new_trades:
                    logger.info(f"Trade executed at price: {trade.price}")
            
            if step_result.done:
                logger.info("Market simulation completed.")
                break
        
        # Reset pending orders for all agents
        for agent in relevant_agents:
            agent.reset_all_pending_orders()
        cumulative_surplus = [sum(per_trade_surplus[:i+1]) for i in range(len(per_trade_surplus))]
        cumulative_quantities = [sum(per_trade_quantities[:i+1]) for i in range(len(per_trade_quantities))]
        assert len(cumulative_quantities) == len(cumulative_surplus)
        # Generate market report
        self.generate_market_report(all_trades, relevant_agents, good_name, max_rounds, cumulative_quantities, cumulative_surplus)

    def generate_market_report(self, trades: List[Trade], agents: List[Union[SimpleAgent, EconomicAgent]], 
                               good_name: str, max_rounds: int, cumulative_quantities: List[int], cumulative_surplus: List[float]):
        analyze_and_plot_market_results(
            trades=trades,
            agents=agents,
            equilibrium=self.equilibrium,
            goods=[good_name],
            max_rounds=max_rounds,
            cumulative_quantities=cumulative_quantities,
            cumulative_surplus=cumulative_surplus
        )


