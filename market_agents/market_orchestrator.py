from market_agents.economics.econ_agent import EconomicAgent
from market_agents.economics.econ_models import Basket, Good, Trade, Bid, Ask
from market_agents.economics.equilibrium import Equilibrium, EquilibriumResults
from market_agents.economics.scenario import Scenario
from market_agents.environments.environment import MultiAgentEnvironment, EnvironmentStep
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionAction, GlobalAuctionAction, AuctionGlobalObservation, AuctionMarket, MarketSummary
from market_agents.economics.analysis import analyze_and_plot_market_results
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import LLMPromptContext, LLMOutput
from market_agents.simple_agent import SimpleAgent
from pydantic import BaseModel, Field, computed_field
from typing import List, Tuple, Optional, Dict, Union, Set, Any
import logging
import json
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketStep(BaseModel):
    market_id: str
    step_number: int
    participating_agent_ids: Set[str]
    equilibrium: EquilibriumResults
    episode: int
    trades: List[Trade]
    market_summary: MarketSummary
    surplus: List[float]


class MarketOrchestratorState(BaseModel):
    name:str = Field(default="orchestrator_state")
    steps: List[MarketStep] = Field(default_factory=list)

    def add_step(self, market_id: str,
                 participating_agent_ids: Set[str],
                 equilibrium_results: EquilibriumResults,
                 episode: int,
                 trades: List[Trade],
                 surplus: List[float],
                 market_summary: MarketSummary):
        step_number = self.get_market_step_count(market_id) + 1
        new_step = MarketStep(
            market_id=market_id,
            step_number=step_number,
            participating_agent_ids=participating_agent_ids,
            equilibrium=equilibrium_results,
            episode=episode,
            trades=trades,
            surplus=surplus,
            market_summary=market_summary
        )
        self.steps.append(new_step)

    @computed_field
    @property
    def market_step_counts(self) -> Dict[str, int]:
        return {market_id: self.get_market_step_count(market_id) 
                for market_id in self.get_market_ids()}

    @computed_field
    @property
    def agent_participation_counts(self) -> Dict[str, int]:
        agent_counts = {}
        for step in self.steps:
            for agent_id in step.participating_agent_ids:
                agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        return agent_counts

    def get_market_step_count(self, market_id: str) -> int:
        return sum(1 for step in self.steps if step.market_id == market_id)

    def get_market_ids(self) -> Set[str]:
        return {step.market_id for step in self.steps}

    def get_participating_agents(self, market_id: str) -> Set[str]:
        return {agent_id 
                for step in self.steps 
                if step.market_id == market_id 
                for agent_id in step.participating_agent_ids}

    def get_market_history(self, market_id: str) -> List[MarketStep]:
        return [step for step in self.steps if step.market_id == market_id]

    def save_to_json(self, folder_path: str) -> str:
        # Create folder if it doesn't exist
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name.replace(' ', '_')}_{timestamp}.json"
        file_path = os.path.join(folder_path, filename)

        # Convert to dict and save as JSON
        data = self.model_dump(mode='json')
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"State saved to {file_path}")
        return file_path

    @classmethod
    def load_from_json(cls, file_path: str) -> 'MarketOrchestratorState':
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert JSON data back to MarketOrchestratorState
        return cls.model_validate(data)

class MarketOrchestrator:
    def __init__(self, 
                llm_agents: List[SimpleAgent],
                goods: List[str],
                ai_utils: ParallelAIUtilities = ParallelAIUtilities(),
                max_rounds: int = 10,
                scenario: Optional[Scenario] = None):
        self.llm_agents = llm_agents
        self.goods = goods
        self.max_rounds = max_rounds
        self.markets = self.create_markets(self.goods)
        self.ai_utils = ai_utils
        self.scenario = scenario
        self.agents_dict, self.llm_agents_dict, self.zi_agents_dict = self.create_agents_dicts()
        self.agents = list(self.agents_dict.values())
        self.failed_actions: List[LLMOutput] = []
        if self.scenario:
            name = self.scenario.name
        else:
            name = "orchestrator_state"
        self.state = MarketOrchestratorState(name=name)

    def get_current_equilibrium(self) -> Equilibrium:
        typed_agents = [agent for agent in self.agents if isinstance(agent, EconomicAgent)]
        return Equilibrium(agents=typed_agents, goods=self.goods)

    def create_markets(self, goods: List[str]) -> Dict[str, AuctionMarket]:
        markets_dict = {}
        for good in goods:
            markets_dict[good] = AuctionMarket(name=f"Auction {good} Market",mechanism = DoubleAuction(good_name=good, max_rounds=self.max_rounds))
        return markets_dict
    
    def create_agents_dicts(self) -> Tuple[Dict[str, Union[SimpleAgent, EconomicAgent]],
                                            Dict[str, SimpleAgent], 
                                            Dict[str, EconomicAgent]]:
        zi_dict = {}
        if self.scenario:
            zi_agents = self.scenario.agents
            zi_dict = {agent.id: agent for agent in zi_agents}
        llm_dict = {agent.id: agent for agent in self.llm_agents}
        return {**zi_dict, **llm_dict}, llm_dict, zi_dict
    
    def get_zero_intelligence_agents(self) -> List[EconomicAgent]:
        return list(self.zi_agents_dict.values())
    
    def get_llm_agents(self) -> List[SimpleAgent]:
        return list(self.llm_agents_dict.values())
    
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
    
    async def run_parallel_ai_completion(self, prompts: List[SimpleAgent], update_history: bool = True) -> List[LLMOutput]:
        typed_prompts: List[LLMPromptContext] = [p for p in prompts if isinstance(p, LLMPromptContext)]
        return await self.ai_utils.run_parallel_ai_completion(typed_prompts, update_history)
    
    def validate_output(self, output: LLMOutput, good_name: str) -> Union[Bid, Ask, None]:
        agent_id = output.source_id
        agent = self.get_agent(agent_id)
        if agent is None:
            raise ValueError(f"Agent {agent_id} not found")
        market_action = None
        if output.json_object is not None:
            if "Bid" in output.json_object.name:
                bid = Bid.model_validate(output.json_object.object)
                current_value = agent.get_current_value(good_name)
                print(f"bid price: {bid.price}, current_value: {current_value}")
                if current_value is not None and bid.price < current_value:
                    print(f"adding bid to pending orders")
                    market_action = bid
                else:
                    print(f"not adding bid to pending orders")
            elif "Ask" in output.json_object.name:
                ask = Ask.model_validate(output.json_object.object)
                current_cost = agent.get_current_cost(good_name)
                print(f"ask price: {ask.price}, current_cost: {current_cost}")
                if current_cost is not None and ask.price > current_cost:
                    print(f"adding ask {ask} from agent {agent.id} with current_cost {current_cost} to pending orders")
                    market_action = ask
                else:
                    print(f"not adding ask to pending orders")
        if market_action is not None:
            agent.pending_orders.setdefault(good_name, []).append(market_action)
        else:
            self.failed_actions.append(output)
        return market_action
    
    def generate_market_report(self, trades: List[Trade], agents: List[Union[SimpleAgent, EconomicAgent]], 
                               good_name: str, max_rounds: int, cumulative_quantities: List[int], cumulative_surplus: List[float]):
        analyze_and_plot_market_results(
            trades=trades,
            agents=agents,
            equilibrium=self.get_current_equilibrium(),
            goods=[good_name],
            max_rounds=max_rounds,
            cumulative_quantities=cumulative_quantities,
            cumulative_surplus=cumulative_surplus
        )
    
    def create_local_actions_llm(self, good_name: str, llm_outputs: List[LLMOutput]) -> Dict[str, AuctionAction]:
        actions = {}
        for output in llm_outputs:
            market_action = self.validate_output(output, good_name)
            if market_action is not None:
                actions[output.source_id] = AuctionAction(agent_id=output.source_id, action=market_action)
        return actions
    
    def execute_trade(self, trade: Trade) -> float:
        buyer = self.get_agent(trade.buyer_id)
        seller = self.get_agent(trade.seller_id)
        # print(f"buyer_id: {trade.buyer_id}, seller_id: {trade.seller_id}")
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
        environment = self.markets[good_name]
        llm_agents = self.get_llm_agents()
        
        # Generate actions for LLM agents
        llm_outputs = await self.run_parallel_ai_completion(llm_agents, update_history=True)
        llm_actions = self.create_local_actions_llm(good_name, llm_outputs)
        
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
        
        # Update state
        participating_agent_ids = set(all_actions.keys())
        self.state.add_step(good_name,
                            participating_agent_ids,
                            equilibrium_results=self.get_current_equilibrium().equilibrium[good_name],
                            episode=self.scenario._current_episode if self.scenario else 0,
                            trades=step_result.global_observation.all_trades,
                            surplus=surplus,
                            market_summary=step_result.global_observation.market_summary,
                            )
        
        return step_result, surplus
    
    async def run_auction_episode(self, max_rounds: int, good_name: str,report:bool=True,reset_endowments:bool=True):
        relevant_agents = [agent for agent in self.agents if good_name in agent.cost_schedules.keys() or good_name in agent.value_schedules.keys()]
        
        all_trades: List[Trade] = []
        per_trade_surplus: List[float] = []
        per_trade_quantities: List[int] = []

        for round in range(max_rounds):
            round_surplus : List[float]= []
            logger.info(f"Round {round + 1}")
            
            # Run one step of the orchestrator
            step_result, surplus = await self.run_auction_step(good_name)
            per_trade_surplus.extend(surplus)
            # Process trades
            global_observation = step_result.global_observation
            assert isinstance(global_observation, AuctionGlobalObservation)
            new_trades = global_observation.all_trades
            all_trades.extend(new_trades)
            quantities = [trade.quantity for trade in new_trades]
            per_trade_quantities.extend(quantities)
            
            if new_trades:
                logger.info(f"Trades executed in this round: {len(new_trades)}")
                for trade in new_trades:
                    logger.info(f"Trade executed at price: {trade.price}")
            
            if step_result.done:
                logger.info("Market simulation completed.")
                break
        if self.scenario:
            self.scenario.next_episode()
        # Reset pending orders for all agents
        for agent in relevant_agents:
            agent.reset_all_pending_orders()
        
        cumulative_surplus = [sum(per_trade_surplus[:i+1]) for i in range(len(per_trade_surplus))]
        cumulative_quantities = [sum(per_trade_quantities[:i+1]) for i in range(len(per_trade_quantities))]
        assert len(cumulative_quantities) == len(cumulative_surplus)
        
        # Generate market report
        if report:
            self.generate_market_report(all_trades, relevant_agents, good_name, max_rounds, cumulative_quantities, cumulative_surplus)
        #reset endowments for llm agents
        if reset_endowments:
            relevant_llm_agents = [agent for agent in relevant_agents if isinstance(agent, SimpleAgent)]
            for agent in relevant_llm_agents:
                    agent.archive_endowment()

    async def run_scenario(self):
        if self.scenario:
            for episode in range(self.scenario.num_episodes):
                for good in self.scenario.goods:
                    await self.run_auction_episode(self.markets[good].mechanism.max_rounds, good)
        else:
            for good in self.goods:
                await self.run_auction_episode(self.markets[good].mechanism.max_rounds, good)

