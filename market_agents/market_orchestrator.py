from market_agents.economics.econ_agent import EconomicAgent, ZiFactory, ZiParams
from market_agents.economics.econ_models import Good, Trade, Bid, Ask,SavableBaseModel
from market_agents.economics.equilibrium import Equilibrium, EquilibriumResults
from market_agents.economics.scenario import Scenario
from market_agents.environments.environment import  EnvironmentStep
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionAction, GlobalAuctionAction, AuctionGlobalObservation, AuctionMarket, MarketSummary
from market_agents.economics.analysis import analyze_and_plot_market_results
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMPromptContext, LLMOutput, LLMConfig
from market_agents.simple_agent import SimpleAgent, create_simple_agents_from_zi
from pydantic import BaseModel, Field, computed_field
from typing import List, Tuple, Optional, Dict, Union, Set
import logging
from statistics import mean, stdev
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

class MarketStep(BaseModel):
    market_id: str
    step_number: int
    participating_agent_ids: List[str]
    equilibrium: EquilibriumResults
    episode: int
    trades: List[Trade]
    market_summary: MarketSummary
    surplus: List[float]


class EpisodeSummary(BaseModel):
    episode: int
    surplus: float
    theoretical_surplus: float
    efficiency: float
    quantity: int
    average_price: float
    min_price: float
    max_price: float
    std_price: float
    equilibrium_price: float
    equilibrium_quantity: int

class MarketOrchestratorState(SavableBaseModel):
    name: str = Field(default="orchestrator_state")
    steps: Dict[int, List[MarketStep]] = Field(default_factory=dict)

    def add_step(self, market_id: str,
                 participating_agent_ids: Set[str],
                 equilibrium_results: EquilibriumResults,
                 episode: int,
                 trades: List[Trade],
                 surplus: List[float],
                 market_summary: MarketSummary):
        step_number = self.get_market_step_count(market_id, episode) + 1
        new_step = MarketStep(
            market_id=market_id,
            step_number=step_number,
            participating_agent_ids=list(participating_agent_ids),
            equilibrium=equilibrium_results,
            episode=episode,
            trades=trades,
            surplus=surplus,
            market_summary=market_summary
        )
        if episode not in self.steps:
            self.steps[episode] = []
        self.steps[episode].append(new_step)

    @computed_field
    @property
    def market_step_counts(self) -> Dict[str, Dict[int, int]]:
        return {market_id: {episode: self.get_market_step_count(market_id, episode) 
                            for episode in self.steps.keys()}
                for market_id in self.get_market_ids()}

    @computed_field
    @property
    def agent_participation_counts(self) -> Dict[str, int]:
        agent_counts = {}
        for episode_steps in self.steps.values():
            for step in episode_steps:
                for agent_id in step.participating_agent_ids:
                    agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        return agent_counts

    @computed_field
    @property
    def surplus(self) -> Dict[int, float]:
        episode_surplus = {}
        for episode, episode_steps in self.steps.items():
            total_surplus = sum(sum(step.surplus) for step in episode_steps)
            episode_surplus[episode] = total_surplus
        return episode_surplus

    @computed_field
    @property
    def theoretical_surplus(self) -> Dict[int, float]:
        return {
            episode: episode_steps[0].equilibrium.total_surplus
            for episode, episode_steps in self.steps.items()
            if episode_steps
        }

    @computed_field
    @property
    def efficiency(self) -> Dict[int, float]:
        return {
            episode: self.surplus[episode] / self.theoretical_surplus[episode]
            if self.theoretical_surplus[episode] != 0 else 0
            for episode in self.surplus.keys()
        }

    @computed_field
    @property
    def quantity(self) -> Dict[int, int]:
        return {
            episode: sum(sum(trade.quantity for trade in step.trades) for step in episode_steps)
            for episode, episode_steps in self.steps.items()
        }

    @computed_field
    @property
    def average_price(self) -> Dict[int, float]:
        return {
            episode: mean(trade.price for step in episode_steps for trade in step.trades)
            if any(step.trades for step in episode_steps) else 0
            for episode, episode_steps in self.steps.items()
        }

    @computed_field
    @property
    def min_price(self) -> Dict[int, float]:
        return {
            episode: min((trade.price for step in episode_steps for trade in step.trades), default=0)
            for episode, episode_steps in self.steps.items()
        }

    @computed_field
    @property
    def max_price(self) -> Dict[int, float]:
        return {
            episode: max((trade.price for step in episode_steps for trade in step.trades), default=0)
            for episode, episode_steps in self.steps.items()
        }

    @computed_field
    @property
    def std_price(self) -> Dict[int, float]:
        return {
            episode: stdev((trade.price for step in episode_steps for trade in step.trades))
            if len([trade for step in episode_steps for trade in step.trades]) > 1 else 0
            for episode, episode_steps in self.steps.items()
        }

    @computed_field
    @property
    def equilibrium_price(self) -> Dict[int, float]:
        return {
            episode: episode_steps[0].equilibrium.price if episode_steps else 0
            for episode, episode_steps in self.steps.items()
        }

    @computed_field
    @property
    def equilibrium_quantity(self) -> Dict[int, int]:
        return {
            episode: episode_steps[0].equilibrium.quantity if episode_steps else 0
            for episode, episode_steps in self.steps.items()
        }

    @computed_field
    @property
    def summary(self) -> Dict[int, EpisodeSummary]:
        summaries = {}
        for episode in self.steps.keys():
            summaries[episode] = EpisodeSummary(
                episode=episode,
                surplus=self.surplus[episode],
                theoretical_surplus=self.theoretical_surplus[episode],
                efficiency=self.efficiency[episode],
                quantity=self.quantity[episode],
                average_price=self.average_price[episode],
                min_price=self.min_price[episode],
                max_price=self.max_price[episode],
                std_price=self.std_price[episode],
                equilibrium_price=self.equilibrium_price[episode],
                equilibrium_quantity=self.equilibrium_quantity[episode]
            )
        return summaries

    def get_market_step_count(self, market_id: str, episode: int) -> int:
        if episode not in self.steps:
            return 0
        return sum(1 for step in self.steps[episode] if step.market_id == market_id)

    def get_market_ids(self) -> Set[str]:
        return {step.market_id for episode_steps in self.steps.values() for step in episode_steps}

    def get_participating_agents(self, market_id: str) -> Set[str]:
        return {agent_id 
                for episode_steps in self.steps.values()
                for step in episode_steps 
                if step.market_id == market_id 
                for agent_id in step.participating_agent_ids}

    def get_market_history(self, market_id: str, episode: Optional[int] = None) -> List[MarketStep]:
        if episode is not None:
            return [step for step in self.steps.get(episode, []) if step.market_id == market_id]
        return [step for episode_steps in self.steps.values() for step in episode_steps if step.market_id == market_id]

    

class MarketOrchestrator:
    def __init__(self, 
                llm_agents: List[SimpleAgent],
                goods: List[str],
                ai_utils: ParallelAIUtilities = ParallelAIUtilities(),
                max_rounds: int = 10,
                scenario: Optional[Scenario] = None,clones_config:Optional[LLMConfig]=None):
        self.llm_agents = llm_agents
        self.goods = goods
        self.max_rounds = max_rounds
        self.markets = self.create_markets(self.goods)
        self.ai_utils = ai_utils
        self.scenario = scenario
        self.agents_dict, self.llm_agents_dict, self.zi_agents_dict = self.create_agents_dicts(clones_config)
        self.agents = list(self.agents_dict.values())
        self.failed_actions: List[LLMOutput] = []
        if self.scenario:
            name = self.scenario.name+"_state"
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
    
    def create_agents_dicts(self, clones_config:Optional[LLMConfig]=None) -> Tuple[Dict[str, Union[SimpleAgent, EconomicAgent]],
                                            Dict[str, SimpleAgent], 
                                            Dict[str, EconomicAgent]]:
        zi_dict = {}
        llm_cloned_dict = {}
        if self.scenario and self.scenario.generate_zi_agents:
            zi_agents = self.scenario.agents
            zi_dict = {agent.id: agent for agent in zi_agents}
        if self.scenario and clones_config is not None:
            llm_cloned_agents = create_simple_agents_from_zi(self.scenario.agents, clones_config)
            self.llm_agents.extend(llm_cloned_agents)
            llm_cloned_dict = {agent.id: agent for agent in llm_cloned_agents}
        llm_dict = {**{agent.id: agent for agent in self.llm_agents}, **llm_cloned_dict}
        return {**zi_dict, **llm_dict}, llm_dict, zi_dict
    
    def clone_zi_dict(self, clones_config:LLMConfig) -> Dict[str, SimpleAgent]:
        llm_cloned_agents = create_simple_agents_from_zi(list(self.zi_agents_dict.values()), clones_config)
        self.llm_agents.extend(llm_cloned_agents)
        return {agent.id: agent for agent in llm_cloned_agents}
    
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
                # print(f"bid price: {bid.price}, current_value: {current_value}")
                if current_value is not None and bid.price < current_value:
                    # print(f"adding bid to pending orders")
                    market_action = bid
                else:
                    print(f"not adding bid to pending orders")
            elif "Ask" in output.json_object.name:
                ask = Ask.model_validate(output.json_object.object)
                current_cost = agent.get_current_cost(good_name)
                # print(f"ask price: {ask.price}, current_cost: {current_cost}")
                if current_cost is not None and ask.price > current_cost:
                    # print(f"adding ask {ask} from agent {agent.id} with current_cost {current_cost} to pending orders")
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
            logger.info(f"Round {round + 1}")
            # print(f"Round {round + 1} with agent names: {[agent.id for agent in relevant_agents]}")
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

    async def run_scenario(self,report:bool=True):
        if self.scenario:
            for episode in range(self.scenario.num_episodes):
                for good in self.scenario.goods:
                    print(f"Running episode {episode} for good {good}")
                    await self.run_auction_episode(self.markets[good].mechanism.max_rounds, good,report=report)
                    print(f"Episode {episode} for good {good} completed")
        else:
            for good in self.goods:
                await self.run_auction_episode(self.markets[good].mechanism.max_rounds, good,report=report)


async def run_zi_scenario(buyer_params: ZiParams, seller_params: ZiParams,max_rounds: int = 1,num_buyers: int = 25,num_sellers:int=25):
    load_dotenv()
        # Create a good
    apple = Good(name="apple", quantity=0)
    # Create DoubleAuction mechanism
    factories = [
        ZiFactory(
            id=f"factory_episode_{0}",
            goods=["apple"],
            num_buyers=num_buyers,  # Increase buyers by 1 each episode
            num_sellers=num_sellers,     # Keep sellers constant
            buyer_params=buyer_params,
            seller_params=seller_params
        )
    ]
    scenario = Scenario(
        name="Static Apple Market",
        goods=["apple"],
        factories=factories
    )
    starting_scenario = scenario.model_copy(deep=True)
    orchestrator = MarketOrchestrator(llm_agents=[], goods=[apple.name], max_rounds=max_rounds,scenario=scenario)

    # Run the market simulation
    await orchestrator.run_scenario(report=False)
    #plot the market results
    return starting_scenario,orchestrator.state


async def run_llm_matched_scenario(zi_scenario:Scenario, clones_config:Optional[LLMConfig]=None, ai_utils:Optional[ParallelAIUtilities]=None, max_rounds:int=10):
    load_dotenv()
    scenario = zi_scenario.model_copy(deep=True,update={"generate_zi_agents": False})
    # Set up ParallelAIUtilities
    
    parallel_ai = ParallelAIUtilities(
    ) if ai_utils is None else ai_utils

    # Create a good


    orchestrator = MarketOrchestrator(llm_agents=[],
                                       goods=scenario.goods,
                                       ai_utils=parallel_ai,
                                       max_rounds=max_rounds,
                                       scenario=scenario,
                                       clones_config=LLMConfig(model="gpt-4o-mini",
                                                               temperature=0.0,
                                                               client="openai",
                                                               response_format="tool",
                                                               max_tokens=250) if clones_config is None else clones_config) 
    # return orchestrator
    await orchestrator.run_scenario()
    return orchestrator