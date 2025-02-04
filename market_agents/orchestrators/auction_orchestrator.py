# auction_orchestrator.py

import asyncio
from datetime import datetime
import json
import logging
from typing import List, Dict, Any

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import EnvironmentStep, MultiAgentEnvironment
from market_agents.environments.mechanisms.auction import (
    AuctionAction,
    AuctionActionSpace,
    AuctionGlobalObservation,
    AuctionObservationSpace,
    DoubleAuction,
    GlobalAuctionAction
)
from market_agents.economics.econ_models import (
    Ask,
    Bid,
    Trade
)
from market_agents.orchestrators.config import AuctionConfig, OrchestratorConfig
from market_agents.orchestrators.logger_utils import (
    log_section,
    log_environment_setup,
    log_persona,
    log_running,
    log_perception,
    log_action,
    log_round
)
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter
from market_agents.orchestrators.parallel_cognitive_steps import AgentCognitiveProcessor

# Define AuctionTracker for tracking auction-specific data
class AuctionTracker:
    def __init__(self):
        self.all_trades: List[Trade] = []
        self.per_round_surplus: List[float] = []
        self.per_round_quantities: List[int] = []

    def add_trade(self, trade: Trade, buyer_surplus: float, seller_surplus: float, round_num: int):
        trade_data = {
            'buyer_id': str(trade.buyer_id),
            'seller_id': str(trade.seller_id),
            'quantity': trade.quantity,
            'price': trade.price,
            'buyer_surplus': round(buyer_surplus, 2),
            'seller_surplus': round(seller_surplus, 2),
            'total_surplus': round(buyer_surplus + seller_surplus, 2),
            'round': round_num
        }
        self.all_trades.append(trade_data)
        logging.debug("Added trade to tracker: {trade_data}")

    def add_round_data(self, surplus: float, quantity: int):
        self.per_round_surplus.append(surplus)
        self.per_round_quantities.append(quantity)

    def get_summary(self):
        return {
            "total_trades": len(self.all_trades),
            "total_surplus": sum(self.per_round_surplus),
            "total_quantity": sum(self.per_round_quantities)
        }

    def get_trades_data(self) -> List[Dict]:
        return [
            {
                'buyer_id': trade['buyer_id'],
                'seller_id': trade['seller_id'],
                'quantity': trade['quantity'],
                'price': trade['price'],
                'buyer_surplus': trade['buyer_surplus'],
                'seller_surplus': trade['seller_surplus'],
                'total_surplus': trade['total_surplus'],
                'round': trade['round']
            }
            for trade in self.all_trades
        ]

# Implement the AuctionOrchestrator class
class AuctionOrchestrator(BaseEnvironmentOrchestrator):
    def __init__(
        self,
        config: AuctionConfig,
        orchestrator_config: OrchestratorConfig, 
        agents: List[MarketAgent],
        ai_utils,
        data_inserter: OrchestrationDataInserter,
        logger=None
    ):
        super().__init__(
            config=config,
            agents=agents,
            ai_utils=ai_utils,
            data_inserter=data_inserter,
            logger=logger
        )
        self.orchestrator_config = orchestrator_config
        self.environment_name = 'auction'
        self.environment = None
        self.tracker = AuctionTracker()
        self.agent_surpluses: Dict[str, float] = {}
        self.logger = logger or logging.getlogger(__name__)
        self.cognitive_processor = AgentCognitiveProcessor(ai_utils, data_inserter, self.logger, self.orchestrator_config.tool_mode)

        
    async def setup_environment(self):
        """Sets up the auction environment and assigns it to agents"""
        log_section(self.logger, "CONFIGURING AUCTION ENVIRONMENT")
        
        # Create the auction mechanism
        double_auction = DoubleAuction(
            max_rounds=self.config.max_rounds,
            good_name=self.orchestrator_config.agent_config.good_name
        )
        
        # Set up the multi-agent environment
        self.environment = MultiAgentEnvironment(
            name=self.config.name,
            address=self.config.address,
            max_steps=self.config.max_rounds,
            action_space=AuctionActionSpace(),
            observation_space=AuctionObservationSpace(),
            mechanism=double_auction
        )
        
        # Assign the environment to agents
        for agent in self.agents:
            if not hasattr(agent, 'environments') or agent.environments is None:
                agent.environments = {}
            agent.environments[self.environment_name] = self.environment
            
        log_environment_setup(self.logger, self.environment_name)
        self.logger.info("Auction environment setup complete.")
        
        # Verify environments are properly set
        for agent in self.agents:
            if self.environment_name not in agent.environments:
                self.logger.error(f"Agent {agent.index} missing auction environment!")
            else:
                self.logger.info(f"Agent {agent.index} environments: {list(agent.environments.keys())}")

        return self.environment

    async def run_environment(self, round_num: int):
        log_running(self.logger, self.environment_name)
        env = self.environment

        # Reset agents' pending orders at the beginning of the round
        for agent in self.agents:
            agent.economic_agent.reset_all_pending_orders()

        # Set system messages for agents
        self.set_agent_system_messages(round_num, env.mechanism.good_name)

        log_section(self.logger, "AGENT PERCEPTIONS")
        # Run agents' perception in parallel using imported cognitive method
        perceptions = await self.cognitive_processor.run_parallel_perceive(
            self.agents, 
            self.environment_name
        )

        # Map perceptions to agents
        perceptions_map = {perception.source_id: perception for perception in perceptions}

        for agent in self.agents:
            perception = perceptions_map.get(agent.id)
            if perception:
                log_persona(self.logger, agent.index, agent.persona)
                log_perception(self.logger, agent.index, perception.json_object.object if perception and perception.json_object else None)
                agent.last_perception = perception.json_object.object if perception and perception.json_object else ""
            else:
                self.logger.warning(f"No perception found for agent {agent.index}")
                agent.last_perception = None

        # Extract perception contents for action generation
        perception_contents = [agent.last_perception or "" for agent in self.agents]

        log_section(self.logger, "AGENT ACTIONS")
        # Run agents' action generation in parallel using imported cognitive method
        actions = await self.cognitive_processor.run_parallel_action(
            self.agents,
            self.environment_name
        )

        actions_map = {action.source_id: action for action in actions}

        # Collect actions from agents
        agent_actions = {}
        for agent in self.agents:
            action = actions_map.get(agent.id)
            if action:
                try:
                    action_content = action.json_object.object if action.json_object else json.loads(action.str_content or '{}')
                    agent.last_action = action_content
                    if 'price' in action_content and 'quantity' in action_content:
                        action_content['quantity'] = 1  # Ensure quantity is 1

                        if agent.role == "buyer":
                            auction_action = Bid(**action_content)
                        elif agent.role == "seller":
                            auction_action = Ask(**action_content)
                        else:
                            raise ValueError(f"Invalid agent role: {agent.role}")

                        agent_actions[agent.id] = AuctionAction(agent_id=agent.id, action=auction_action)
                        # Update agent's pending orders
                        good_name = env.mechanism.good_name
                        agent.economic_agent.pending_orders.setdefault(good_name, []).append(auction_action)

                        action_type = "Bid" if isinstance(auction_action, Bid) else "Ask"
                        log_action(self.logger, agent.index, f"{action_type}: {auction_action}")
                    else:
                        raise ValueError(f"Invalid action content: {action_content}")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self.logger.error(f"Error creating AuctionAction for agent {agent.index}: {str(e)}")
            else:
                self.logger.warning(f"No action found for agent {agent.index}")

        # Create global action and step the environment
        global_action = GlobalAuctionAction(actions=agent_actions)
        try:
            env_state = env.step(global_action)
        except Exception as e:
            self.logger.error(f"Error in environment {self.environment_name}: {str(e)}")
            raise e

        self.logger.info(f"Completed {self.environment_name} step")

        # Process the environment state
        if isinstance(env_state.global_observation, AuctionGlobalObservation):
            self.process_environment_state(env_state)

        # Store the last environment state
        self.last_env_state = env_state

        # Run reflection step
        log_section(self.logger, "AGENT REFLECTIONS")
        await self.cognitive_processor.run_parallel_reflect(
            self.agents,
            self.environment_name
        )

    def set_agent_system_messages(self, round_num: int, good_name: str):
        # Set system messages for agents based on their role and round number
        for agent in self.agents:
            current_cash = agent.economic_agent.endowment.current_basket.cash
            current_goods = agent.economic_agent.endowment.current_basket.get_good_quantity(good_name)
            if round_num == 1:
                if agent.role == "buyer":
                    current_value = agent.economic_agent.get_current_value(good_name)
                    if current_value is not None:
                        suggested_price = current_value * 0.99
                        agent.system = (
                            f"This is the first round of the market so there are no bids or asks yet. "
                            f"You have {current_cash:.2f} cash and {current_goods} units of {good_name}. "
                            f"You can make a profit by buying at {suggested_price:.2f} or lower."
                        )
                    else:
                        agent.system = f"You have reached your maximum quantity for {good_name}."
                elif agent.role == "seller":
                    if current_goods <= 0:
                        agent.system = f"You have no {good_name} to sell."
                    else:
                        current_cost = agent.economic_agent.get_current_cost(good_name)
                        if current_cost is not None:
                            suggested_price = current_cost * 1.01
                            agent.system = (
                                f"This is the first round of the market so there are no bids or asks yet. "
                                f"You have {current_cash:.2f} cash and {current_goods} units of {good_name}. "
                                f"You can make a profit by selling at {suggested_price:.2f} or higher."
                            )
                        else:
                            agent.system = f"You have no more {good_name} to sell."
            else:
                if agent.role == "buyer":
                    marginal_value = agent.economic_agent.get_current_value(good_name)
                    if marginal_value is not None:
                        suggested_price = marginal_value * 0.99
                        agent.system = (
                            f"Your current cash: {current_cash:.2f}, goods: {current_goods}. "
                            f"Your marginal value for the next unit of {good_name} is {marginal_value:.2f}. "
                            f"You can make a profit by buying at {suggested_price:.2f} or lower."
                        )
                    else:
                        agent.system = f"You have reached your maximum quantity for {good_name}."
                elif agent.role == "seller":
                    if current_goods <= 0:
                        agent.system = f"You have no {good_name} to sell."
                    else:
                        marginal_cost = agent.economic_agent.get_current_cost(good_name)
                        if marginal_cost is not None:
                            suggested_price = marginal_cost * 1.01
                            agent.system = (
                                f"Your current cash: {current_cash:.2f}, goods: {current_goods}. "
                                f"Your marginal cost for selling the next unit of {good_name} is {marginal_cost:.2f}. "
                                f"You can make a profit by selling at {suggested_price:.2f} or higher."
                            )
                        else:
                            agent.system = f"You have no more {good_name} to sell."

    def process_environment_state(self, env_state: EnvironmentStep):
        global_observation = env_state.global_observation
        if not isinstance(global_observation, AuctionGlobalObservation):
            self.logger.error(f"Unexpected global observation type: {type(global_observation)}")
            return

        round_surplus = 0
        round_quantity = 0
        agent_surpluses = {}
        self.logger.info(f"Processing {len(global_observation.all_trades)} trades")
        
        log_section(self.logger, "TRADES")
        
        # Process each trade from the global observation
        for trade in global_observation.all_trades:
            try:
                buyer = next(agent for agent in self.agents if agent.id == trade.buyer_id)
                seller = next(agent for agent in self.agents if agent.id == trade.seller_id)
                
                # Process the trade for both agents
                buyer.economic_agent.process_trade(trade)
                seller.economic_agent.process_trade(trade)
                
                # Calculate surpluses
                buyer_surplus = round(buyer.economic_agent.calculate_individual_surplus(), 2)
                seller_surplus = round(seller.economic_agent.calculate_individual_surplus(), 2)
                
                self.logger.info(f"Buyer surplus: {buyer_surplus:.2f}, Seller surplus: {seller_surplus:.2f}")
                
                # Update agent surpluses
                agent_surpluses[buyer.id] = round(agent_surpluses.get(buyer.id, 0) + buyer_surplus, 2)
                agent_surpluses[seller.id] = round(agent_surpluses.get(seller.id, 0) + seller_surplus, 2)
                
                # Add trade to tracker with current round number from mechanism
                self.tracker.add_trade(trade, buyer_surplus, seller_surplus, global_observation.market_summary.trades_count)
                
                trade_surplus = round(buyer_surplus + seller_surplus, 2)
                round_surplus += trade_surplus
                round_quantity += trade.quantity
                
                self.logger.info(f"Executed trade: {trade}")
                self.logger.info(f"Trade surplus: {trade_surplus:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error processing trade: {str(e)}")
                self.logger.exception("Exception details:")
        
        self.tracker.add_round_data(round_surplus, round_quantity)
        self.logger.info(f"Round summary - Surplus: {round_surplus}, Quantity: {round_quantity}")
        
        # Update agent states
        for agent_id, agent_observation in global_observation.observations.items():
            try:
                agent = next(agent for agent in self.agents if agent.id == agent_id)
                # Pre-serialize the observation
                if hasattr(agent_observation, 'serialize_json'):
                    serialized_observation = json.loads(agent_observation.serialize_json())
                elif hasattr(agent_observation, 'model_dump'):
                    serialized_observation = agent_observation.model_dump()
                else:
                    serialized_observation = str(agent_observation)
                    
                agent.last_observation = serialized_observation
                agent.last_step = env_state
            except Exception as e:
                self.logger.error(f"Error updating agent {agent_id} state: {str(e)}")
                self.logger.exception("Exception details:")

        # Store agent_surpluses for reflection
        env_state.info['agent_rewards'] = agent_surpluses
        self.agent_surpluses = agent_surpluses

        # Store the last environment state
        self.last_env_state = env_state

    def get_round_summary(self, round_num: int) -> dict:
        # Return a summary of the round
        summary = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'is_buyer': agent.role == "buyer",
                'cash': agent.economic_agent.endowment.current_basket.cash,
                'goods': agent.economic_agent.endowment.current_basket.goods_dict,
                'last_action': agent.last_action,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'environment_state': self.environment.get_global_state(),
            'tracker_summary': self.tracker.get_summary()
        }
        return summary

    async def process_round_results(self, round_num: int):
        try:
            agents_data = [
                {
                    'id': str(agent.id),
                    'role': agent.role,
                    'is_llm': agent.use_llm,
                    'max_iter': self.orchestrator_config.max_rounds,
                    'llm_config': agent.llm_config if isinstance(agent.llm_config, dict) else agent.llm_config.dict()
                }
                for agent in self.agents
            ]
            agent_id_map = self.data_inserter.insert_agents(agents_data)
            
            trades_data = self.tracker.get_trades_data()
            if trades_data:
                self.data_inserter.insert_trades(trades_data, agent_id_map)
            
            self.data_inserter.insert_round_data(
                round_num,
                self.agents,
                {self.environment_name: self.environment},
                self.orchestrator_config,
                {self.environment_name: self.tracker}
            )
            
            self.logger.info(f"Data for round {round_num} inserted successfully.")
        except Exception as e:
            self.logger.error(f"Error inserting data for round {round_num}: {str(e)}")
            self.logger.exception("Exception details:")

    async def run(self):
        self.setup_environment()
        for round_num in range(1, self.config.max_rounds + 1):
            log_round(self.logger, round_num)
            await self.run_environment(round_num)
            await self.process_round_results(round_num)
        # Print simulation summary after all rounds
        self.print_summary()

    async def print_summary(self):
        log_section(self.logger, "AUCTION SIMULATION SUMMARY")

        total_buyer_surplus = sum(
            agent.economic_agent.calculate_individual_surplus() for agent in self.agents if agent.role == "buyer"
        )
        total_seller_surplus = sum(
            agent.economic_agent.calculate_individual_surplus() for agent in self.agents if agent.role == "seller"
        )
        total_empirical_surplus = total_buyer_surplus + total_seller_surplus

        print(f"Total Empirical Buyer Surplus: {total_buyer_surplus:.2f}")
        print(f"Total Empirical Seller Surplus: {total_seller_surplus:.2f}")
        print(f"Total Empirical Surplus: {total_empirical_surplus:.2f}")

        global_state = self.environment.get_global_state()
        equilibria = global_state.get('equilibria', {})

        if equilibria:
            theoretical_total_surplus = sum(data['total_surplus'] for data in equilibria.values())
            print(f"\nTheoretical Total Surplus: {theoretical_total_surplus:.2f}")

            efficiency = (total_empirical_surplus / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0
            print(f"\nEmpirical Efficiency: {efficiency:.2f}%")
        else:
            print("\nTheoretical equilibrium data not available.")

        summary = self.tracker.get_summary()
        print(f"\nAuction Environment:")
        print(f"Total number of trades: {summary['total_trades']}")
        print(f"Total surplus: {summary['total_surplus']:.2f}")
        print(f"Total quantity traded: {summary['total_quantity']}")

        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index} ({agent.role}):")
            print(f"  Cash: {agent.economic_agent.endowment.current_basket.cash:.2f}")
            print(f"  Goods: {agent.economic_agent.endowment.current_basket.goods_dict}")
            surplus = agent.economic_agent.calculate_individual_surplus()
            print(f"  Individual Surplus: {surplus:.2f}")
            
            # Get the most recent reflection from short-term memory
            try:
                recent_reflections = await agent.short_term_memory.retrieve_recent_memories(cognitive_step='reflection', limit=1)
                if recent_reflections:
                    print(f"  Last Reflection: {recent_reflections[0].content}")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve reflection for agent {agent.index}: {e}")
            print()
