import logging
from typing import List, Dict, Any, Type
from environments.auction.auction import DoubleAuction
from pydantic import BaseModel, Field
from colorama import Fore, Style
import threading
import os
import yaml
from pathlib import Path
import psycopg2
from psycopg2.extras import Json, UUID_adapter
import uuid
from pgvector.psycopg2 import register_vector
import numpy as np

from market_agent.market_agents import MarketAgent
from environments.environment import Environment
from environments.auction.auction_environment import AuctionEnvironment
from base_agent.aiutilities import LLMConfig
from base_agent.agent import Agent
from protocols.protocol import Protocol
from protocols.acl_message import ACLMessage
from simulation_app import create_dashboard
from logger_utils import *
from personas.persona import generate_persona, save_persona_to_file, Persona

logger = setup_logger(__name__)

class AgentConfig(BaseModel):
    num_units: int
    base_value: float
    use_llm: bool
    initial_cash: float
    initial_goods: int
    noise_factor: float = Field(default=0.1)
    max_relative_spread: float = Field(default=0.2)

class AuctionConfig(BaseModel):
    name: str
    address: str
    auction_type: str
    max_steps: int

class OrchestratorConfig(BaseModel):
    num_agents: int
    max_rounds: int
    agent_config: AgentConfig
    llm_config: LLMConfig
    environment_configs: Dict[str, AuctionConfig]
    protocol: Type[Protocol]
    database_config: Dict[str, Any]

class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.environments: Dict[str, Environment] = {}
        self.dashboard = None
        self.database = None
        self.simulation_order = ['auction']
        self.simulation_data: List[Dict[str, Any]] = []
        self.latest_data = None
        self.conn = None

    def load_or_generate_personas(self) -> List[Persona]:
        personas_dir = Path("./personas/generated_personas")
        existing_personas = []

        # Check if the directory exists and load existing personas
        if personas_dir.exists():
            for filename in personas_dir.glob("*.yaml"):
                with filename.open('r') as file:
                    persona_data = yaml.safe_load(file)
                    existing_personas.append(Persona(**persona_data))

        # Generate additional personas if needed
        while len(existing_personas) < self.config.num_agents:
            new_persona = generate_persona()
            existing_personas.append(new_persona)
            save_persona_to_file(new_persona, str(personas_dir))

        return existing_personas[:self.config.num_agents]

    def generate_agents(self):
        log_section(logger, "INITIALIZING MARKET AGENTS")
        personas = self.load_or_generate_personas()
        for i, persona in enumerate(personas):
            agent = MarketAgent.create(
                agent_id=i,
                is_buyer=persona.role.lower() == "buyer",
                **self.config.agent_config.dict(),
                llm_config=self.config.llm_config,
                protocol=self.config.protocol,
                environments=self.environments,
                persona=persona
            )
            self.agents.append(agent)
            log_agent_init(logger, i, persona.role.lower() == "buyer")
        
        logger.info(f"Generated {len(self.agents)} agents")

    def setup_environments(self):
        log_section(logger, "CONFIGURING MARKET ENVIRONMENTS")
        for env_name, env_config in self.config.environment_configs.items():
            if env_name == 'auction':
                env = AuctionEnvironment(
                    agents=self.agents,
                    max_steps=env_config.max_steps,
                    protocol=self.config.protocol,
                    name=env_config.name,
                    address=env_config.address,
                    auction_type=env_config.auction_type
                )
                self.environments[env_name] = env
                log_environment_setup(logger, env_name)
            else:
                logger.warning(f"Unknown environment type: {env_name}")

        logger.info(f"Set up {len(self.environments)} environments")

        for agent in self.agents:
            agent.environments = self.environments

    def setup_dashboard(self):
        log_section(logger, "INITIALIZING DASHBOARD")
        self.dashboard = create_dashboard(self.data_source)
        logger.info("Dashboard setup complete")

    def setup_database(self):
        try:
            self.conn = psycopg2.connect(
                dbname=os.environ.get('DB_NAME', 'market_simulation'),
                user=os.environ.get('DB_USER', 'db_user'),
                password=os.environ.get('DB_PASSWORD', 'db_pwd@123'),
                host=os.environ.get('DB_HOST', 'localhost'),
                port=os.environ.get('DB_PORT', '5433')
            )
            register_vector(self.conn)
            logger.info("Database connection established successfully with pgvector support.")
        except (Exception, psycopg2.Error) as error:
            logger.error(f"Error while connecting to PostgreSQL: {error}")

    def save_memory_embedding(self, agent_id, embedding, memory_data):
        if self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute("""
                INSERT INTO memory_embeddings (agent_id, embedding, memory_data)
                VALUES (%s, %s, %s)
                """, (UUID_adapter(agent_id), embedding, Json(memory_data)))
                self.conn.commit()
                logger.info(f"Memory embedding saved for agent {agent_id}")
            except (Exception, psycopg2.Error) as error:
                logger.error(f"Error while saving memory embedding: {error}")
                self.conn.rollback()
            finally:
                cursor.close()

    def retrieve_similar_memories(self, agent_id, query_embedding, limit=5):
        if self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute("""
                SELECT memory_data, embedding <-> %s AS distance
                FROM memory_embeddings
                WHERE agent_id = %s
                ORDER BY distance
                LIMIT %s
                """, (query_embedding, UUID_adapter(agent_id), limit))
                results = cursor.fetchall()
                return [{'memory_data': result[0], 'distance': result[1]} for result in results]
            except (Exception, psycopg2.Error) as error:
                logger.error(f"Error while retrieving similar memories: {error}")
            finally:
                cursor.close()
        return []

    def run_simulation(self):
        logger.info("Starting simulation")
        
        for round_num in range(1, self.config.max_rounds + 1):
            log_round(logger, round_num)
            
            for env_name in self.simulation_order:
                env_state = self.run_environment(env_name)
                self.update_simulation_state(env_name, env_state)
            
            for agent in self.agents:
                reflection = agent.reflect(env_name)
                if reflection:
                    log_reflection(logger, int(agent.id), f"{Fore.MAGENTA}{reflection}{Style.RESET_ALL}")
                    
                    # Generate an embedding for the reflection (you'll need to implement this)
                    embedding = self.generate_embedding(reflection)
                    
                    # Save the memory embedding
                    self.save_memory_embedding(agent.id, embedding, {'reflection': reflection, 'round': round_num})
                else:
                    logger.warning(f"Agent {agent.id} returned empty reflection")
            
            self.save_round_data(round_num)
            self.update_dashboard()
        
        logger.info("Simulation completed")

    def run_environment(self, env_name: str):
        env = self.environments[env_name]
        
        log_running(logger, env_name)
        agent_actions = {}
        for agent in self.agents:
            log_section(logger, f"Current Agent:\nAgent {agent.id} with persona:\n{agent.persona}")
            perception = agent.perceive(env_name)
            log_perception(logger, int(agent.id), f"{Fore.CYAN}{perception}{Style.RESET_ALL}")

            action = agent.generate_action(env_name, perception)
            agent_actions[agent.id] = action
            action_type = action['content'].get('action', 'Unknown')
            if action_type == 'bid':
                color = Fore.BLUE
            elif action_type == 'ask':
                color = Fore.GREEN
            else:
                color = Fore.YELLOW
            log_action(logger, int(agent.id), f"{color}{action_type}: {action['content']}{Style.RESET_ALL}")
        
        env_state = env.update(agent_actions)
        logger.info(f"Completed {env_name} step")
        return env_state

    def update_simulation_state(self, env_name: str, env_state: Dict[str, Any]):
        # Update agent states based on the environment state
        for agent in self.agents:
            agent_observation = env_state['observations'].get(agent.id)
            if agent_observation:
                # Convert the observation to a dictionary if it's not already
                if not isinstance(agent_observation, dict):
                    agent_observation = agent_observation.dict()
                # Call the EconomicAgent's update_state method directly
                agent.update_state(agent_observation)

        # Create a new state dictionary for this round if it doesn't exist
        if not self.simulation_data or 'state' not in self.simulation_data[-1]:
            self.simulation_data.append({'state': {}})

        # Update the simulation state
        self.simulation_data[-1]['state'][env_name] = env_state.get('market_info', {})
        self.simulation_data[-1]['state']['trade_info'] = env_state.get('trade_info', {})

    def save_round_data(self, round_num):
        if self.conn:
            cursor = self.conn.cursor()
            try:
                logger.info(f"Attempting to save round {round_num} data to the database.")
                
                env = self.environments['auction']
                auction = env.auction

                # Insert into auctions table
                cursor.execute("""
                INSERT INTO auctions (max_rounds, current_round, total_surplus_extracted, average_prices, order_book, trade_history)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """, (auction.max_rounds, auction.current_round, auction.total_surplus_extracted, 
                      Json(auction.average_prices), Json(auction.order_book.dict()), Json([trade.dict() for trade in auction.trade_history])))
                auction_id = cursor.fetchone()[0]
                logger.info(f"Inserted auction data with id {auction_id}")

                # Insert or update agent data
                for agent in self.agents:
                    agent_uuid = getattr(agent, 'uuid', uuid.uuid4())
                    
                    cursor.execute("""
                    INSERT INTO agents (id, role, is_llm, max_iter, llm_config)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                    role = EXCLUDED.role,
                    is_llm = EXCLUDED.is_llm,
                    max_iter = EXCLUDED.max_iter,
                    llm_config = EXCLUDED.llm_config
                    """, (UUID_adapter(agent_uuid), agent.role, agent.use_llm, self.config.max_rounds, Json(agent.llm_config.dict())))
                    logger.info(f"Inserted/updated agent data for agent {agent.id}")

                    # Insert preference schedules
                    cursor.execute("""
                    INSERT INTO preference_schedules (agent_id, is_buyer, values, initial_endowment)
                    VALUES (%s, %s, %s, %s)
                    """, (UUID_adapter(agent_uuid), agent.is_buyer, Json(agent.preference_schedule.dict()), agent.endowment.cash))
                    logger.info(f"Inserted preference schedule for agent {agent.id}")

                    # Insert allocations
                    cursor.execute("""
                    INSERT INTO allocations (agent_id, goods, cash, locked_goods, locked_cash, initial_goods, initial_cash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (UUID_adapter(agent_uuid), agent.endowment.goods, agent.endowment.cash, 0, 0, self.config.agent_config.initial_goods, self.config.agent_config.initial_cash))
                    logger.info(f"Inserted allocation data for agent {agent.id}")

                # Insert orders
                for bid in auction.order_book.bids:
                    cursor.execute("""
                    INSERT INTO orders (agent_id, is_buy, quantity, price, base_value, base_cost)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (UUID_adapter(bid.agent_id), True, bid.quantity, bid.price, bid.base_value, None))
                for ask in auction.order_book.asks:
                    cursor.execute("""
                    INSERT INTO orders (agent_id, is_buy, quantity, price, base_value, base_cost)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (UUID_adapter(ask.agent_id), False, ask.quantity, ask.price, None, ask.base_cost))
                logger.info("Inserted order data")

                # Insert trades
                for trade in auction.trade_history:
                    cursor.execute("""
                    INSERT INTO trades (buyer_id, seller_id, quantity, price, buyer_value, seller_cost, round)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (UUID_adapter(trade.bid.agent_id), UUID_adapter(trade.ask.agent_id), trade.quantity, trade.price, trade.buyer_value, trade.seller_cost, trade.round))
                logger.info("Inserted trade data")

                # Insert interactions
                for agent in self.agents:
                    if agent.last_action:
                        cursor.execute("""
                        INSERT INTO interactions (agent_id, round, task, response)
                        VALUES (%s, %s, %s, %s)
                        """, (UUID_adapter(agent_uuid), round_num, 'action', Json(agent.last_action)))
                logger.info("Inserted interaction data")

                self.conn.commit()
                logger.info(f"Round {round_num} data saved to the database.")
                
                # Print loaded data
                self.print_loaded_data()
            except (Exception, psycopg2.Error) as error:
                logger.error(f"Error while saving round data to PostgreSQL: {error}")
                self.conn.rollback()
            finally:
                cursor.close()
        logger.info(f"Data for round {round_num} saved successfully")

    def print_loaded_data(self):
        if self.conn:
            cursor = self.conn.cursor()
            try:
                tables = ['auctions', 'agents', 'preference_schedules', 'allocations', 'orders', 'trades', 'interactions']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"Table {table}: {count} rows")
                    
                    if count > 0:
                        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                        rows = cursor.fetchall()
                        print(f"Sample data from {table}:")
                        for row in rows:
                            print(row)
                    print()
            except (Exception, psycopg2.Error) as error:
                logger.error(f"Error while fetching data from PostgreSQL: {error}")
            finally:
                cursor.close()

    def data_source(self):
        env = self.environments['auction']
        return env.get_global_state()

    def update_dashboard(self):
        if self.dashboard:
            logger.info("Updating dashboard data...")
            self.latest_data = self.data_source()

    def run_dashboard(self):
        if self.dashboard:
            log_section(logger, "LAUNCHING DASHBOARD UI")
            self.dashboard.run_server(debug=True, use_reloader=False)

    def close_db_connection(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")

    def generate_embedding(self, text):
        # This is a placeholder function. You should replace this with your actual embedding generation logic.
        # For example, you might use a pre-trained model from Hugging Face or OpenAI's embedding API.
        return np.random.rand(1536).tolist()  # pgvector typically uses 1536-dimensional vectors

    def start(self):
        self.setup_database()
        log_section(logger, "MARKET SIMULATION INITIALIZING")
        self.generate_agents()
        self.setup_environments()

        # Start the dashboard in a separate thread
        if self.dashboard:
            dashboard_thread = threading.Thread(target=self.run_dashboard)
            dashboard_thread.start()

        # Run the simulation
        self.run_simulation()

        # Wait for the dashboard thread to finish if it's running
        if self.dashboard:
            dashboard_thread.join()

        # Close the database connection
        self.close_db_connection()

if __name__ == "__main__":
    config = OrchestratorConfig(
        num_agents=2,
        max_rounds=2,
        agent_config=AgentConfig(
            num_units=5,
            base_value=100,
            use_llm=True,
            initial_cash=1000,
            initial_goods=0,
            noise_factor=0.1,
            max_relative_spread=0.2
        ),
        llm_config=LLMConfig(
            client='openai',
            model='gpt-4o-mini',
            temperature=0.5,
            response_format='json_object',
            max_tokens=4096,
            use_cache=True
        ),
        environment_configs={
            'auction': AuctionConfig(
                name='Auction',
                address='auction_env_1',
                auction_type='double',
                max_steps=5,
            ),
        },
        protocol=ACLMessage,
        database_config={
            'db_type': 'postgres',
            'db_name': 'market_simulation'
        }
    )
    
    orchestrator = Orchestrator(config)
    orchestrator.start()
