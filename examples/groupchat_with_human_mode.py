import asyncio
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Type
import psycopg2
from psycopg2.extras import execute_values
import yaml
from pydantic import BaseModel, Field
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.group_chat import (
    GroupChat, GroupChatAction, GroupChatActionSpace, GroupChatGlobalAction, 
    GroupChatMessage, GroupChatObservationSpace
)
from market_agents.inference.message_models import LLMConfig
from market_agents.agents.personas.persona import Persona, generate_persona, save_persona_to_file
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.economics.econ_agent import EconomicAgent

from colorama import Fore, Style, init

import logging



def setup_logging():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        # Initialize colorama
        init(autoreset=True)
        
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Style.BRIGHT,
                'PERCEPTION': Fore.BLUE,
                'ACTION': Fore.MAGENTA,
                'REFLECTION': Fore.YELLOW,
            }
        
            def format(self, record):
                levelname = record.levelname
                message = super().format(record)
                color = self.COLORS.get(getattr(record, 'custom_level', levelname), '')
                return f"{color}{message}{Style.RESET_ALL}"
        
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter('%(levelname)s:%(name)s:%(message)s'))
        
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG) 
        
        logger.propagate = False
    return logger

logger = setup_logging()

class GroupChatConfig(BaseModel):
    max_rounds: int = 2
    num_agents: int = 2
    initial_topic: str = "Market Trends"
    agent_roles: List[str] = ["analyst", "trader", "economist", "news reporter", "investor"]
    llm_config: LLMConfig
    agent_config: Dict[str, Any]
    protocol: Type[ACLMessage] = ACLMessage
    human_mode: bool = Field(default=False, description="Whether to enable human interaction")

class GroupChatOrchestrator:
    def __init__(self, config: GroupChatConfig):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.environment: MultiAgentEnvironment = None
        self.topic_proposer = None
        self.current_topic = None
        self.timeline: List[Dict[str, Any]] = []
        self.human_mode = config.human_mode
        self.db_params = {
            'dbname': 'market_simulation',
            'user': 'db_user',
            'password': 'db_pwd@123',
            'host': 'localhost',
            'port': '5433'
        }
        self.conn = psycopg2.connect(**self.db_params)

    def setup_agents(self):
        personas = self.load_or_generate_personas()
        for i, persona in enumerate(personas):
            agent = MarketAgent.create(
                agent_id=str(i),
                use_llm=self.config.agent_config.get('use_llm', True),
                llm_config=self.config.llm_config,
                environments={"group_chat": self.environment},
                protocol=self.config.protocol,
                persona=persona,
                econ_agent=EconomicAgent(
                    id=str(i),
                    initial_cash=self.config.agent_config.get('initial_cash', 1000),
                    initial_goods=self.config.agent_config.get('initial_goods', 0),
                    good_name=self.config.agent_config.get('good_name', "apple"),
                    endowment={"value": 100} 
                )
            )
            self.agents.append(agent)
            logger.debug(f"Initialized agent {agent.id} with persona {persona}")
        
        # Set the topic proposer
        self.topic_proposer = random.choice(self.agents)
        self.topic_proposer.system = "You are the group chat topic proposer agent. Your role is to propose interesting and relevant topics for the group discussion."
    async def generate_initial_topic(self) -> str:
        topic_action = await self.topic_proposer.generate_action(
            "group_chat", 
            "Consider recent economic events, market trends, or financial news. For this round discuss how Fed rate cut of 50 bps is going to impact the market"
        )
        logger.debug(f"Topic proposer {self.topic_proposer.id} generated action: {topic_action}")
        
        try:
            # The structure of topic_action might be different from what we expect
            # Let's handle different possible structures
            if isinstance(topic_action, dict):
                if 'content' in topic_action:
                    if isinstance(topic_action['content'], dict) and 'action' in topic_action['content']:
                        content = topic_action['content']['action']['content']
                    else:
                        content = topic_action['content']
                elif 'action' in topic_action:
                    content = topic_action['action']['content']
                else:
                    raise ValueError("Unexpected topic_action structure")
            else:
                raise ValueError("topic_action is not a dictionary")

            return content
        except Exception as e:
            logger.error(f"Invalid topic action structure: {e}")
            return "Default topic: Recent market trends"

    def setup_environment(self):
        group_chat = GroupChat(
            max_rounds=self.config.max_rounds,
            current_topic="Placeholder topic",
            speaker_order=[f"{agent.id}" for agent in self.agents]
        )
        self.environment = MultiAgentEnvironment(
            name="group_chat",
            address="group_chat_address",
            max_steps=self.config.max_rounds,
            action_space=GroupChatActionSpace(),
            observation_space=GroupChatObservationSpace(),
            mechanism=group_chat
        )
        logger.debug("Environment setup completed.")
    async def get_human_input(self) -> str:
        print(f"{Fore.RED}Human's turn. Enter your message (or wait 15 seconds to skip):{Style.RESET_ALL}")
        try:
            return await asyncio.wait_for(asyncio.to_thread(input), timeout=15)
        except asyncio.TimeoutError:
            print("No input received within 15 seconds. Continuing...")
            return ""


    async def run_simulation(self):
        # Generate the initial topic
        self.current_topic = await self.generate_initial_topic()
        
        # Update the environment with the generated topic
        self.environment.mechanism.current_topic = self.current_topic

        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"Starting round {round_num}")
            await self.process_round()
        
        self.generate_report()
    

    def insert_groupchat_data(self, round_data: Dict[str, Any]):
        query = """
        INSERT INTO groupchat (round_number, topic, agent_id, message_type, content, is_human)
        VALUES %s
        """
        values = [
            (
                round_data['round'],
                round_data['topic'],
                str(message.agent_id),
                message.message_type,
                message.content,
                message.agent_id == 'human'
            )
            for message in round_data['messages']
        ]
        
        with self.conn.cursor() as cur:
            execute_values(cur, query, values)
        self.conn.commit()
    
    
    async def process_round(self):
        actions = {}
        round_num = len(self.timeline) + 1
        
        logger.info(f"\n--- Round {round_num} ---")
        
        for agent in self.agents:
            logger.info(f"\nAgent {agent.id}:")
            perception = await agent.perceive("group_chat")
            logger.info(f"Perception received: {perception}", extra={'custom_level': 'PERCEPTION'})
            
            action_dict = await agent.generate_action("group_chat", perception)
            logger.info(f"Action generated: {action_dict}", extra={'custom_level': 'ACTION'})
            
            try:
                content = action_dict['content']['action']['content']
                message_type = action_dict['content']['action']['message_type']
                if message_type not in ['propose_topic', 'group_message']:
                    message_type = 'group_message'
                
                group_chat_message = GroupChatMessage(content=content, message_type=message_type, agent_id=str(agent.id))
                group_chat_action = GroupChatAction(agent_id=str(agent.id), action=group_chat_message)
                
                actions[str(agent.id)] = group_chat_action.dict()
                logger.info(f"Action processed: {content}")
            except KeyError as e:
                logger.error(f"Invalid action structure: {e}")
                continue

        if self.human_mode:
            human_input = await self.get_human_input()
            
            if human_input.strip():
                logger.info(f"\nHuman input received: {human_input}")
                human_message = GroupChatMessage(content=human_input, message_type="group_message", agent_id="human")
                actions["human"] = GroupChatAction(agent_id="human", action=human_message).dict()
                
                is_first_round_dissatisfaction = round_num == 1 and "not satisfied" in human_input.lower()
                
                if is_first_round_dissatisfaction:
                    logger.info("\nHuman expressed dissatisfaction. Regenerating agent actions.")
                    for agent in self.agents:
                        logger.info(f"\nRegenerating action for Agent {agent.id}:")
                        perception = await agent.perceive("group_chat")
                        
                        perception += f"\n\nInitial topic: {self.current_topic}\n"
                        perception += f"Your recent response: {actions[str(agent.id)]['action']['content']}\n"
                        perception += "The human expressed dissatisfaction. Please consider both the initial topic and your recent response when generating a new response."
                        
                        logger.info(f"New perception: {perception}", extra={'custom_level': 'PERCEPTION'})
                        
                        action_dict = await agent.generate_action("group_chat", perception)
                        
                        try:
                            content = action_dict['content']['action']['content']
                            message_type = action_dict['content']['action']['message_type']
                            if message_type not in ['propose_topic', 'group_message']:
                                message_type = 'group_message'
                            
                            group_chat_message = GroupChatMessage(content=content, message_type=message_type, agent_id=str(agent.id))
                            group_chat_action = GroupChatAction(agent_id=str(agent.id), action=group_chat_message)
                            
                            actions[str(agent.id)] = group_chat_action.dict()
                            logger.info(f"Action regenerated: {content}", extra={'custom_level': 'ACTION'})
                        except KeyError as e:
                            logger.error(f"Invalid action structure after human input: {e}")
                            continue

        logger.info("\nProcessing environment step")
        global_action = GroupChatGlobalAction(actions=actions)
        
        try:
            step_result = self.environment.step(global_action)
        except Exception as e:
            logger.error(f"Error during environment step: {e}")
            raise
        
        logger.debug(f"Step result: {step_result}")
        
        all_messages = [GroupChatMessage(**action['action']) for action in actions.values()]
        current_topic = step_result.global_observation.current_topic or self.current_topic
        round_data = {
            "round": len(self.timeline) + 1,
            "topic": current_topic,
            "messages": all_messages
        }
        self.update_timeline(all_messages, current_topic)
        self.insert_groupchat_data(round_data)
        self.current_topic = current_topic

        logger.info("\nAgent reflections:")
        for agent in self.agents:
            reflection = await agent.reflect("group_chat")
            logger.info(f"Agent {agent.id} reflection: {reflection}", extra={'custom_level': 'REFLECTION'})

        return step_result

    def load_or_generate_personas(self) -> List[Persona]:
        personas_dir = Path("./market_agents/agents/personas/generated_personas")
        existing_personas = []

        if os.path.exists(personas_dir):
            for filename in os.listdir(personas_dir):
                if filename.endswith(".yaml"):
                    with open(os.path.join(personas_dir, filename), 'r') as file:
                        persona_data = yaml.safe_load(file)
                        existing_personas.append(Persona(**persona_data))

        while len(existing_personas) < self.config.num_agents:
            new_persona = generate_persona()
            existing_personas.append(new_persona)
            save_persona_to_file(new_persona, personas_dir)

        return existing_personas[:self.config.num_agents]

    def update_timeline(self, messages: List[GroupChatMessage], topic: str):
        self.timeline.append({
            "round": len(self.timeline) + 1,
            "topic": topic,
            "messages": messages
        })

    def generate_report(self):
        logger.info("Generating simulation report")
        for round_num, round_data in enumerate(self.timeline, 1):
            logger.info(f"Round {round_num}:")
            logger.info(f"Topic: {round_data['topic']}")
            for message in round_data['messages']:
                # Access attributes directly
                agent_id = message.agent_id
                content = message.content
                logger.info(f"Agent {agent_id}: {content}")

def run_simulation():
    config = GroupChatConfig(
        max_rounds=2,
        num_agents=2,
        initial_topic="Market Trends",
        agent_roles=["analyst", "trader", "economist", "news reporter", "investor"],
        llm_config=LLMConfig(
            client='openai',
            model='gpt-4o-mini',
            temperature=0.7,
            max_tokens=512,
            use_cache=True
        ),
        agent_config={
            'use_llm': True,
            'initial_cash': 1000,
            'initial_goods': 0,
            'good_name': "apple",
        },
        protocol=ACLMessage,
        human_mode=False  
    )
    orchestrator = GroupChatOrchestrator(config)
    orchestrator.setup_environment()
    orchestrator.setup_agents()
    
    try:
        asyncio.run(orchestrator.run_simulation())
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        print(f"Simulation failed: {str(e)}")

    if orchestrator.timeline:
        for round_data in orchestrator.timeline:
            print(f"\n{Fore.CYAN}--- Round {round_data['round']} ---{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Topic: {round_data['topic']}{Style.RESET_ALL}")
            for message in round_data['messages']:
                # Access attributes directly
                agent_id = message.agent_id
                content = message.content
                message_type = message.message_type
                if message_type == 'propose_topic':
                    print(f"{Fore.GREEN}Agent {agent_id} (Topic Proposer): {content}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.MAGENTA}Agent {agent_id}: {content}{Style.RESET_ALL}")
    else:
        print("No simulation data available.")

if __name__ == "__main__":
    run_simulation()