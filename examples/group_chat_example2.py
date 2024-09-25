import asyncio
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Type

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
    max_rounds: int = 10
    num_agents: int = 5
    initial_topic: str = "Market Trends"
    agent_roles: List[str] = ["analyst", "trader", "economist", "news reporter", "investor"]
    llm_config: LLMConfig
    agent_config: Dict[str, Any]
    protocol: Type[ACLMessage] = ACLMessage

class GroupChatOrchestrator:
    def __init__(self, config: GroupChatConfig):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.environment: MultiAgentEnvironment = None
        self.topic_proposer = None
        self.current_topic = None
        self.timeline: List[Dict[str, Any]] = []

    def setup_agents(self):
        personas = self.load_or_generate_personas()
        for i, persona in enumerate(personas):
            agent = MarketAgent.create(
                agent_id=i,
                is_buyer=True,
                **self.config.agent_config,
                llm_config=self.config.llm_config,
                environments={"group_chat": self.environment},
                protocol=self.config.protocol,
                persona=persona
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
            # Ensure the 'actions' key exists
            if 'actions' not in topic_action['content']:
                raise KeyError('actions')
            
            # Extract the content from the new structure
            topic_content = topic_action['content']['actions']['content']
            
            # Ensure the message type is 'propose_topic'
            if topic_action['content']['actions']['message_type'] != 'propose_topic':
                logger.warning(f"Expected 'propose_topic' message type, got '{topic_action['content']['actions']['message_type']}'. Proceeding anyway.")
            
        except KeyError as e:
            logger.error(f"Invalid topic action structure: {e}")
            # Fallback to a default topic if extraction fails
            topic_content = "Discuss the recent Fed rate cut of 50 bps and its potential impact on the market."
        
        logger.info(f"Initial topic: {topic_content}")
        return topic_content

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

    async def run_simulation(self):
        # Generate the initial topic
        self.current_topic = await self.generate_initial_topic()
        
        # Update the environment with the generated topic
        self.environment.mechanism.current_topic = self.current_topic

        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"Starting round {round_num}")
            await self.process_round()
        
        self.generate_report()

    async def process_round(self):
        actions = {}
        
        for agent in self.agents:
            perception = await agent.perceive("group_chat")
            logger.info(f"Agent {agent.id} perception: {perception}", extra={'custom_level': 'PERCEPTION'})
            
            action_dict = await agent.generate_action("group_chat", perception)
            logger.info(f"Agent {agent.id} action_dict: {action_dict}", extra={'custom_level': 'ACTION'})
            
            try:
                content = action_dict['content']['action']['content']
                message_type = action_dict['content']['action']['message_type']
                if message_type not in ['propose_topic', 'group_message']:
                    message_type = 'group_message'  # Default to 'group_message' if not valid
                
                group_chat_message = GroupChatMessage(content=content, message_type=message_type, agent_id=str(agent.id))
                group_chat_action = GroupChatAction(agent_id=str(agent.id), action=group_chat_message)
                
                actions[str(agent.id)] = group_chat_action.dict()  # Convert to dictionary
                logger.info(f"Agent {agent.id} action: {group_chat_message.content}")
            except KeyError as e:
                logger.error(f"Invalid action structure for agent {agent.id}: {e}")
                continue

        logger.debug(f"Aggregated actions: {actions}")

        global_action = GroupChatGlobalAction(actions=actions)
        
        try:
            step_result = self.environment.step(global_action)
        except Exception as e:
            logger.error(f"Error during environment step: {e}")
            raise
        
        logger.debug(f"Step result: {step_result}")
        
        # Update timeline and current topic
        all_messages = [GroupChatMessage(**action['action']) for action in actions.values()]
        current_topic = step_result.global_observation.current_topic or self.current_topic
        self.update_timeline(all_messages, current_topic)
        self.current_topic = current_topic

        # Reflect actions in agents
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
        num_agents=5,
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
            'num_units': 1,
            'base_value': 100,
            'use_llm': True,
            'initial_cash': 1000,
            'initial_goods': 0,
            'good_name': "apple",
        },
        protocol=ACLMessage
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