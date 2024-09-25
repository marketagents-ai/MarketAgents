import asyncio
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Type

import yaml
import gradio as gr
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
            topic_content = topic_action['content']['action']['content']
        except (KeyError, TypeError) as e:
            logger.error(f"Invalid topic action structure: {e}")
            raise
        
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
        
        return self.generate_report()

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
                
                actions[str(agent.id)] = group_chat_action
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
        all_messages = [action.action for action in actions.values()]
        current_topic = step_result.global_observation.current_topic or self.current_topic
        self.update_timeline(all_messages, current_topic)
        self.current_topic = current_topic

        # Reflect actions in agents
        for agent in self.agents:
            reflection = await agent.reflect("group_chat")
            logger.info(f"Agent {agent.id} reflection: {reflection}", extra={'custom_level': 'REFLECTION'})

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
        report = []
        logger.info("Generating simulation report")
        for round_data in self.timeline:
            report.append(f"Round {round_data['round']}:")
            report.append(f"Topic: {round_data['topic']}")
            for message in round_data['messages']:
                if message.message_type == 'propose_topic':
                    report.append(f"Agent {message.agent_id} (Topic Proposer): {message.content}")
                else:
                    report.append(f"Agent {message.agent_id}: {message.content}")
            report.append("\n")
        return "\n".join(report)

def create_orchestrator():
    config = GroupChatConfig(
        max_rounds=10,
        num_agents=5,
        initial_topic="Market Trends",
        agent_roles=["analyst", "trader", "economist", "news reporter", "investor"],
        llm_config=LLMConfig(
            client='openai',
            model='gpt-4-0613',
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
    return orchestrator

async def run_simulation():
    orchestrator = create_orchestrator()
    report = await orchestrator.run_simulation()
    return report

def gradio_interface():
    return asyncio.run(run_simulation())

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[],
    outputs="text",
    title="Group Chat Simulation",
    description="Simulate a group chat discussion about market trends and investment strategies.",
)

if __name__ == "__main__":
    iface.launch()