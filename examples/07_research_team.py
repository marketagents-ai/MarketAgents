import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.market_agent_team import MarketAgentTeam
from market_agents.environments.mechanisms.research import ResearchEnvironment
from market_agents.environments.mechanisms.chat import ChatEnvironment
from minference.lite.models import LLMConfig, ResponseFormat

async def create_research_director():
    """Create a research director agent."""
    
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Research Director",
        persona="I am a research director responsible for leading market research initiatives. I define research objectives, coordinate research activities, and synthesize findings into actionable insights.",
        objectives=[
            "Define clear research objectives and methodologies",
            "Coordinate research team activities and ensure quality",
            "Synthesize research findings into strategic insights",
            "Translate research into actionable recommendations"
        ],
        communication_style="Strategic and integrative",
        skills=[
            "Research leadership",
            "Strategic analysis",
            "Cross-functional coordination",
            "Insight development"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="research_director",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Research Director with ID: {agent.id}")
    
    return agent

async def create_market_analyst():
    """Create a market analyst agent."""
    
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Market Analyst",
        persona="I am a market analyst specializing in industry trends, competitive landscapes, and market sizing. I gather and analyze market data to identify opportunities and threats.",
        objectives=[
            "Analyze industry trends and market dynamics",
            "Evaluate competitive landscapes and market positioning",
            "Conduct market sizing and growth projections",
            "Identify market opportunities and threats"
        ],
        communication_style="Analytical and fact-based",
        skills=[
            "Market research",
            "Competitive analysis",
            "Trend identification",
            "Data visualization"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="market_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Market Analyst with ID: {agent.id}")
    
    return agent

async def create_consumer_insights_specialist():
    """Create a consumer insights specialist agent."""
    
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Consumer Insights Specialist",
        persona="I am a consumer insights specialist focused on understanding customer behaviors, preferences, and needs. I translate customer data into actionable insights for product and marketing strategies.",
        objectives=[
            "Understand customer behaviors and preferences",
            "Identify unmet customer needs and pain points",
            "Segment customers based on meaningful attributes",
            "Translate customer insights into product and marketing recommendations"
        ],
        communication_style="Empathetic and customer-focused",
        skills=[
            "Customer research",
            "Behavioral analysis",
            "Segmentation",
            "Qualitative research interpretation"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="consumer_insights_specialist",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Consumer Insights Specialist with ID: {agent.id}")
    
    return agent

async def create_research_team():
    """Create a market research team with multiple specialized agents."""
    
    # Create individual agents
    research_director = await create_research_director()
    market_analyst = await create_market_analyst()
    consumer_insights_specialist = await create_consumer_insights_specialist()
    
    # Define research environment
    research_env = {
        "name": "market_research",
        "mechanism": "research",
        "knowledge_base_name": "market_research_kb",
        "research_tools": ["web_search", "data_analysis", "survey_analysis"],
        "research_depth": "comprehensive",
        "output_format": "structured_report"
    }
    
    # Define chat environment for team discussions
    chat_env = {
        "name": "research_team_chat",
        "mechanism": "chat",
        "form_cohorts": False,
        "sub_rounds": 3,
        "group_size": 3,
        "task_prompt": "Collaborate to conduct market research and develop insights."
    }
    
    # Create the research team with multiple environments
    research_team = MarketAgentTeam(
        name="Market Research Team",
        manager=research_director,
        agents=[
            market_analyst,
            consumer_insights_specialist
        ],
        mode="collaborative",
        use_group_chat=True,
        shared_context={
            "research_methodology": "Mixed-methods approach combining quantitative market analysis and qualitative consumer insights",
            "research_objectives": ["Identify market opportunities", "Understand customer needs", "Evaluate competitive landscape"],
            "data_sources": ["Industry reports", "Customer surveys", "Competitor analysis", "Market trends"],
            "output_requirements": "Actionable insights with supporting data and clear recommendations"
        },
        environments=[
            research_env,
            chat_env
        ]
    )
    
    print(f"Created Research Team: {research_team.name}")
    print(f"Team members: {[agent.id for agent in research_team.agents]}")
    print(f"Team manager: {research_team.manager.id}")
    print(f"Team environments: {[env['name'] for env in research_team.environments]}")
    
    return research_team

async def run_market_research(team, research_brief):
    """Run a market research project with the research team."""
    
    task = f"""
    Conduct comprehensive market research based on the following research brief:
    
    RESEARCH BRIEF:
    {research_brief}
    
    Your research team should:
    
    1. The Research Director should:
       - Define the research approach and methodology
       - Coordinate the research activities
       - Synthesize findings into strategic insights
       - Develop final recommendations
    
    2. The Market Analyst should:
       - Analyze industry trends and market dynamics
       - Evaluate the competitive landscape
       - Conduct market sizing and growth projections
       - Identify market opportunities and threats
    
    3. The Consumer Insights Specialist should:
       - Analyze customer behaviors and preferences
       - Identify unmet customer needs and pain points
       - Develop customer segments and personas
       - Translate customer insights into product and marketing implications
    
    Use the research environment to gather and analyze information, and the chat environment to collaborate and discuss findings.
    
    The final output should be a comprehensive research report that includes:
    - Executive summary
    - Market analysis (size, trends, competition)
    - Consumer insights (behaviors, needs, segments)
    - Strategic implications and opportunities
    - Actionable recommendations
    - Supporting data and methodology
    """
    
    print(f"\nStarting market research project: {research_brief[:100]}...")
    print("-" * 80)
    
    # Run the team research project
    result = await team.execute(task)
    
    print("\nMarket Research Complete")
    print("-" * 80)
    print(result)
    
    return result

async def main():
    # Create a market research team
    team = await create_research_team()
    
    # Run market research projects for different briefs
    research_briefs = [
        """
        Research Topic: Smart Home Technology Market
        
        Research Objectives:
        1. Understand the current state and future growth potential of the smart home technology market
        2. Identify key customer segments, their needs, and adoption barriers
        3. Analyze competitive landscape and identify market opportunities
        4. Develop recommendations for product development and go-to-market strategy
        
        Key Questions:
        - What is the current market size and projected growth for smart home technologies?
        - Who are the key customer segments and what are their primary use cases?
        - What are the main barriers to adoption and how can they be addressed?
        - Who are the key competitors and what are their strengths/weaknesses?
        - What are the most promising product categories and features?
        - What distribution channels and partnerships are most effective?
        
        Target Audience: Product development and marketing teams planning to enter the smart home market
        with new connected devices and services.
        """,
        
        """
        Research Topic: Sustainable Consumer Products
        
        Research Objectives:
        1. Assess consumer attitudes and behaviors regarding sustainable products
        2. Identify price sensitivity and willingness to pay premiums for sustainability
        3. Evaluate competitive landscape and positioning strategies
        4. Develop recommendations for sustainable product development and marketing
        
        Key Questions:
        - How do consumers define and evaluate "sustainability" in products?
        - What sustainability features/attributes drive purchase decisions?
        - How price sensitive are different consumer segments for sustainable products?
        - What messaging and certification strategies are most effective?
        - Who are the leading brands in sustainable products and what drives their success?
        - What are the emerging trends and innovations in sustainable consumer products?
        
        Target Audience: Consumer packaged goods company planning to launch a new line of
        sustainable household products.
        """
    ]
    
    for brief in research_briefs:
        await run_market_research(team, brief)

if __name__ == "__main__":
    asyncio.run(main())
