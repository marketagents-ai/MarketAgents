import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.market_agent_team import MarketAgentTeam
from market_agents.environments.mechanisms.chat import ChatEnvironment
from market_agents.environments.mechanisms.research import ResearchEnvironment
from market_agents.orchestrators.mcp_server.finance_mcp_server import FinanceMCPServer
from minference.lite.models import LLMConfig, ResponseFormat

async def create_chief_strategy_officer():
    """Create a chief strategy officer agent."""
    
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
        role="Chief Strategy Officer",
        persona="I am a Chief Strategy Officer responsible for developing and executing strategic initiatives. I analyze market opportunities, competitive landscapes, and organizational capabilities to drive growth and innovation.",
        objectives=[
            "Develop comprehensive strategic plans",
            "Identify growth opportunities and market trends",
            "Evaluate competitive positioning and threats",
            "Align organizational resources with strategic priorities"
        ],
        communication_style="Strategic, visionary, and decisive",
        skills=[
            "Strategic planning",
            "Market analysis",
            "Competitive intelligence",
            "Business model innovation"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="chief_strategy_officer",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Chief Strategy Officer with ID: {agent.id}")
    
    return agent

async def create_market_research_director():
    """Create a market research director agent."""
    
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
        role="Market Research Director",
        persona="I am a Market Research Director specializing in gathering and analyzing market data to inform strategic decisions. I design and oversee research initiatives to provide actionable insights on markets, customers, and competitors.",
        objectives=[
            "Design comprehensive market research methodologies",
            "Analyze market trends and customer behaviors",
            "Identify market opportunities and threats",
            "Translate research findings into strategic recommendations"
        ],
        communication_style="Analytical, thorough, and insight-driven",
        skills=[
            "Market research design",
            "Data analysis",
            "Consumer behavior analysis",
            "Competitive intelligence"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="market_research_director",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Market Research Director with ID: {agent.id}")
    
    return agent

async def create_financial_analyst():
    """Create a financial analyst agent."""
    
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
        role="Financial Analyst",
        persona="I am a Financial Analyst specializing in financial modeling, valuation, and investment analysis. I evaluate financial data to assess business performance, investment opportunities, and financial risks.",
        objectives=[
            "Analyze financial statements and performance metrics",
            "Develop financial models and forecasts",
            "Evaluate investment opportunities and risks",
            "Provide data-driven financial recommendations"
        ],
        communication_style="Precise, quantitative, and evidence-based",
        skills=[
            "Financial modeling",
            "Valuation analysis",
            "Financial statement analysis",
            "Investment analysis"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="financial_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Financial Analyst with ID: {agent.id}")
    
    return agent

async def setup_environments():
    """Set up multiple environments for multi-environment orchestration."""
    
    # Set up Research Environment
    research_env = {
        "name": "market_research",
        "mechanism": "research",
        "knowledge_base_name": "strategic_research_kb",
        "research_tools": ["web_search", "data_analysis", "trend_analysis"],
        "research_depth": "comprehensive",
        "output_format": "structured_report"
    }
    
    # Set up Chat Environment
    chat_env = {
        "name": "strategy_team_chat",
        "mechanism": "chat",
        "form_cohorts": False,
        "sub_rounds": 3,
        "group_size": 3,
        "task_prompt": "Collaborate to develop strategic recommendations."
    }
    
    # Set up Finance MCP Environment
    mcp_finance_env = {
        "name": "mcp_finance",
        "mechanism": "mcp_server",
        "api_url": "local://mcp_server",
        "mcp_server_module": "market_agents.orchestrators.mcp_server.finance_mcp_server",
        "mcp_server_class": "FinanceMCPServer",
        "form_cohorts": False,
        "sub_rounds": 2,
        "group_size": 3,
        "task_prompt": "Use finance tools to analyze financial data and provide recommendations."
    }
    
    return {
        "research": research_env,
        "chat": chat_env,
        "mcp_finance": mcp_finance_env
    }

async def create_strategy_team_with_multi_environments():
    """Create a strategy team with multiple environments for orchestration."""
    
    # Create team members
    cso = await create_chief_strategy_officer()
    market_research_director = await create_market_research_director()
    financial_analyst = await create_financial_analyst()
    
    # Set up environments
    environments = await setup_environments()
    
    # Create the strategy team with multiple environments
    strategy_team = MarketAgentTeam(
        name="Strategic Planning Team",
        manager=cso,
        agents=[
            market_research_director,
            financial_analyst
        ],
        mode="hierarchical",
        use_group_chat=True,
        shared_context={
            "strategic_objectives": [
                "Identify new growth opportunities",
                "Evaluate market expansion options",
                "Assess competitive positioning",
                "Develop data-driven strategic recommendations"
            ],
            "planning_horizon": "3-5 years",
            "key_focus_areas": [
                "Market trends and disruptions",
                "Competitive landscape evolution",
                "Financial implications and investment requirements",
                "Strategic partnerships and ecosystem development"
            ]
        },
        environments=[
            environments["research"],
            environments["chat"],
            environments["mcp_finance"]
        ]
    )
    
    print(f"Created Strategy Team: {strategy_team.name}")
    print(f"Team members: {[agent.id for agent in strategy_team.agents]}")
    print(f"Team manager: {strategy_team.manager.id}")
    print(f"Team environments: {[env['name'] for env in strategy_team.environments]}")
    
    return strategy_team

async def run_multi_environment_strategic_analysis(team, strategic_challenge):
    """Run a strategic analysis using multiple environments for orchestration."""
    
    task = f"""
    As a strategic planning team, conduct a comprehensive analysis of the following strategic challenge and develop recommendations:
    
    STRATEGIC CHALLENGE:
    {strategic_challenge}
    
    Your team should utilize all available environments to conduct a thorough analysis:
    
    1. Use the Research Environment to:
       - Gather market data and industry trends
       - Analyze competitive landscape
       - Identify relevant case studies and best practices
       - Research potential disruptions and opportunities
    
    2. Use the Finance MCP Environment to:
       - Analyze financial implications and requirements
       - Evaluate potential ROI and financial risks
       - Model different scenarios and their financial outcomes
       - Assess investment requirements and funding options
    
    3. Use the Chat Environment to:
       - Collaborate on findings from research and financial analysis
       - Debate strategic options and their implications
       - Develop consensus on recommendations
       - Refine the strategic plan
    
    The Chief Strategy Officer should:
    - Coordinate the overall analysis process
    - Ensure alignment with strategic objectives
    - Synthesize insights from different analyses
    - Develop the final strategic recommendations
    
    The Market Research Director should:
    - Lead the market research and competitive analysis
    - Identify key market trends and customer needs
    - Evaluate market opportunities and threats
    - Provide market-based recommendations
    
    The Financial Analyst should:
    - Conduct financial analysis and modeling
    - Evaluate financial feasibility and risks
    - Assess resource requirements and ROI
    - Provide financial recommendations
    
    The final output should be a comprehensive strategic plan that includes:
    - Executive summary
    - Situation analysis (market, competition, internal capabilities)
    - Strategic options evaluation
    - Recommended strategy with rationale
    - Implementation roadmap
    - Financial projections and resource requirements
    - Risk assessment and mitigation strategies
    - Key performance indicators and success metrics
    """
    
    print(f"\nStarting multi-environment strategic analysis for challenge: {strategic_challenge[:100]}...")
    print("-" * 80)
    
    # Run the team analysis using multiple environments
    result = await team.execute(task)
    
    print("\nStrategic Analysis Complete")
    print("-" * 80)
    print(result)
    
    return result

async def main():
    # Create a strategy team with multiple environments
    team = await create_strategy_team_with_multi_environments()
    
    # Run strategic analyses for different challenges
    strategic_challenges = [
        """
        Strategic Challenge: Digital Transformation in Financial Services
        
        Our mid-sized financial services company (5,000 employees, $2B annual revenue) is facing increasing pressure from digital-native 
        competitors and changing customer expectations. Traditional revenue streams from in-person banking services and legacy products 
        are declining at 5-7% annually, while digital channels are growing but not fast enough to offset the decline.
        
        Key Considerations:
        - Our technology infrastructure is aging (15+ years old) and integration costs are high
        - Customer acquisition costs have increased 30% in the last 3 years
        - Fintech competitors are capturing younger demographics with superior digital experiences
        - Regulatory requirements for data security and privacy are becoming more stringent
        - We have strong brand recognition and customer trust in our established markets
        - We have significant capital reserves ($500M) available for strategic investments
        
        Develop a comprehensive digital transformation strategy that addresses:
        - Technology modernization approach and priorities
        - Product and service innovation opportunities
        - Customer experience transformation
        - Organizational capabilities and talent requirements
        - Implementation roadmap and resource allocation
        - Financial projections and expected outcomes
        - Risk management and mitigation strategies
        """,
        
        """
        Strategic Challenge: Sustainable Product Innovation and Market Expansion
        
        Our consumer products company specializes in household goods and personal care products ($3B annual revenue). 
        We are facing increasing consumer demand for sustainable products, regulatory pressure on packaging and ingredients, 
        and competition from both premium eco-friendly brands and private label alternatives.
        
        Key Considerations:
        - Sustainability initiatives typically increase production costs by 15-25%
        - Our current R&D capabilities in sustainable materials are limited
        - Consumer willingness to pay premiums for sustainable products varies significantly by market segment
        - Retailers are demanding more sustainable packaging and carbon footprint reductions
        - We have strong manufacturing capabilities and distribution networks
        - Competitors are making significant sustainability claims, some with questionable verification
        - Emerging markets represent significant growth opportunities but have different sustainability priorities
        
        Develop a comprehensive strategy that addresses:
        - Sustainable product innovation roadmap
        - Market segmentation and targeting strategy
        - Pricing and positioning approach
        - Supply chain transformation requirements
        - Geographic expansion priorities
        - Sustainability messaging and certification strategy
        - Financial implications and investment requirements
        - Implementation timeline and resource allocation
        """
    ]
    
    for challenge in strategic_challenges:
        await run_multi_environment_strategic_analysis(team, challenge)

if __name__ == "__main__":
    asyncio.run(main())
