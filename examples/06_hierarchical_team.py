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
from minference.lite.models import LLMConfig, ResponseFormat

async def create_investment_strategist():
    """Create an investment strategist agent."""
    
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
        role="Investment Strategist",
        persona="I am an investment strategist with expertise in portfolio construction, asset allocation, and market analysis. I develop comprehensive investment strategies based on client goals, risk tolerance, and market conditions.",
        objectives=[
            "Develop sound investment strategies aligned with client objectives",
            "Analyze market trends and economic indicators",
            "Optimize asset allocation for risk-adjusted returns",
            "Provide strategic investment recommendations"
        ],
        communication_style="Analytical and thoughtful",
        skills=[
            "Portfolio construction",
            "Asset allocation",
            "Risk management",
            "Market analysis"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="investment_strategist",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Investment Strategist with ID: {agent.id}")
    
    return agent

async def create_equity_analyst():
    """Create an equity analyst agent."""
    
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
        role="Equity Analyst",
        persona="I am an equity analyst specializing in fundamental analysis of public companies. I evaluate financial statements, competitive positioning, and growth prospects to determine investment potential.",
        objectives=[
            "Conduct thorough fundamental analysis of companies",
            "Evaluate competitive advantages and market positioning",
            "Assess financial health and growth prospects",
            "Provide valuation estimates and investment recommendations"
        ],
        communication_style="Detail-oriented and evidence-based",
        skills=[
            "Financial statement analysis",
            "Industry research",
            "Valuation modeling",
            "Competitive analysis"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="equity_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Equity Analyst with ID: {agent.id}")
    
    return agent

async def create_macro_economist():
    """Create a macro economist agent."""
    
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
        role="Macro Economist",
        persona="I am a macro economist focused on analyzing economic trends, monetary policy, and global economic conditions. I provide insights on how macroeconomic factors impact investment opportunities and risks.",
        objectives=[
            "Analyze economic indicators and trends",
            "Evaluate monetary and fiscal policy implications",
            "Assess global economic conditions and geopolitical risks",
            "Provide macroeconomic outlook and investment implications"
        ],
        communication_style="Thoughtful and contextual",
        skills=[
            "Economic analysis",
            "Policy interpretation",
            "Forecasting",
            "Global markets understanding"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="macro_economist",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Macro Economist with ID: {agent.id}")
    
    return agent

async def create_risk_manager():
    """Create a risk manager agent."""
    
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
        role="Risk Manager",
        persona="I am a risk manager specializing in identifying, measuring, and mitigating investment risks. I ensure portfolios maintain appropriate risk levels and implement risk management strategies.",
        objectives=[
            "Identify and assess investment risks",
            "Develop risk mitigation strategies",
            "Monitor portfolio risk metrics",
            "Ensure compliance with risk parameters"
        ],
        communication_style="Precise and cautious",
        skills=[
            "Risk modeling",
            "Scenario analysis",
            "Portfolio stress testing",
            "Correlation analysis"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="risk_manager",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Risk Manager with ID: {agent.id}")
    
    return agent

async def create_investment_team():
    """Create an investment team with multiple specialized agents."""
    
    # Create individual agents
    investment_strategist = await create_investment_strategist()
    equity_analyst = await create_equity_analyst()
    macro_economist = await create_macro_economist()
    risk_manager = await create_risk_manager()
    
    # Define chat environment
    chat_env = {
        "name": "investment_team_chat",
        "mechanism": "chat",
        "form_cohorts": False,
        "sub_rounds": 3,
        "group_size": 4,
        "task_prompt": "Collaborate to develop investment strategies and recommendations."
    }
    
    # Create the investment team
    investment_team = MarketAgentTeam(
        name="Investment Committee",
        manager=investment_strategist,
        agents=[
            equity_analyst,
            macro_economist,
            risk_manager
        ],
        mode="hierarchical",  # Using hierarchical mode with investment strategist as manager
        use_group_chat=True,
        shared_context={
            "investment_philosophy": "Long-term value investing with a focus on quality companies and risk management.",
            "client_profile": {
                "risk_tolerance": "Moderate",
                "investment_horizon": "7-10 years",
                "objectives": ["Capital appreciation", "Income generation", "Wealth preservation"]
            },
            "market_environment": {
                "current_conditions": "Late-cycle economic expansion with elevated valuations",
                "interest_rates": "Rising interest rate environment",
                "inflation_outlook": "Moderate inflation with upside risks"
            }
        },
        environments=[
            chat_env
        ]
    )
    
    print(f"Created Investment Team: {investment_team.name}")
    print(f"Team members: {[agent.id for agent in investment_team.agents]}")
    print(f"Team manager: {investment_team.manager.id}")
    
    return investment_team

async def run_investment_analysis(team, investment_scenario):
    """Run an investment analysis with the investment team."""
    
    task = f"""
    As an investment committee, analyze the following investment scenario and develop a comprehensive investment strategy:
    
    INVESTMENT SCENARIO:
    {investment_scenario}
    
    Your team should:
    
    1. The Investment Strategist (Manager) should:
       - Lead the discussion and synthesize insights from team members
       - Develop an overall investment strategy and asset allocation
       - Provide final investment recommendations
       - Ensure alignment with client objectives and risk tolerance
    
    2. The Equity Analyst should:
       - Analyze specific equity sectors and companies mentioned
       - Provide valuation perspectives and growth outlooks
       - Identify potential investment opportunities and risks
       - Recommend specific equity investments if appropriate
    
    3. The Macro Economist should:
       - Assess the macroeconomic environment and implications
       - Evaluate monetary and fiscal policy impacts
       - Identify economic trends and risks
       - Provide sector allocation recommendations based on economic outlook
    
    4. The Risk Manager should:
       - Identify key risks in the proposed investments
       - Suggest risk mitigation strategies
       - Evaluate portfolio-level risk considerations
       - Ensure recommendations align with risk parameters
    
    Through collaborative discussion, develop a comprehensive investment strategy that addresses the scenario while balancing return potential and risk management.
    
    The final output should include:
    - Executive summary of recommendations
    - Strategic asset allocation
    - Specific investment recommendations
    - Key risks and mitigation strategies
    - Implementation timeline and monitoring approach
    """
    
    print(f"\nStarting investment analysis for scenario: {investment_scenario[:100]}...")
    print("-" * 80)
    
    # Run the team discussion in hierarchical mode
    result = await team.execute(task)
    
    print("\nInvestment Analysis Complete")
    print("-" * 80)
    print(result)
    
    return result

async def main():
    # Create an investment team
    team = await create_investment_team()
    
    # Run investment analyses for different scenarios
    investment_scenarios = [
        """
        Client Profile: A high-net-worth individual (age 55) planning for retirement in 10 years with $5 million to invest.
        The client seeks a balanced approach with capital preservation and moderate growth. They have existing real estate
        investments and are concerned about inflation and market volatility.
        
        Current Market Environment: Equity markets are at all-time highs with signs of sector rotation from growth to value.
        Interest rates are expected to rise over the next 12-18 months. Inflation indicators are showing upward pressure.
        Geopolitical tensions are creating uncertainty in global markets.
        
        Investment Considerations:
        - Need for portfolio to generate retirement income within 10 years
        - Inflation protection
        - Tax efficiency
        - Appropriate risk management given current market valuations
        - Potential for market correction in the near term
        """,
        
        """
        Client Profile: A university endowment fund with $100 million in assets and a perpetual time horizon. The endowment
        needs to generate 4% annual distributions to support university operations while maintaining purchasing power over time.
        The investment committee has expressed interest in sustainable investing and is willing to consider alternative investments.
        
        Current Market Environment: Global economic recovery is uneven across regions. Technology and healthcare sectors continue
        to show strong growth prospects. Private market valuations are elevated but offer diversification benefits. ESG investing
        is gaining momentum with regulatory support. Climate transition risks are increasingly material to long-term investors.
        
        Investment Considerations:
        - Long-term capital appreciation while supporting current distribution needs
        - Diversification across public and private markets
        - Integration of ESG factors and climate risk
        - Liquidity management to support distributions
        - Governance structure for alternative investments
        """
    ]
    
    for scenario in investment_scenarios:
        await run_investment_analysis(team, scenario)

if __name__ == "__main__":
    asyncio.run(main())
