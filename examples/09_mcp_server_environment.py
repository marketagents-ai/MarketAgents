import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.market_agent_team import MarketAgentTeam
from market_agents.orchestrators.mcp_server.finance_mcp_server import FinanceMCPServer
from minference.lite.models import LLMConfig, ResponseFormat

async def create_financial_advisor():
    """Create a financial advisor agent."""
    
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
        role="Financial Advisor",
        persona="I am a financial advisor specializing in personal finance, investment planning, and wealth management. I provide comprehensive financial advice to help clients achieve their financial goals.",
        objectives=[
            "Develop personalized financial plans based on client goals",
            "Provide investment recommendations aligned with risk tolerance",
            "Analyze financial data to identify opportunities and risks",
            "Educate clients on financial concepts and strategies"
        ],
        communication_style="Clear, educational, and client-focused",
        skills=[
            "Financial planning",
            "Investment management",
            "Risk assessment",
            "Retirement planning"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="financial_advisor",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Financial Advisor with ID: {agent.id}")
    
    return agent

async def setup_finance_mcp_environment(agent):
    """Set up the Finance MCP Server environment for the agent."""
    
    # Create Finance MCP Server
    finance_mcp = FinanceMCPServer()
    
    # Define MCP environment configuration
    mcp_finance_env = {
        "name": "mcp_finance",
        "mechanism": "mcp_server",
        "api_url": "local://mcp_server",
        "mcp_server_module": "market_agents.orchestrators.mcp_server.finance_mcp_server",
        "mcp_server_class": "FinanceMCPServer",
        "form_cohorts": False,
        "sub_rounds": 2,
        "group_size": 1,
        "task_prompt": "Use finance tools to analyze financial data and provide recommendations."
    }
    
    # Add environment to agent
    agent.add_environment("mcp_finance", mcp_finance_env)
    
    print(f"Added Finance MCP environment to agent {agent.id}")
    
    return mcp_finance_env

async def run_financial_analysis(agent, query):
    """Run a financial analysis using the Finance MCP Server environment."""
    
    print(f"\nRunning financial analysis for query: {query}")
    print("-" * 80)
    
    # Create task for the agent
    task = f"""
    You are {agent.role}. {agent.persona}
    
    Use the Finance MCP Server tools to analyze the following financial query and provide a comprehensive response:
    
    QUERY: {query}
    
    Available Finance Tools:
    1. get_stock_data - Retrieve current and historical stock data
    2. get_company_financials - Retrieve company financial statements
    3. calculate_financial_ratios - Calculate key financial ratios
    4. analyze_portfolio - Analyze portfolio performance and risk
    5. get_economic_indicators - Retrieve current economic indicators
    
    Your response should:
    - Use appropriate finance tools to gather relevant data
    - Analyze the data to provide insights and recommendations
    - Explain your analysis in clear, educational terms
    - Provide specific, actionable recommendations
    - Include relevant charts or visualizations when helpful
    
    Ensure your response is comprehensive, data-driven, and tailored to the query.
    """
    
    # Execute the task in the MCP environment
    result = await agent.execute_in_environment("mcp_finance", task)
    
    print("\nFinancial Analysis Complete")
    print("-" * 80)
    print(result)
    
    return result

async def create_investment_team_with_mcp():
    """Create an investment team with access to the Finance MCP Server."""
    
    # Create financial advisor as the team manager
    financial_advisor = await create_financial_advisor()
    
    # Set up Finance MCP environment
    mcp_finance_env = await setup_finance_mcp_environment(financial_advisor)
    
    # Create additional team members
    # Portfolio Manager
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    portfolio_manager_persona = Persona(
        role="Portfolio Manager",
        persona="I am a portfolio manager responsible for constructing and managing investment portfolios. I focus on asset allocation, security selection, and risk management to achieve client objectives.",
        objectives=[
            "Develop optimal portfolio allocations based on client goals",
            "Select appropriate securities for each asset class",
            "Monitor and adjust portfolios based on market conditions",
            "Manage portfolio risk through diversification and hedging"
        ],
        communication_style="Analytical and precise",
        skills=[
            "Asset allocation",
            "Security selection",
            "Portfolio optimization",
            "Risk management"
        ]
    )
    
    portfolio_manager = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="portfolio_manager",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=portfolio_manager_persona
    )
    
    # Investment Analyst
    investment_analyst_persona = Persona(
        role="Investment Analyst",
        persona="I am an investment analyst specializing in security analysis and market research. I conduct in-depth research on investment opportunities to identify attractive investments.",
        objectives=[
            "Research and analyze investment opportunities",
            "Evaluate securities based on fundamental and technical factors",
            "Monitor market trends and economic indicators",
            "Generate investment ideas and recommendations"
        ],
        communication_style="Detail-oriented and research-focused",
        skills=[
            "Financial analysis",
            "Market research",
            "Valuation modeling",
            "Investment recommendation"
        ]
    )
    
    investment_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="investment_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=investment_analyst_persona
    )
    
    # Define chat environment for team discussions
    chat_env = {
        "name": "investment_team_chat",
        "mechanism": "chat",
        "form_cohorts": False,
        "sub_rounds": 3,
        "group_size": 3,
        "task_prompt": "Collaborate to develop investment strategies and recommendations."
    }
    
    # Create the investment team with both MCP and chat environments
    investment_team = MarketAgentTeam(
        name="Investment Advisory Team",
        manager=financial_advisor,
        agents=[
            portfolio_manager,
            investment_analyst
        ],
        mode="collaborative",
        use_group_chat=True,
        shared_context={
            "investment_philosophy": "Long-term, value-oriented approach with focus on quality and risk management",
            "client_objectives": ["Capital preservation", "Income generation", "Long-term growth"],
            "market_outlook": "Moderate growth with elevated volatility due to economic uncertainties"
        },
        environments=[
            mcp_finance_env,
            chat_env
        ]
    )
    
    print(f"Created Investment Team with MCP: {investment_team.name}")
    print(f"Team members: {[agent.id for agent in investment_team.agents]}")
    print(f"Team manager: {investment_team.manager.id}")
    print(f"Team environments: {[env['name'] for env in investment_team.environments]}")
    
    return investment_team

async def run_team_investment_analysis(team, client_scenario):
    """Run an investment analysis with the team using both MCP and chat environments."""
    
    task = f"""
    As an investment advisory team, analyze the following client scenario and develop a comprehensive investment plan:
    
    CLIENT SCENARIO:
    {client_scenario}
    
    Your team should:
    
    1. The Financial Advisor should:
       - Lead the discussion and understand the client's needs
       - Coordinate the team's analysis and recommendations
       - Ensure the final plan aligns with client objectives
       - Provide holistic financial planning advice
    
    2. The Portfolio Manager should:
       - Develop an appropriate asset allocation strategy
       - Recommend specific portfolio construction approaches
       - Address portfolio risk management considerations
       - Suggest implementation and monitoring strategies
    
    3. The Investment Analyst should:
       - Research specific investment opportunities
       - Analyze market sectors and trends
       - Evaluate potential investments using Finance MCP tools
       - Provide data-driven investment recommendations
    
    Use the Finance MCP environment to access financial data and analysis tools, and the chat environment to collaborate and develop your recommendations.
    
    The final output should be a comprehensive investment plan that includes:
    - Executive summary of recommendations
    - Client goals and constraints analysis
    - Strategic asset allocation
    - Specific investment recommendations with rationale
    - Risk management strategy
    - Implementation and monitoring plan
    """
    
    print(f"\nStarting team investment analysis for client scenario: {client_scenario[:100]}...")
    print("-" * 80)
    
    # Run the team analysis using both environments
    result = await team.execute(task)
    
    print("\nTeam Investment Analysis Complete")
    print("-" * 80)
    print(result)
    
    return result

async def main():
    # Create a financial advisor with MCP environment
    advisor = await create_financial_advisor()
    await setup_finance_mcp_environment(advisor)
    
    # Run individual financial analyses
    queries = [
        "Analyze Apple (AAPL) stock and provide an investment recommendation based on current financials and market trends.",
        "Evaluate a balanced portfolio allocation for a moderate-risk investor nearing retirement in 5 years with $500,000 to invest.",
        "Analyze current economic indicators and their implications for different asset classes over the next 12 months."
    ]
    
    for query in queries:
        await run_financial_analysis(advisor, query)
    
    # Create and run team analysis with MCP
    team = await create_investment_team_with_mcp()
    
    client_scenarios = [
        """
        Client Profile: John and Mary Smith, both 55 years old, planning to retire in 10 years.
        Financial Situation:
        - Combined annual income: $200,000
        - Current retirement savings: $800,000 (60% in 401(k), 30% in IRAs, 10% in taxable accounts)
        - Home value: $600,000 with $200,000 remaining on mortgage
        - Monthly expenses: $8,000
        - Risk tolerance: Moderate
        Goals:
        - Retire at 65 with $2 million in savings
        - Generate $100,000 annual retirement income
        - Fund college education for one grandchild ($150,000 in 8 years)
        - Leave a legacy for children and charitable causes
        Concerns:
        - Market volatility affecting retirement timeline
        - Inflation eroding purchasing power
        - Healthcare costs in retirement
        - Tax efficiency of investments
        """
    ]
    
    for scenario in client_scenarios:
        await run_team_investment_analysis(team, scenario)

if __name__ == "__main__":
    asyncio.run(main())
