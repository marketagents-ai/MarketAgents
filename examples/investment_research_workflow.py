from typing import Dict, Any
from market_agents.workflows.workflow_utils import setup_mcp_environment
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import load_config_from_yaml
from market_agents.workflows.market_agent_workflow import Workflow, WorkflowStep
from minference.lite.models import (
    CallableMCPTool,
    LLMConfig, 
    ResponseFormat,
)
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironment, MCPServerEnvironmentConfig, MCPServerMechanism

async def create_research_workflow() -> Workflow:
    """Create an investment research workflow using MCP Finance tools."""
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug("Starting create_research_workflow")

    # Create environment config
    config = MCPServerEnvironmentConfig(
        name="mcp_finance",
        mechanism="mcp_server",
        mcp_server_module="market_agents.orchestrators.mcp_server.finance_mcp_server",
        mcp_server_class="mcp",
        form_cohorts=False,
        sub_rounds=1,
        group_size=1,
        task_prompt=""
    )
    logger.debug(f"Created config: {config}")

    # Set up the environment
    logger.debug("Setting up MCP environment...")
    finance_mcp = await setup_mcp_environment(config)
    logger.debug("MCP environment setup complete")

    # Debug log the action space
    if finance_mcp.action_space:
        logger.debug(f"Action space has {len(finance_mcp.action_space.allowed_actions)} tools")
    else:
        logger.error("No action space found!")

    # Create tool dictionary
    tool_dict = {tool.name: tool for tool in finance_mcp.action_space.allowed_actions}
    logger.debug(f"Available tools: {list(tool_dict.keys())}")

    # Get specific tools for each step based on actually available tools
    fundamentals_tool = tool_dict["get_stock_fundamentals"]
    financials_tool = tool_dict["get_financial_statements"]
    technical_indicators_tool = tool_dict["get_technical_indicators"]
    key_ratios_tool = tool_dict["get_key_financial_ratios"]

    # Create workflow steps with available tools
    fundamental_analysis_step = WorkflowStep(
        name="fundamental_analysis",
        description="Analyze company fundamentals and financials",
        environment_name="mcp_finance",
        tools=[fundamentals_tool, financials_tool, key_ratios_tool],
        instruction_prompt="""
        Analyze the fundamental data for {ticker}:
        1. Review key financial metrics and ratios
        2. Assess financial statements and growth trends
        3. Evaluate competitive position and market share
        
        Use the available tools to gather and analyze the data.
        """,
        run_full_episode=True
    )

    technical_analysis_step = WorkflowStep(
        name="technical_analysis",
        description="Analyze price action and technical indicators",
        environment_name="mcp_finance",
        tools=[technical_indicators_tool],
        instruction_prompt="""
        Perform technical analysis for {ticker}:
        1. Analyze price trends and patterns
        2. Review key technical indicators
        3. Identify support/resistance levels
        
        Previous fundamental analysis results: {fundamental_analysis_result}
        """,
        run_full_episode=False
    )

    # Get additional analysis tools
    analyst_recs_tool = tool_dict["get_analyst_recommendations"]
    company_profile_tool = tool_dict["get_company_profile"]
    company_news_tool = tool_dict["get_company_news"]

    valuation_step = WorkflowStep(
        name="valuation_analysis",
        description="Determine valuation and price targets",
        environment_name="mcp_finance",
        tools=[analyst_recs_tool, company_profile_tool, company_news_tool],
        instruction_prompt="""
        Develop valuation analysis for {ticker}:
        1. Review analyst recommendations and price targets
        2. Analyze company profile and recent news
        3. Synthesize findings into investment recommendation
        
        Consider:
        - Fundamental Analysis: {fundamental_analysis_result}
        - Technical Analysis: {technical_analysis_result}
        """,
        run_full_episode=False
    )

    
    research_workflow = Workflow.create(
        name="investment_research",
        description="Comprehensive investment research workflow",
        steps=[
            fundamental_analysis_step,
            technical_analysis_step,
            valuation_step
        ],
        mcp_servers={"mcp_finance": finance_mcp}
    )
       # Verify the workflow has the environment
    logger.debug(f"Workflow created with environments: {list(research_workflow.mcp_servers.keys())}")
    logger.debug(f"MCP Finance environment in workflow: {research_workflow.mcp_servers.get('mcp_finance') is not None}")

    return research_workflow

async def create_research_agent() -> MarketAgent:
    """Create a research agent with appropriate persona and config."""
    
    # Load storage config
    storage_config = load_config_from_yaml("market_agents/memory/storage_config.yaml")
    storage_utils = AgentStorageAPIUtils(config=storage_config)

    # Create Research Analyst Persona
    research_analyst_persona = Persona(
        name="James Chen",
        role="Investment Research Analyst",
        persona="Experienced research analyst with comprehensive market knowledge",
        objectives=[
            "Conduct thorough investment research",
            "Analyze multiple data sources",
            "Generate actionable investment insights"
        ],
        trader_type=["Expert", "Rational", "Analytical"],
        communication_style="Formal",
        routines=[
            "Review financial data",
            "Analyze market trends",
            "Develop investment theses"
        ],
        skills=[
            "Financial analysis",
            "Technical analysis",
            "Valuation modeling",
            "Market research"
        ]
    )

    # Create the agent
    research_agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="research_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        ),
        persona=research_analyst_persona
    )

    return research_agent

async def run_investment_research(ticker: str) -> Dict[str, Any]:
    """Run a complete investment research workflow for a given ticker."""
    
    # Create agent and workflow
    agent = await create_research_agent()
    workflow = await create_research_workflow()
    
    # Execute workflow
    result = await workflow.execute(
        agent=agent,
        initial_inputs={"ticker": ticker}
    )
    
    return result

async def main():
    # Run research workflow for NVDA
    result = await run_investment_research("NVDA")
    print("Research Results:", result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())