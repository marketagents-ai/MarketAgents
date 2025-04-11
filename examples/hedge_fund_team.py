from market_agents.agents.market_agent import MarketAgent
from market_agents.orchestrators.market_agent_team import MarketAgentTeam
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import load_config_from_yaml
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.agents.personas.persona import Persona
from minference.lite.models import LLMConfig, ResponseFormat
from typing import Dict, Any

async def create_hedge_fund_team() -> MarketAgentTeam:
    """Create a hierarchical hedge fund team with specialized agents."""
    
    # Create Portfolio Manager Persona
    portfolio_manager_persona = Persona(
        role="Portfolio Manager",
        persona="Experienced investment professional with strong risk management background",
        objectives=[
            "Analyze team insights and make final investment decisions",
            "Manage portfolio risk and allocation",
            "Coordinate team analysis efforts"
        ],
        skills=[
            "Portfolio management",
            "Risk assessment",
            "Team leadership"
        ]
    )

    # Create Portfolio Manager
    portfolio_manager = await MarketAgent.create(
        name="portfolio_manager",
        persona=portfolio_manager_persona,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.3,
            use_cache=True
        ),
        econ_agent=EconomicAgent(
            generate_wallet=True,
            initial_holdings={"USDC": 10000000.0}
        )
    )

    # Create Fundamental Analyst Persona
    fundamental_analyst_persona = Persona(
        role="Fundamental Analysis Specialist",
        persona="Detail-oriented financial analyst focused on company fundamentals",
        objectives=[
            "Analyze financial statements and metrics",
            "Evaluate business models and competitive positions",
            "Assess company valuations and fair value estimates"
        ],
        skills=[
            "Financial analysis",
            "Valuation modeling",
            "Industry research"
        ]
    )

    # Create Fundamental Analyst
    fundamental_analyst = await MarketAgent.create(
        name="fundamental_analyst",
        persona=fundamental_analyst_persona,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        )
    )

    # Create Technical Analyst Persona
    technical_analyst_persona = Persona(
        role="Technical Analysis Specialist",
        persona="Experienced technical trader focused on price patterns",
        objectives=[
            "Analyze price trends and patterns",
            "Identify key support/resistance levels",
            "Generate trading signals based on technical indicators"
        ],
        skills=[
            "Technical analysis",
            "Pattern recognition",
            "Momentum trading"
        ]
    )

    # Create Technical Analyst
    technical_analyst = await MarketAgent.create(
        name="technical_analyst",
        persona=technical_analyst_persona,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        ),
    )

    # Create Macro Analyst Persona
    macro_analyst_persona = Persona(
        role="Macro Research Specialist",
        persona="Global macro analyst focused on economic trends",
        objectives=[
            "Monitor global economic indicators",
            "Analyze central bank policies and implications",
            "Assess geopolitical risks and market impacts"
        ],
        skills=[
            "Economic analysis",
            "Policy research",
            "Geopolitical assessment"
        ]
    )

    # Create Macro Analyst
    macro_analyst = await MarketAgent.create(
        name="macro_analyst",
        persona=macro_analyst_persona,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        ),
    )

    # Create Risk Analyst Persona
    risk_analyst_persona = Persona(
        role="Risk Management Specialist",
        persona="Risk-focused analyst specializing in portfolio risk assessment",
        objectives=[
            "Monitor portfolio risk metrics",
            "Analyze position sizing and leverage",
            "Assess market volatility and correlation risks"
        ],
        skills=[
            "Risk modeling",
            "Portfolio analysis",
            "Quantitative methods"
        ]
    )

    # Create Risk Analyst
    risk_analyst = await MarketAgent.create(
        name="risk_analyst",
        persona=risk_analyst_persona,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        )
    )

    # Define environment orchestrators
    mcp_finance = {
        "name": "mcp_finance",
        "mechanism": "mcp_server",
        "mcp_server_module": "market_agents.orchestrators.mcp_server.finance_mcp_server",
        "form_cohorts": False,
        "sub_rounds": 2,
        "group_size": 5
    }

    # Create the hedge fund team with environments
    hedge_fund_team = MarketAgentTeam(
        name="Quantum Hedge Fund",
        manager=portfolio_manager,
        agents=[
            fundamental_analyst,
            technical_analyst,
            macro_analyst,
            risk_analyst
        ],
        mode="hierarchical",
        use_group_chat=False,
        shared_context={
            "investment_coverage": {
                "focus_areas": ["Technology", "AI/ML", "Cloud Computing", "Digital Platforms", "Semiconductors"],
                "investment_thesis": "Focus on market-leading tech companies with strong moats, sustainable growth, and exposure to AI transformation",
                "selection_criteria": "Companies with robust cash flows, high R&D investment, and dominant market positions",
                "strategic_approach": "Identify companies benefiting from digital transformation and AI adoption trends"
            },
            "risk_management_strategy": {
                "position_sizing": "Maintain moderate position sizes, never exceeding one-fifth of portfolio for any single investment",
                "sector_diversification": "Ensure broad exposure across sectors while allowing strategic overweighting in high-conviction areas",
                "downside_protection": "Implement disciplined exit strategies when investments move against thesis beyond acceptable thresholds"
            },
            "portfolio_strategy": {
                "diversification_approach": "Maintain a focused but diversified portfolio of 5-15 high-conviction positions",
                "capital_efficiency": "Employ modest leverage selectively to enhance returns while preserving capital protection",
                "rebalancing_discipline": "Regularly reassess position weights to maintain alignment with risk tolerance and market conditions"
            }
        },
        environments=[ 
            mcp_finance,
        ]
    )

    return hedge_fund_team 

async def run_investment_analysis(team: MarketAgentTeam, ticker: str):
    """Run a comprehensive investment analysis using the hedge fund team."""
    
    task = f"""
    Conduct a comprehensive investment analysis for {ticker} to determine position sizing and timing.
    
    Required Analysis Components:
    1. Fundamental Analysis
       - Use MCP Finance tools to gather financial metrics and ratios
       - Analyze company financials and competitive position
       - Develop valuation models using available data
    
    2. Technical Analysis
       - Utilize MCP Finance for price data and technical indicators
       - Identify key chart patterns and levels
       - Analyze volume and momentum metrics
    
    3. Macro Context
       - Research economic indicators and sector trends
       - Analyze policy impacts using research tools
       - Evaluate market sentiment and sector correlations
    
    4. Risk Assessment
       - Calculate position risk metrics using MCP Finance
       - Analyze portfolio impact and correlations
       - Determine risk-adjusted position sizing
    
    Collaboration Guidelines:
    - Use group chat for coordinating analysis and sharing insights
    - Leverage MCP Finance tools for data-driven analysis
    - Access research environment for deeper context
    - Share findings and discuss implications
    
    The Portfolio Manager will synthesize all analyses and tools' outputs into a final investment decision
    including position size, entry timing, and risk parameters.
    """
    
    result = await team.execute(task)
    return result

async def main():
    # Create the hedge fund team
    team = await create_hedge_fund_team()
    
    # Run analysis for NVDA
    result = await run_investment_analysis(team, "NVDA")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())