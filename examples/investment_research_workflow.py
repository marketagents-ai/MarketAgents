from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from market_agents.workflows.workflow_utils import setup_mcp_environment
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import load_config_from_yaml
from market_agents.workflows.market_agent_workflow import Workflow, WorkflowStep, WorkflowStepIO
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironment, MCPServerEnvironmentConfig
from minference.lite.models import LLMConfig, ResponseFormat

# Define Pydantic models for our workflow step inputs
class CompanyInfo(BaseModel):
    ticker: str
    industry: str
    sector: Optional[str] = None

class IndustryMetrics(BaseModel):
    profitability: List[str] = Field(default_factory=list)
    asset_quality: List[str] = Field(default_factory=list)
    capital: List[str] = Field(default_factory=list)
    valuation: List[str] = Field(default_factory=list)

# Define company classifications
TICKER_CLASSIFICATIONS = {
    "JPM": {
        "industry": "Banks",
        "sector": "Financial Services"
    },
    "GS": {
        "industry": "Banks",
        "sector": "Financial Services"
    },
    "NVDA": {
        "industry": "Semiconductors",
        "sector": "Technology"
    },
    "AMD": {
        "industry": "Semiconductors",
        "sector": "Technology"
    }
}

# Industry-specific metrics remain the same
INDUSTRY_METRICS = {
    "Banks": IndustryMetrics(
        profitability=["NIM", "ROE", "Cost/Income Ratio"],
        asset_quality=["NPL Ratio", "Coverage Ratio"],
        capital=["CET1 Ratio", "Tier 1 Ratio"],
        valuation=["P/B Ratio", "P/E Ratio", "Dividend Yield"]
    ),
    "Semiconductors": IndustryMetrics(
        profitability=["Gross Margin", "R&D as % of Revenue"],
        asset_quality=["Inventory Turnover"],
        capital=["ROIC", "CapEx to Revenue"],
        valuation=["P/E Ratio", "EV/EBITDA", "PEG Ratio"]
    )
}

async def create_research_workflow() -> Workflow:
    """Create an investment research workflow using MCP Finance tools."""
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

    finance_mcp = await setup_mcp_environment(config)
    tool_dict = {tool.name: tool for tool in finance_mcp.action_space.allowed_actions}

    fundamentals_tool = tool_dict["get_stock_fundamentals"]
    financials_tool = tool_dict["get_financial_statements"]
    technical_indicators_tool = tool_dict["get_technical_indicators"]
    analyst_recs_tool = tool_dict["get_analyst_recommendations"]
    company_profile_tool = tool_dict["get_company_profile"]

    fundamental_analysis_step = WorkflowStep(
        name="fundamental_analysis",
        environment_name="mcp_finance",
        tools=[fundamentals_tool, financials_tool],
        subtask="""
        Analyze the company's fundamental data focusing on:
        1. Key financial metrics and ratios
        2. Financial statements and growth trends
        3. Competitive position and market share
        """,
        inputs=[
            WorkflowStepIO(
                name="company_info",
                data=CompanyInfo
            ),
            WorkflowStepIO(
                name="industry_metrics",
                data=IndustryMetrics
            )
        ],
        run_full_episode=False,
        sequential_tools=True
    )

    technical_analysis_step = WorkflowStep(
        name="technical_analysis",
        environment_name="mcp_finance",
        tools=[technical_indicators_tool],
        subtask="""
        Perform technical analysis focusing on:
        1. Analyze price trends and patterns
        2. Review key technical indicators
        3. Identify support/resistance levels
        """,
        inputs=[
            WorkflowStepIO(
                name="company_info",
                data=CompanyInfo
            )
        ],
        run_full_episode=False,
        sequential_tools=False
    )

    valuation_step = WorkflowStep(
        name="valuation_analysis",
        environment_name="mcp_finance",
        tools=[analyst_recs_tool, company_profile_tool],
        subtask="""
        Develop a comprehensive valuation analysis:
        1. Review analyst recommendations and price targets
        2. Assess current market valuation
        3. Provide valuation assessment and investment recommendation
        """,
        inputs=[
            WorkflowStepIO(
                name="company_info",
                data=CompanyInfo
            )
        ],
        run_full_episode=False,
        sequential_tools=True
    )

    research_workflow = Workflow.create(
        name="investment_research",
        task="Conduct comprehensive investment research for {ticker}.",
        steps=[
            fundamental_analysis_step,
            technical_analysis_step,
            valuation_step
        ],
        mcp_servers={"mcp_finance": finance_mcp}
    )

    return research_workflow

async def create_research_agent() -> MarketAgent:
    """Create a research agent with appropriate persona and config."""
    storage_config = load_config_from_yaml("market_agents/memory/storage_config.yaml")
    storage_utils = AgentStorageAPIUtils(config=storage_config)

    research_agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="research_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        )
    )

    return research_agent

async def run_investment_research(ticker: str) -> Dict[str, Any]:
    """Run investment research workflow for a given ticker."""
    if ticker not in TICKER_CLASSIFICATIONS:
        raise ValueError(f"Ticker {ticker} not found in supported classifications")
    
    classification = TICKER_CLASSIFICATIONS[ticker]
    
    company_info = CompanyInfo(
        ticker=ticker,
        industry=classification["industry"],
        sector=classification["sector"]
    )
    
    industry_metrics = INDUSTRY_METRICS[classification["industry"]]
    
    agent = await create_research_agent()
    workflow = await create_research_workflow()
    
    result = await workflow.execute(
        agent=agent,
        initial_inputs={
            "ticker": ticker,
            "company_info": company_info,
            "industry_metrics": industry_metrics
        }
    )
    
    return result

async def main():
    import json
    from uuid import UUID
    from datetime import datetime

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, UUID):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    # Simple example running analysis for a single ticker
    result = await run_investment_research("NVDA")
    result_dict = result.model_dump()
    print(f"\nResearch Results for NVDA:", 
        json.dumps(result_dict, indent=2, cls=CustomJSONEncoder))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())