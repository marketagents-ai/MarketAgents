from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from market_agents.workflows.workflow_utils import setup_mcp_environment
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.workflows.market_agent_workflow import Workflow, WorkflowStep, WorkflowStepIO
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironmentConfig
from minference.lite.models import LLMConfig, ResponseFormat, StructuredTool

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

class InvestmentAnalysisSummary(BaseModel):
    """Schema for investment analysis summary based on fundamental, technical, and valuation analysis"""
    ticker: str = Field(
        ..., 
        description="Stock ticker symbol"
    )
    fundamental_metrics: Dict[str, str] = Field(
        default_factory=dict,
        description="Key financial metrics and their interpretations"
    )
    growth_trends: List[str] = Field(
        default_factory=list,
        description="Key growth trends from financial statements"
    )
    technical_signals: Dict[str, str] = Field(
        default_factory=dict,
        description="Technical indicator signals and interpretations"
    )
    analyst_consensus: str = Field(
        ...,
        description="Overall analyst consensus rating"
    )
    investment_recommendation: str = Field(
        ...,
        description="Final investment recommendation with rationale"
    )

async def create_research_workflow() -> Workflow:
    """Create an investment research workflow using MCP Finance tools."""
    config = MCPServerEnvironmentConfig(
        name="mcp_finance",
        mechanism="mcp_server",
        mcp_server_module="market_agents.orchestrators.mcp_server.finance_mcp_server"
    )

    finance_mcp = await setup_mcp_environment(config)
    tools = finance_mcp.get_tools()

    fundamentals_tool = tools["get_stock_fundamentals"]
    financials_tool = tools["get_financial_statements"]
    technical_indicators_tool = tools["get_technical_indicators"]
    analyst_recs_tool = tools["get_analyst_recommendations"]
    company_profile_tool = tools["get_company_profile"]

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

    investment_analysis_tool = StructuredTool.from_pydantic(
        model=InvestmentAnalysisSummary,
        name="investment_analysis_summary",
        description="Generate a structured investment analysis summary with fundamental, technical, and valuation insights"
    )

     # Add new summary step
    summary_step = WorkflowStep(
        name="analysis_summary",
        environment_name="mcp_finance",
        tools=[investment_analysis_tool],
        subtask="""
        Based on the previous analysis steps, generate a comprehensive stock analysis summary including:
        1. Overall stock rating and rationale
        2. Target price assessment
        3. Market sentiment analysis
        4. Key qualitative catalysts (do not repeat KPIs)
        5. Key quantitative KPIs (do not repeat catalysts)
        6. Portfolio action recommendation with reasoning
        
        Format the output using the stock_analysis_summary tool structure.
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

    research_workflow = Workflow.create(
        name="investment_research",
        task="Conduct comprehensive investment research for {ticker}.",
        steps=[
            fundamental_analysis_step,
            technical_analysis_step,
            valuation_step,
            summary_step
        ],
        mcp_servers={"mcp_finance": finance_mcp}
    )

    return research_workflow

async def create_analyst_agent() -> MarketAgent:
    """Create a research agent with appropriate persona and config."""

        # Create investment analyst persona
    investment_analyst_persona = Persona(
        role="Investment Analyst",
        persona="You are an investment analyst specializing in security analysis and market research. I conduct in-depth research on investment opportunities to identify attractive investments.",
        objectives=[
            "Research and analyze investment opportunities",
            "Evaluate securities based on fundamental and technical factors",
            "Monitor market trends and economic indicators",
            "Generate investment ideas and recommendations"
        ],
        skills=[
            "Fundamental Analysis", "Technical Analysis", "Risk Assessment", 
            "Valuation Methods", "Industry Analysis"
        ]
    )

    analyst_agent = await MarketAgent.create(
        name="investment_analyst",
        persona=investment_analyst_persona,
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            client="openai",
            response_format=ResponseFormat.auto_tools,
            temperature=0.2,
            use_cache=True
        )
    )

    return analyst_agent

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
    
    agent = await create_analyst_agent()
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