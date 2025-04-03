import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.tools.callable_tool import CallableTool
from market_agents.tools.structured_tool import StructuredTool
from minference.lite.models import LLMConfig, ResponseFormat

# Define structured output models
class StockAnalysis(BaseModel):
    """Analysis of a stock's fundamentals and technical indicators."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    sector: str = Field(..., description="Industry sector")
    current_price: float = Field(..., description="Current stock price")
    
    # Fundamental metrics
    pe_ratio: Optional[float] = Field(None, description="Price to earnings ratio")
    market_cap: Optional[float] = Field(None, description="Market capitalization in billions")
    revenue_growth: Optional[float] = Field(None, description="Year-over-year revenue growth percentage")
    profit_margin: Optional[float] = Field(None, description="Net profit margin percentage")
    debt_to_equity: Optional[float] = Field(None, description="Debt to equity ratio")
    
    # Technical indicators
    moving_avg_50day: Optional[float] = Field(None, description="50-day moving average")
    moving_avg_200day: Optional[float] = Field(None, description="200-day moving average")
    rsi_14day: Optional[float] = Field(None, description="14-day relative strength index")
    
    # Analysis and recommendation
    fundamental_analysis: str = Field(..., description="Analysis of company fundamentals")
    technical_analysis: str = Field(..., description="Analysis of technical indicators")
    recommendation: str = Field(..., description="Investment recommendation (Buy, Hold, Sell)")
    target_price: Optional[float] = Field(None, description="12-month price target")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")
    
    # Investment thesis
    bull_case: str = Field(..., description="Bull case scenario")
    bear_case: str = Field(..., description="Bear case scenario")
    catalysts: List[str] = Field(..., description="Potential catalysts for stock movement")
    risks: List[str] = Field(..., description="Key risks to watch")

class PortfolioRecommendation(BaseModel):
    """Portfolio allocation recommendation based on risk profile and market conditions."""
    
    risk_profile: str = Field(..., description="Client risk profile (Conservative, Moderate, Aggressive)")
    investment_horizon: str = Field(..., description="Investment time horizon (Short-term, Medium-term, Long-term)")
    
    # Asset allocation
    allocation: Dict[str, float] = Field(..., description="Recommended asset allocation percentages")
    
    # Sector weights
    sector_weights: Dict[str, float] = Field(..., description="Recommended sector weightings")
    
    # Specific recommendations
    recommended_investments: List[Dict[str, Any]] = Field(..., description="Specific investment recommendations")
    
    # Rationale and strategy
    market_outlook: str = Field(..., description="Current market outlook")
    strategy_rationale: str = Field(..., description="Rationale for recommended strategy")
    rebalancing_frequency: str = Field(..., description="Recommended portfolio rebalancing frequency")
    
    # Performance expectations
    expected_return: float = Field(..., description="Expected annual return percentage")
    expected_volatility: float = Field(..., description="Expected volatility/standard deviation")
    worst_case_scenario: str = Field(..., description="Worst case scenario description and impact")

# Define callable tools
def get_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Retrieve current stock data for the specified ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing stock data
    """
    # In a real implementation, this would call an API or database
    # For this example, we'll return mock data
    
    stock_data = {
        "AAPL": {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "current_price": 187.32,
            "pe_ratio": 30.8,
            "market_cap": 2950.5,
            "revenue_growth": 7.2,
            "profit_margin": 25.3,
            "debt_to_equity": 1.8,
            "moving_avg_50day": 182.45,
            "moving_avg_200day": 175.20,
            "rsi_14day": 62.5
        },
        "MSFT": {
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "sector": "Technology",
            "current_price": 415.75,
            "pe_ratio": 35.2,
            "market_cap": 3100.2,
            "revenue_growth": 15.8,
            "profit_margin": 33.5,
            "debt_to_equity": 0.5,
            "moving_avg_50day": 410.30,
            "moving_avg_200day": 385.15,
            "rsi_14day": 58.2
        },
        "AMZN": {
            "ticker": "AMZN",
            "company_name": "Amazon.com Inc.",
            "sector": "Consumer Cyclical",
            "current_price": 178.25,
            "pe_ratio": 42.5,
            "market_cap": 1850.3,
            "revenue_growth": 12.3,
            "profit_margin": 7.8,
            "debt_to_equity": 1.2,
            "moving_avg_50day": 175.40,
            "moving_avg_200day": 165.80,
            "rsi_14day": 55.7
        }
    }
    
    if ticker in stock_data:
        return stock_data[ticker]
    else:
        return {
            "ticker": ticker,
            "company_name": f"Unknown Company ({ticker})",
            "sector": "Unknown",
            "current_price": 0.0,
            "error": f"No data available for ticker {ticker}"
        }

def get_market_data() -> Dict[str, Any]:
    """
    Retrieve current market data and indicators.
    
    Returns:
        Dictionary containing market data
    """
    # In a real implementation, this would call an API or database
    # For this example, we'll return mock data
    
    return {
        "indices": {
            "S&P500": {
                "value": 5320.45,
                "change_percent": 0.35,
                "ytd_return": 11.8
            },
            "NASDAQ": {
                "value": 16750.25,
                "change_percent": 0.42,
                "ytd_return": 14.2
            },
            "DOW": {
                "value": 38950.75,
                "change_percent": 0.28,
                "ytd_return": 8.5
            }
        },
        "sectors": {
            "Technology": {
                "performance_ytd": 15.8,
                "outlook": "Positive"
            },
            "Healthcare": {
                "performance_ytd": 7.2,
                "outlook": "Neutral"
            },
            "Financials": {
                "performance_ytd": 9.5,
                "outlook": "Positive"
            },
            "Energy": {
                "performance_ytd": -3.2,
                "outlook": "Negative"
            },
            "Consumer Staples": {
                "performance_ytd": 4.8,
                "outlook": "Neutral"
            }
        },
        "economic_indicators": {
            "inflation_rate": 3.2,
            "unemployment_rate": 3.8,
            "fed_funds_rate": 5.25,
            "gdp_growth": 2.4,
            "consumer_sentiment": 78.5
        },
        "market_sentiment": {
            "fear_greed_index": 65,  # 0-100 scale, higher means more greedy
            "vix": 18.5,
            "put_call_ratio": 0.85
        }
    }

def calculate_portfolio_metrics(allocation: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate expected portfolio metrics based on the provided asset allocation.
    
    Args:
        allocation: Dictionary mapping asset classes to allocation percentages
        
    Returns:
        Dictionary containing calculated portfolio metrics
    """
    # In a real implementation, this would use historical data and models
    # For this example, we'll use simplified calculations with mock data
    
    # Mock expected returns and volatilities for asset classes
    asset_metrics = {
        "US_Stocks": {"return": 8.5, "volatility": 15.0, "correlation": {}},
        "International_Stocks": {"return": 7.8, "volatility": 17.0, "correlation": {}},
        "Bonds": {"return": 4.2, "volatility": 5.5, "correlation": {}},
        "Real_Estate": {"return": 6.5, "volatility": 12.0, "correlation": {}},
        "Commodities": {"return": 5.0, "volatility": 18.0, "correlation": {}},
        "Cash": {"return": 3.0, "volatility": 0.5, "correlation": {}}
    }
    
    # Correlation matrix (simplified)
    asset_metrics["US_Stocks"]["correlation"] = {
        "US_Stocks": 1.0, "International_Stocks": 0.8, "Bonds": -0.2, 
        "Real_Estate": 0.5, "Commodities": 0.3, "Cash": -0.1
    }
    asset_metrics["International_Stocks"]["correlation"] = {
        "US_Stocks": 0.8, "International_Stocks": 1.0, "Bonds": -0.1, 
        "Real_Estate": 0.4, "Commodities": 0.4, "Cash": -0.1
    }
    asset_metrics["Bonds"]["correlation"] = {
        "US_Stocks": -0.2, "International_Stocks": -0.1, "Bonds": 1.0, 
        "Real_Estate": 0.2, "Commodities": 0.0, "Cash": 0.3
    }
    asset_metrics["Real_Estate"]["correlation"] = {
        "US_Stocks": 0.5, "International_Stocks": 0.4, "Bonds": 0.2, 
        "Real_Estate": 1.0, "Commodities": 0.3, "Cash": -0.1
    }
    asset_metrics["Commodities"]["correlation"] = {
        "US_Stocks": 0.3, "International_Stocks": 0.4, "Bonds": 0.0, 
        "Real_Estate": 0.3, "Commodities": 1.0, "Cash": -0.2
    }
    asset_metrics["Cash"]["correlation"] = {
        "US_Stocks": -0.1, "International_Stocks": -0.1, "Bonds": 0.3, 
        "Real_Estate": -0.1, "Commodities": -0.2, "Cash": 1.0
    }
    
    # Calculate expected return (weighted average)
    expected_return = sum(allocation.get(asset, 0) * asset_metrics[asset]["return"] / 100 
                         for asset in allocation if asset in asset_metrics)
    
    # Calculate portfolio variance (including correlations)
    portfolio_variance = 0
    for asset1 in allocation:
        if asset1 not in asset_metrics:
            continue
        for asset2 in allocation:
            if asset2 not in asset_metrics:
                continue
            weight1 = allocation[asset1] / 100
            weight2 = allocation[asset2] / 100
            vol1 = asset_metrics[asset1]["volatility"]
            vol2 = asset_metrics[asset2]["volatility"]
            corr = asset_metrics[asset1]["correlation"][asset2]
            portfolio_variance += weight1 * weight2 * vol1 * vol2 * corr
    
    # Calculate portfolio volatility
    portfolio_volatility = portfolio_variance ** 0.5
    
    # Calculate Sharpe ratio (assuming risk-free rate of 3%)
    risk_free_rate = 3.0
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # Calculate max drawdown (simplified estimate based on volatility)
    estimated_max_drawdown = portfolio_volatility * 1.5
    
    return {
        "expected_return": round(expected_return, 2),
        "expected_volatility": round(portfolio_volatility, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "estimated_max_drawdown": round(estimated_max_drawdown, 2),
        "risk_level": "High" if portfolio_volatility > 12 else "Medium" if portfolio_volatility > 7 else "Low",
        "diversification_score": round(100 - (sum((allocation.get(asset, 0)/100)**2 for asset in allocation) * 100), 2)
    }

async def create_financial_analyst_with_tools():
    """Create a financial analyst agent with structured and callable tools."""
    
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
        persona="I am a financial analyst specializing in investment analysis and portfolio management. I provide data-driven insights and recommendations to help clients make informed investment decisions.",
        objectives=[
            "Analyze investment opportunities using fundamental and technical analysis",
            "Develop portfolio strategies aligned with client objectives and risk profiles",
            "Monitor market conditions and economic indicators",
            "Provide actionable investment recommendations"
        ],
        communication_style="Clear, precise, and data-driven",
        skills=[
            "Investment analysis",
            "Portfolio construction",
            "Risk assessment",
            "Financial modeling"
        ]
    )
    
    # Create callable tools
    stock_data_tool = CallableTool(
        name="get_stock_data",
        description="Retrieve current stock data for a specific ticker",
        function=get_stock_data,
        parameters={"ticker": "str"}
    )
    
    market_data_tool = CallableTool(
        name="get_market_data",
        description="Retrieve current market data and indicators",
        function=get_market_data,
        parameters={}
    )
    
    portfolio_metrics_tool = CallableTool(
        name="calculate_portfolio_metrics",
        description="Calculate expected portfolio metrics based on asset allocation",
        function=calculate_portfolio_metrics,
        parameters={"allocation": "Dict[str, float]"}
    )
    
    # Create structured tools
    stock_analysis_tool = StructuredTool(
        name="analyze_stock",
        description="Analyze a stock and provide investment recommendation",
        output_schema=StockAnalysis
    )
    
    portfolio_recommendation_tool = StructuredTool(
        name="recommend_portfolio",
        description="Generate portfolio allocation recommendation based on client profile",
        output_schema=PortfolioRecommendation
    )
    
    # Create agent with tools
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="financial_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7
        ),
        persona=persona,
        tools=[
            stock_data_tool,
            market_data_tool,
            portfolio_metrics_tool,
            stock_analysis_tool,
            portfolio_recommendation_tool
        ]
    )
    
    print(f"Created Financial Analyst with ID: {agent.id}")
    print(f"Available tools: {[tool.name for tool in agent.tools]}")
    
    return agent

async def analyze_stock(agent, ticker):
    """Use the agent to analyze a stock using structured output."""
    
    print(f"\nAnalyzing stock: {ticker}")
    print("-" * 80)
    
    # Create prompt for stock analysis
    prompt = f"""
    You are {agent.role}. {agent.persona}
    
    Analyze the stock with ticker symbol {ticker} and provide a comprehensive investment recommendation.
    
    First, use the get_stock_data tool to retrieve current data for {ticker}.
    Then, use the get_market_data tool to understand the broader market context.
    Finally, use the analyze_stock structured tool to provide your analysis and recommendation.
    
    Your analysis should include:
    - Fundamental analysis of the company's financial health and growth prospects
    - Technical analysis of price trends and momentum
    - Clear investment recommendation (Buy, Hold, or Sell)
    - Risk assessment and potential catalysts
    - Bull and bear case scenarios
    
    Provide a comprehensive and well-structured analysis.
    """
    
    # Generate response using the agent's LLM
    response = await agent.llm_orchestrator.generate(
        model=agent.llm_config.model,
        messages=[{"role": "system", "content": prompt}],
        tools=agent.tools_for_llm
    )
    
    # Process and extract the structured output
    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
    
    for tool_call in tool_calls:
        if tool_call.get('name') == 'analyze_stock':
            try:
                analysis_data = json.loads(tool_call.get('arguments', '{}'))
                analysis = StockAnalysis(**analysis_data)
                
                # Print the structured analysis
                print(f"Stock Analysis for {analysis.ticker} ({analysis.company_name})")
                print(f"Sector: {analysis.sector}")
                print(f"Current Price: ${analysis.current_price}")
                print(f"Recommendation: {analysis.recommendation}")
                if analysis.target_price:
                    print(f"Target Price: ${analysis.target_price}")
                print(f"Risk Level: {analysis.risk_level}")
                
                print("\nFundamental Analysis:")
                print(analysis.fundamental_analysis)
                
                print("\nTechnical Analysis:")
                print(analysis.technical_analysis)
                
                print("\nBull Case:")
                print(analysis.bull_case)
                
                print("\nBear Case:")
                print(analysis.bear_case)
                
                print("\nCatalysts:")
                for i, catalyst in enumerate(analysis.catalysts, 1):
                    print(f"{i}. {catalyst}")
                
                print("\nRisks:")
                for i, risk in enumerate(analysis.risks, 1):
                    print(f"{i}. {risk}")
                
                return analysis
            except Exception as e:
                print(f"Error parsing stock analysis: {str(e)}")
    
    print("No structured stock analysis found in the response.")
    return None

async def recommend_portfolio(agent, risk_profile, investment_horizon):
    """Use the agent to generate a portfolio recommendation using structured output."""
    
    print(f"\nGenerating portfolio recommendation for {risk_profile} investor with {investment_horizon} horizon")
    print("-" * 80)
    
    # Create prompt for portfolio recommendation
    prompt = f"""
    You are {agent.role}. {agent.persona}
    
    Generate a comprehensive portfolio recommendation for a {risk_profile} investor with a {investment_horizon} investment horizon.
    
    First, use the get_market_data tool to understand the current market conditions.
    Then, determine an appropriate asset allocation based on the client's risk profile and investment horizon.
    Use the calculate_portfolio_metrics tool to evaluate your proposed allocation.
    Finally, use the recommend_portfolio structured tool to provide your comprehensive recommendation.
    
    Your recommendation should include:
    - Asset allocation percentages across major asset classes
    - Sector weightings within equities
    - Specific investment recommendations (funds, ETFs, or individual securities)
    - Rationale for the recommended strategy
    - Expected performance metrics and risk assessment
    
    Provide a comprehensive and well-structured recommendation.
    """
    
    # Generate response using the agent's LLM
    response = await agent.llm_orchestrator.generate(
        model=agent.llm_config.model,
        messages=[{"role": "system", "content": prompt}],
        tools=agent.tools_for_llm
    )
    
    # Process and extract the structured output
    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
    
    for tool_call in tool_calls:
        if tool_call.get('name') == 'recommend_portfolio':
            try:
                recommendation_data = json.loads(tool_call.get('arguments', '{}'))
                recommendation = PortfolioRecommendation(**recommendation_data)
                
                # Print the structured recommendation
                print(f"Portfolio Recommendation for {recommendation.risk_profile} investor with {recommendation.investment_horizon} horizon")
                
                print("\nAsset Allocation:")
                for asset, allocation in recommendation.allocation.items():
                    print(f"{asset}: {allocation}%")
                
                print("\nSector Weights:")
                for sector, weight in recommendation.sector_weights.items():
                    print(f"{sector}: {weight}%")
                
                print("\nRecommended Investments:")
                for i, investment in enumerate(recommendation.recommended_investments, 1):
                    print(f"{i}. {investment.get('name', 'Unknown')} - {investment.get('allocation', 0)}% - {investment.get('rationale', '')}")
                
                print("\nMarket Outlook:")
                print(recommendation.market_outlook)
                
                print("\nStrategy Rationale:")
                print(recommendation.strategy_rationale)
                
                print("\nPerformance Expectations:")
                print(f"Expected Return: {recommendation.expected_return}%")
                print(f"Expected Volatility: {recommendation.expected_volatility}%")
                print(f"Rebalancing Frequency: {recommendation.rebalancing_frequency}")
                
                print("\nWorst Case Scenario:")
                print(recommendation.worst_case_scenario)
                
                return recommendation
            except Exception as e:
                print(f"Error parsing portfolio recommendation: {str(e)}")
    
    print("No structured portfolio recommendation found in the response.")
    return None

async def main():
    # Create a financial analyst with tools
    agent = await create_financial_analyst_with_tools()
    
    # Analyze stocks
    tickers = ["AAPL", "MSFT", "AMZN"]
    for ticker in tickers:
        await analyze_stock(agent, ticker)
    
    # Generate portfolio recommendations
    scenarios = [
        ("Conservative", "Short-term"),
        ("Moderate", "Medium-term"),
        ("Aggressive", "Long-term")
    ]
    
    for risk_profile, investment_horizon in scenarios:
        await recommend_portfolio(agent, risk_profile, investment_horizon)

if __name__ == "__main__":
    import json
    asyncio.run(main())
