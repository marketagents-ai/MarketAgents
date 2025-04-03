import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.market_agent_team import MarketAgentTeam
from market_agents.environments.mechanisms.chat import ChatEnvironment
from market_agents.environments.mechanisms.research import ResearchEnvironment
from minference.lite.models import LLMConfig, ResponseFormat

# Define industry sectors for market survey
INDUSTRY_SECTORS = [
    "Technology",
    "Healthcare",
    "Financial Services",
    "Consumer Goods",
    "Energy",
    "Telecommunications",
    "Manufacturing",
    "Retail"
]

# Define survey questions template
SURVEY_QUESTIONS_TEMPLATE = """
# Market Survey: {sector} Industry

## Industry Trends
1. What are the 3-5 most significant trends currently shaping the {sector} industry?
2. Which emerging technologies are having the greatest impact on {sector} companies?
3. How are customer/client expectations evolving in the {sector} space?

## Competitive Landscape
1. Who are the current market leaders in the {sector} industry?
2. Which companies are the most innovative disruptors in this space?
3. What competitive advantages separate market leaders from followers?

## Growth Opportunities
1. Which segments within the {sector} industry have the highest growth potential?
2. What unmet customer needs represent the biggest opportunities?
3. Which geographic markets offer the best expansion opportunities?

## Challenges and Risks
1. What are the most significant challenges facing {sector} companies today?
2. Which regulatory issues are most impactful for this industry?
3. What potential disruptions could significantly alter the competitive landscape?

## Future Outlook
1. How do you expect the {sector} industry to evolve over the next 3-5 years?
2. Which companies are best positioned for future success?
3. What capabilities will be most critical for companies to develop?
"""

async def create_industry_analyst(sector):
    """Create an industry analyst agent specialized in a specific sector."""
    
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
        role=f"{sector} Industry Analyst",
        persona=f"I am an industry analyst specializing in the {sector} sector. I track industry trends, competitive dynamics, and market opportunities to provide strategic insights and recommendations.",
        objectives=[
            f"Analyze {sector} industry trends and market dynamics",
            "Evaluate competitive positioning and strategies",
            "Identify growth opportunities and emerging threats",
            "Provide data-driven insights and recommendations"
        ],
        communication_style="Analytical, insightful, and evidence-based",
        skills=[
            "Industry analysis",
            "Competitive intelligence",
            "Market forecasting",
            "Strategic recommendation"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id=f"{sector.lower()}_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created {sector} Industry Analyst with ID: {agent.id}")
    
    return agent

async def create_research_environment():
    """Create a research environment for market surveys."""
    
    research_env = {
        "name": "market_survey_research",
        "mechanism": "research",
        "knowledge_base_name": "industry_research_kb",
        "research_tools": ["web_search", "data_analysis", "trend_analysis"],
        "research_depth": "comprehensive",
        "output_format": "structured_report"
    }
    
    return research_env

async def create_market_survey_swarm():
    """Create a swarm of industry analyst agents for parallel market surveys."""
    
    # Create a research environment
    research_env = await create_research_environment()
    
    # Create a team for each industry sector
    teams = []
    
    for sector in INDUSTRY_SECTORS:
        # Create the industry analyst
        analyst = await create_industry_analyst(sector)
        
        # Create a team with just this analyst (for parallel execution)
        team = MarketAgentTeam(
            name=f"{sector} Survey Team",
            manager=analyst,
            agents=[],  # No additional agents needed for this simple team
            mode="solo",  # Solo mode since there's just one agent
            use_group_chat=False,
            shared_context={
                "industry_sector": sector,
                "survey_focus": "Comprehensive industry analysis",
                "output_format": "Structured survey responses"
            },
            environments=[
                research_env
            ]
        )
        
        teams.append(team)
        print(f"Created {sector} Survey Team with analyst: {analyst.id}")
    
    # Create the swarm coordinator
    coordinator_storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    coordinator_storage_utils = AgentStorageAPIUtils(config=coordinator_storage_config)
    
    coordinator_persona = Persona(
        role="Market Survey Coordinator",
        persona="I am a market survey coordinator responsible for orchestrating industry research across multiple sectors. I design research methodologies, coordinate analyst activities, and synthesize cross-industry insights.",
        objectives=[
            "Design comprehensive market survey methodologies",
            "Coordinate parallel research across industry sectors",
            "Identify cross-industry patterns and insights",
            "Synthesize findings into actionable recommendations"
        ],
        communication_style="Clear, organized, and integrative",
        skills=[
            "Research coordination",
            "Cross-industry analysis",
            "Insight synthesis",
            "Strategic recommendation"
        ]
    )
    
    coordinator = await MarketAgent.create(
        storage_utils=coordinator_storage_utils,
        agent_id="survey_coordinator",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=coordinator_persona
    )
    
    print(f"Created Market Survey Coordinator with ID: {coordinator.id}")
    
    return {
        "coordinator": coordinator,
        "teams": teams,
        "sectors": INDUSTRY_SECTORS
    }

async def run_parallel_market_surveys(swarm):
    """Run parallel market surveys across multiple industry sectors."""
    
    coordinator = swarm["coordinator"]
    teams = swarm["teams"]
    sectors = swarm["sectors"]
    
    print(f"\nInitiating parallel market surveys across {len(sectors)} industry sectors")
    print("-" * 80)
    
    # Create survey tasks for each team
    survey_tasks = []
    
    for i, team in enumerate(teams):
        sector = sectors[i]
        survey_questions = SURVEY_QUESTIONS_TEMPLATE.format(sector=sector)
        
        task = f"""
        As a {sector} Industry Analyst, conduct a comprehensive market survey of the {sector} industry.
        
        Please respond to the following survey questions with detailed, data-driven insights:
        
        {survey_questions}
        
        Your response should:
        - Be based on your knowledge of the {sector} industry
        - Include specific examples and evidence
        - Provide nuanced analysis rather than general statements
        - Identify actionable insights for strategic decision-making
        
        Format your response as a structured survey report with clear sections for each question category.
        """
        
        # Create an async task for each survey
        survey_tasks.append(team.execute(task))
    
    # Run all surveys in parallel
    print(f"Running {len(survey_tasks)} market surveys in parallel...")
    survey_results = await asyncio.gather(*survey_tasks)
    
    print(f"Completed {len(survey_results)} parallel market surveys")
    
    # Create a task for the coordinator to synthesize the results
    synthesis_task = f"""
    As the Market Survey Coordinator, synthesize the findings from the following industry surveys into a comprehensive cross-industry analysis.
    
    The surveys cover the following industries:
    {', '.join(sectors)}
    
    For each industry, here are the detailed survey results:
    
    {'='*50}
    
    {chr(10).join([f"## {sectors[i]} INDUSTRY SURVEY RESULTS {chr(10)}{survey_results[i]}{chr(10)}{'='*50}{chr(10)}" for i in range(len(sectors))])}
    
    Your synthesis should:
    
    1. Identify common trends and patterns across industries
    2. Highlight significant differences between industries
    3. Analyze cross-industry implications and opportunities
    4. Provide strategic recommendations based on the comprehensive analysis
    5. Identify potential areas for further research
    
    Format your response as a structured cross-industry analysis report with clear sections for each of the above elements.
    """
    
    print("\nSynthesizing cross-industry insights...")
    synthesis_result = await coordinator.llm_orchestrator.generate(
        model=coordinator.llm_config.model,
        messages=[{"role": "system", "content": synthesis_task}]
    )
    
    print("\nCross-Industry Analysis Complete")
    print("-" * 80)
    print(synthesis_result.content)
    
    return {
        "individual_surveys": dict(zip(sectors, survey_results)),
        "cross_industry_analysis": synthesis_result.content
    }

async def create_consensus_metrics_team():
    """Create a team of analysts for generating consensus metrics."""
    
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create different analyst personas with varying perspectives
    
    # Bullish Analyst
    bullish_persona = Persona(
        role="Bullish Market Analyst",
        persona="I am a market analyst with a generally optimistic outlook on market opportunities. I focus on growth potential, innovation, and positive economic indicators.",
        objectives=[
            "Identify growth opportunities and positive trends",
            "Highlight innovative companies and technologies",
            "Analyze potential upside scenarios",
            "Provide optimistic but evidence-based forecasts"
        ],
        communication_style="Confident, growth-oriented, and opportunity-focused",
        skills=[
            "Growth analysis",
            "Innovation assessment",
            "Upside scenario modeling",
            "Opportunity identification"
        ]
    )
    
    bullish_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="bullish_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=bullish_persona
    )
    
    # Bearish Analyst
    bearish_persona = Persona(
        role="Bearish Market Analyst",
        persona="I am a market analyst with a cautious outlook on market risks. I focus on identifying potential challenges, overvaluations, and economic headwinds.",
        objectives=[
            "Identify market risks and negative trends",
            "Highlight potential overvaluations and bubbles",
            "Analyze downside scenarios and vulnerabilities",
            "Provide cautious but evidence-based forecasts"
        ],
        communication_style="Cautious, risk-aware, and detail-oriented",
        skills=[
            "Risk analysis",
            "Valuation assessment",
            "Downside scenario modeling",
            "Vulnerability identification"
        ]
    )
    
    bearish_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="bearish_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=bearish_persona
    )
    
    # Value Analyst
    value_persona = Persona(
        role="Value-Oriented Analyst",
        persona="I am a market analyst focused on fundamental value and long-term investment opportunities. I prioritize financial stability, cash flow, and sustainable business models.",
        objectives=[
            "Identify fundamentally sound investment opportunities",
            "Analyze financial stability and cash flow generation",
            "Evaluate long-term competitive advantages",
            "Provide value-oriented investment recommendations"
        ],
        communication_style="Methodical, fundamental, and long-term focused",
        skills=[
            "Fundamental analysis",
            "Cash flow assessment",
            "Competitive advantage evaluation",
            "Long-term forecasting"
        ]
    )
    
    value_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="value_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=value_persona
    )
    
    # Growth Analyst
    growth_persona = Persona(
        role="Growth-Oriented Analyst",
        persona="I am a market analyst focused on high-growth opportunities and emerging trends. I prioritize revenue growth, market expansion, and disruptive potential.",
        objectives=[
            "Identify high-growth investment opportunities",
            "Analyze revenue growth trajectories and TAM expansion",
            "Evaluate disruptive potential and market capture",
            "Provide growth-oriented investment recommendations"
        ],
        communication_style="Dynamic, forward-looking, and trend-focused",
        skills=[
            "Growth trajectory analysis",
            "TAM assessment",
            "Disruption evaluation",
            "Emerging trend identification"
        ]
    )
    
    growth_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="growth_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=growth_persona
    )
    
    # Technical Analyst
    technical_persona = Persona(
        role="Technical Analyst",
        persona="I am a market analyst focused on price patterns, momentum, and technical indicators. I analyze market behavior and price action to identify trading opportunities.",
        objectives=[
            "Identify technical patterns and trading signals",
            "Analyze price momentum and market sentiment",
            "Evaluate support/resistance levels and trend strength",
            "Provide technically-oriented trading recommendations"
        ],
        communication_style="Pattern-focused, data-driven, and precise",
        skills=[
            "Technical pattern recognition",
            "Momentum analysis",
            "Support/resistance identification",
            "Trend strength evaluation"
        ]
    )
    
    technical_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="technical_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=technical_persona
    )
    
    # Create Chief Investment Strategist to coordinate
    strategist_persona = Persona(
        role="Chief Investment Strategist",
        persona="I am a Chief Investment Strategist responsible for synthesizing diverse market perspectives into coherent investment strategies. I evaluate different viewpoints, identify consensus, and develop balanced investment recommendations.",
        objectives=[
            "Synthesize diverse market perspectives",
            "Identify areas of consensus and disagreement",
            "Develop balanced investment strategies",
            "Provide comprehensive market outlook"
        ],
        communication_style="Balanced, integrative, and strategic",
        skills=[
            "Perspective synthesis",
            "Consensus building",
            "Strategy development",
            "Risk-reward balancing"
        ]
    )
    
    chief_strategist = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="chief_strategist",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=strategist_persona
    )
    
    # Define chat environment for consensus building
    chat_env = {
        "name": "consensus_building_chat",
        "mechanism": "chat",
        "form_cohorts": False,
        "sub_rounds": 3,
        "group_size": 6,
        "task_prompt": "Develop consensus market metrics through collaborative discussion."
    }
    
    # Create the consensus team
    consensus_team = MarketAgentTeam(
        name="Market Consensus Team",
        manager=chief_strategist,
        agents=[
            bullish_analyst,
            bearish_analyst,
            value_analyst,
            growth_analyst,
            technical_analyst
        ],
        mode="collaborative",
        use_group_chat=True,
        shared_context={
            "analysis_framework": "Multi-perspective market evaluation",
            "consensus_goal": "Develop balanced market metrics that incorporate diverse viewpoints",
            "output_requirements": "Quantitative consensus metrics with supporting rationale"
        },
        environments=[
            chat_env
        ]
    )
    
    print(f"Created Market Consensus Team: {consensus_team.name}")
    print(f"Team members: {[agent.id for agent in consensus_team.agents]}")
    print(f"Team manager: {consensus_team.manager.id}")
    
    return consensus_team

async def run_consensus_metrics_generation(team, market_scenario):
    """Run a consensus metrics generation process with the team."""
    
    task = f"""
    As a market analysis team, develop consensus metrics for the following market scenario:
    
    MARKET SCENARIO:
    {market_scenario}
    
    Your team should:
    
    1. The Chief Investment Strategist should:
       - Facilitate the discussion and ensure all perspectives are considered
       - Identify areas of consensus and disagreement
       - Synthesize the diverse viewpoints into balanced metrics
       - Provide the final consensus recommendations
    
    2. The Bullish Market Analyst should:
       - Identify positive trends and growth opportunities
       - Highlight supportive economic indicators
       - Provide optimistic but evidence-based forecasts
       - Advocate for the bull case scenario
    
    3. The Bearish Market Analyst should:
       - Identify risks and negative trends
       - Highlight concerning economic indicators
       - Provide cautious but evidence-based forecasts
       - Advocate for the bear case scenario
    
    4. The Value-Oriented Analyst should:
       - Evaluate fundamental valuations and financial stability
       - Assess cash flow generation and sustainability
       - Identify value opportunities and overvalued segments
       - Provide long-term perspective on market conditions
    
    5. The Growth-Oriented Analyst should:
       - Evaluate growth trajectories and market expansion
       - Identify high-growth segments and opportunities
       - Assess innovation and disruption potential
       - Provide perspective on growth vs. valuation balance
    
    6. The Technical Analyst should:
       - Analyze price patterns and momentum
       - Evaluate technical indicators and market sentiment
       - Identify support/resistance levels and trend strength
       - Provide short to medium-term price projections
    
    Through collaborative discussion, develop consensus metrics for the following:
    
    1. Market Direction (6-12 month outlook)
       - Consensus price targets (index levels)
       - Probability distribution of bullish/neutral/bearish scenarios
       - Key inflection points to monitor
    
    2. Sector Recommendations
       - Overweight/Neutral/Underweight ratings for major sectors
       - Highest conviction sector picks (both positive and negative)
       - Sector rotation expectations
    
    3. Risk Assessment
       - Probability of significant correction (>10%)
       - Key risk factors ranked by importance
       - Potential surprise factors (both positive and negative)
    
    4. Investment Strategy Recommendations
       - Asset allocation guidance
       - Factor/style preferences (value/growth, large/small cap, etc.)
       - Tactical positioning recommendations
    
    The final output should include quantitative consensus metrics with supporting rationale that reflects the team's collective wisdom while acknowledging areas of disagreement.
    """
    
    print(f"\nGenerating consensus metrics for scenario: {market_scenario[:100]}...")
    print("-" * 80)
    
    # Run the team discussion to generate consensus metrics
    result = await team.execute(task)
    
    print("\nConsensus Metrics Generation Complete")
    print("-" * 80)
    print(result)
    
    return result

async def main():
    # Create and run market survey swarm
    print("\n=== MARKET SURVEY SWARM EXAMPLE ===\n")
    swarm = await create_market_survey_swarm()
    survey_results = await run_parallel_market_surveys(swarm)
    
    # Create and run consensus metrics team
    print("\n=== CONSENSUS METRICS TEAM EXAMPLE ===\n")
    consensus_team = await create_consensus_metrics_team()
    
    market_scenarios = [
        """
        Current Market Environment: Major equity indices are near all-time highs after a 15% rally over the past 6 months.
        Valuations are elevated with the S&P 500 P/E ratio at 22x forward earnings, above the 10-year average of 17x.
        
        Economic Indicators:
        - GDP growth: 2.3% year-over-year
        - Inflation: 3.2%, down from 4.1% a year ago
        - Unemployment: 3.9%
        - Fed Funds Rate: 4.75-5.00%, with market expectations of 2-3 rate cuts in the next 12 months
        - 10-Year Treasury Yield: 4.2%
        - Consumer Sentiment: Moderately positive but showing signs of caution
        
        Corporate Fundamentals:
        - Earnings growth: 5% year-over-year
        - Profit margins: Beginning to compress from peak levels
        - Corporate debt levels: Elevated but manageable given current interest rates
        - Share buybacks: Slowing from previous year's pace
        
        Sector Performance (Last 6 Months):
        - Technology: +22%
        - Healthcare: +8%
        - Financials: +15%
        - Energy: -3%
        - Consumer Discretionary: +12%
        - Consumer Staples: +4%
        - Industrials: +10%
        - Utilities: +2%
        
        Geopolitical Factors:
        - Ongoing trade tensions between major economies
        - Regional conflicts creating supply chain uncertainties
        - Upcoming elections in several major economies
        
        Technical Indicators:
        - Major indices above 50-day and 200-day moving averages
        - RSI levels indicating overbought conditions in some sectors
        - Market breadth narrowing with fewer stocks driving index gains
        - Volatility (VIX) at relatively low levels of 16
        """
    ]
    
    for scenario in market_scenarios:
        await run_consensus_metrics_generation(consensus_team, scenario)

if __name__ == "__main__":
    asyncio.run(main())
