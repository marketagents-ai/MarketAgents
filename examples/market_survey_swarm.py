import asyncio
import math
from typing import List
from market_agents.agents.market_agent import MarketAgent
from market_agents.orchestrators.market_agent_swarm import MarketAgentSwarm
from market_agents.agents.personas.persona import Persona
from minference.lite.models import LLMConfig, ResponseFormat
from market_agents.orchestrators.config import GroupChatConfig, WebResearchConfig
from pydantic import BaseModel, Field

# 1. Define the list of goods
GOODS = [
    "Eggs",
    "Milk",
    "Bread",
    "Chicken (whole)",
    "Ground Beef",
    "Fresh Vegetables",
    "Fresh Fruit",
    "Coffee",
    "Canned Tuna",
    "Pasta"
]
class GoodPriceImpact(BaseModel):
    """Impact details for a single good"""
    percent_change: float = Field(..., description="Estimated percentage price change")
    drivers: str = Field(..., description="Key drivers including tariffs and supply chain factors")
    confidence: float = Field(..., description="Confidence level in the estimate (0-1)")
    sources: List[str] = Field(default_factory=list, description="Sources used for the analysis")

class TariffPriceImpactSummary(BaseModel):
    """Overall summary containing impacts for all specified goods"""
    eggs: GoodPriceImpact = Field(..., description="Price impact analysis for eggs")
    milk: GoodPriceImpact = Field(..., description="Price impact analysis for milk")
    bread: GoodPriceImpact = Field(..., description="Price impact analysis for bread")
    chicken: GoodPriceImpact = Field(..., alias="chicken_whole", description="Price impact analysis for whole chicken")
    ground_beef: GoodPriceImpact = Field(..., description="Price impact analysis for ground beef")
    fresh_vegetables: GoodPriceImpact = Field(..., description="Price impact analysis for fresh vegetables")
    fresh_fruit: GoodPriceImpact = Field(..., description="Price impact analysis for fresh fruit")
    coffee: GoodPriceImpact = Field(..., description="Price impact analysis for coffee")
    canned_tuna: GoodPriceImpact = Field(..., description="Price impact analysis for canned tuna")
    pasta: GoodPriceImpact = Field(..., description="Price impact analysis for pasta")

    class Config:
        allow_population_by_field_name = True

async def create_survey_agents(goods: List[str]) -> List[MarketAgent]:
    """Instantiate one Price‑Impact Surveyor agent per good."""
    persona = Persona(
        role="Market Researcher",
        persona="Market researcher specialized in assessing the impact of trade policies on consumer good prices",
        objectives=[
            "Estimate percent price change since the Trump administration's reciprocal tariffs",
            "Identify key drivers (tariff rates, import shares, supply‑chain factors)",
            "Summarize findings in a concise JSON-friendly format"
        ],
        skills=[
            "Trade policy analysis",
            "CPI interpretation",
            "Economic data analysis"
        ]
    )

    llm_cfg = LLMConfig(
        model="gpt-4o",
        client="openai",
        response_format=ResponseFormat.tool,
        temperature=0.2,
        use_cache=True
    )

    agents = []
    for i in range(4):
        agent = await MarketAgent.create(
            name=f"analyst_{i}",
            persona=persona,
            llm_config=llm_cfg
        )
        agents.append(agent)
    return agents

async def main():
    # Create the survey agents
    agents = await create_survey_agents(GOODS)

    # Define the unified survey task
    task = (
        "Research and analyze the price impact of Trump administration's reciprocal tariffs "
        "on each of these specific consumer goods. For each good listed below, you must provide:\n"
        "1. The estimated percentage price change\n"
        "2. Key drivers (tariff rates, supply chain impacts)\n"
        "3. Your confidence level in the estimate (0-1)\n"
        "4. Sources used\n\n"
        f"Required goods to analyze (all must be included):\n{GOODS}"
        "Provide a complete structured analysis that includes ALL of these goods."
    )

    # Define environments configuration
    environments = [
        GroupChatConfig(
            name="market_survey_chat",
            initial_topic=task,
            mechanism="group_chat",
            sub_rounds=1,
            group_size=2,
        ).model_dump(),
        WebResearchConfig(
            name="tariff_research",
            initial_query=task,
            mechanism="web_research",
            urls_per_query=1,
            summary_model=TariffPriceImpactSummary,
            sub_rounds=1
        ).model_dump(),
    ]

    # Assemble the swarm
    swarm = MarketAgentSwarm(
        name="TariffPriceImpactSurvey",
        agents=agents,
        form_cohorts=True,
        cohort_size=2,
        max_rounds=2,
        environments=environments
    )

    # Deploy the swarm survey
    await swarm.deploy_swarm(task)
             
if __name__ == "__main__":
    asyncio.run(main())
