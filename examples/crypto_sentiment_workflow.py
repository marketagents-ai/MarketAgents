# /examples/crypto_sentiment_workflow.py
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from market_agents.workflows.workflow_utils import setup_mcp_environment
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironmentConfig
from market_agents.workflows.market_agent_workflow import (
    Workflow, WorkflowStep, WorkflowStepIO
)
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona
from minference.lite.models import LLMConfig, ResponseFormat, StructuredTool

class ChainInfo(BaseModel):
    """Input schema for chain selection"""
    chain: str = Field(..., description="Chain identifier (e.g. 'solana', 'ethereum')")

class TokenDiscoveryOutput(BaseModel):
    """Output schema for token discovery step"""
    token_symbol: str = Field(..., description="Discovered token symbol")
    chain: str = Field(..., description="Chain ID where token was found")
    rationale: str = Field(None, description="Rationale for picking this token")

class TokenMetadata(BaseModel):
    """Schema for token metadata"""
    token_symbol: str
    chain: str
    token_address: Optional[str]
    website: Optional[str]
    socials: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str]

class DexStats(BaseModel):
    """Schema for DEX stats"""
    pair_address: str
    price_usd: float
    volume_24h: float
    liquidity_usd: float
    price_change_24h: float

class SentimentReport(BaseModel):
    meta_data: TokenMetadata
    dex_stats: DexStats
    sentiment_score: float    = Field(..., description="sentiment score between 0 and 1")
    analysis: str             = Field(..., description="analysis with rationale for recommendation")
    recommendation: str       = Field(..., description="Buy / Hold / Sell")

async def create_crypto_sentiment_workflow() -> Workflow:
    cfg = MCPServerEnvironmentConfig(
        name="mcp_crypto",
        mechanism="mcp_server",
        mcp_server_module="market_agents.orchestrators.mcp_server.crypto_mcp_server"
    )
    crypto_env = await setup_mcp_environment(cfg)
    tools = crypto_env.get_tools()

    print(f"Available tools: {list(tools.keys())}")
    for name, tool in tools.items():
        print(f"Tool {name} schema: {tool.input_schema}")

    boosted_tokens_tool = tools["get_boosted_tokens"]
    token_profile_tool = tools["get_token_profile"]
    dex_pair_tool = tools["get_dex_pair"]

    discover_step = WorkflowStep(
        name="discover_token",
        environment_name="mcp_crypto",
        tools=[boosted_tokens_tool],
        subtask="Return the symbol and chainId of the first boosted token from provided chain",
        inputs=[
            WorkflowStepIO(
                name="chain",
                data=ChainInfo
            )
        ],
        output=WorkflowStepIO( 
            name="token_discovery",
            data=TokenDiscoveryOutput
        ),
        run_full_episode=False,
        sequential_tools=True
    )

    metadata_step = WorkflowStep(
        name="collect_metadata",
        environment_name="mcp_crypto",
        tools=[token_profile_tool],
        subtask="Use get_token_profile to fetch token metadata.",
        run_full_episode=False,
        sequential_tools=True
    )

    stats_step = WorkflowStep(
        name="fetch_onchain_stats",
        environment_name="mcp_crypto",
        tools=[dex_pair_tool],
        subtask="Use get_dex_pair to fetch DEX pair statistics.",
        run_full_episode=False,
        sequential_tools=True
    )

    report_tool = StructuredTool.from_pydantic(
        model=SentimentReport,
        name="sentiment_report",
        description="Create a structured sentiment report"
    )

    summary_step = WorkflowStep(
        name="generate_report",
        environment_name="mcp_crypto",
        tools=[report_tool],
        subtask="Create a final sentiment report using all collected data.",
        run_full_episode=False,
        sequential_tools=False
    )

    return Workflow.create(
        name="crypto_sentiment",
        task="Analyse social sentiment and on‑chain stats for the top boosted token from {chain_info.chain}",
        steps=[discover_step, metadata_step, stats_step, summary_step],
        mcp_servers={"mcp_crypto": crypto_env}
    )

SUPPORTED_CHAINS = {
    "solana": "solana",
    "ethereum": "ethereum", 
    "arbitrum": "arbitrum",
    "base": "base"
}

async def run_workflow(chain: str) -> Dict[str, Any]:
    """Run crypto sentiment workflow for a given chain."""
    if chain.lower() not in SUPPORTED_CHAINS:
        raise ValueError(f"Chain {chain} not found in supported chains")
    
    chain = SUPPORTED_CHAINS[chain.lower()]
    chain_info = ChainInfo(chain=chain)
    
    analyst = await MarketAgent.create(
        name="crypto_analyst",
        persona=Persona(
            role="Crypto Analyst",
            persona="You study crypto markets and sentiment.",
            objectives=["Detect hype", "Assess on‑chain strength", "Advise traders"],
            skills=["Sentiment analysis", "On‑chain data", "Risk flags"]
        ),
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            client="openai",
            temperature=0.1,
            response_format=ResponseFormat.auto_tools
        )
    )
    wf = await create_crypto_sentiment_workflow()
    return await wf.execute(
        agent=analyst,
        initial_inputs={
            "chain_info": chain_info
        }
    )

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
    
    result = await run_workflow("solana")
    result_dict = result.model_dump()
    print(f"\nCrypto Sentiment Analysis Results for Solana chain:", 
          json.dumps(result_dict, indent=2, cls=CustomJSONEncoder))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())