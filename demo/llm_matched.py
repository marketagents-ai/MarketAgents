import asyncio
import os
from dotenv import load_dotenv
from market_agents.economics.econ_agent import ZiFactory, ZiParams
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.economics.scenario import Scenario
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from market_agents.market_orchestrator import MarketOrchestrator, MarketOrchestratorState, zi_scenario
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMConfig


async def llm_matched_scenario(zi_scenario:Scenario, clones_config:Optional[LLMConfig]=None, ai_utils:Optional[ParallelAIUtilities]=None, max_rounds:int=10):
    load_dotenv()
    scenario = zi_scenario.model_copy(deep=True)
    # Set up ParallelAIUtilities
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=10000)
    parallel_ai = ParallelAIUtilities(
        oai_request_limits=oai_request_limits,
        anthropic_request_limits=anthropic_request_limits
    ) if ai_utils is None else ai_utils

    # Create a good
    scenario.generate_zi_agents = False


    orchestrator = MarketOrchestrator(llm_agents=[],
                                       goods=scenario.goods,
                                       ai_utils=parallel_ai,
                                       max_rounds=max_rounds,
                                       scenario=scenario,
                                       clones_config=LLMConfig(model="gpt-4o-mini",
                                                               temperature=0.0,
                                                               client="openai",
                                                               response_format="tool",
                                                               max_tokens=250) if clones_config is None else clones_config) 
    # return orchestrator
    await orchestrator.run_scenario()
    return orchestrator

if __name__ == "__main__":
    apple = Good(name="apple", quantity=0)
    buyer_params = ZiParams(
        id="buyer_template",
        initial_cash=10000.0,
        initial_goods={"apple": 0},
        base_values={"apple": 20.0},
        num_units=20,
        noise_factor=0.05,
        max_relative_spread=0.2,
        is_buyer=True
    )

    seller_params = ZiParams(
        id="seller_template",
        initial_cash=0,
        initial_goods={"apple": 20},
        base_values={"apple": 10.0},
        num_units=20,
        noise_factor=0.05,
        max_relative_spread=0.2,
        is_buyer=False
    )
    num_buyers = 10
    num_sellers = 10
    # Create DoubleAuction mechanism
    factories = [
        ZiFactory(
            id=f"factory_episode_{0}",
            goods=["apple"],
            num_buyers=num_buyers,  # Increase buyers by 1 each episode
            num_sellers=num_sellers,     # Keep sellers constant
            buyer_params=buyer_params,
            seller_params=seller_params
        )
    ]
    scenario = Scenario(
        name="Static Apple Market",
        goods=["apple"],
        factories=factories
    )
    orchestrator = asyncio.run(llm_matched_scenario(scenario, max_rounds=10))
    state = orchestrator.state
