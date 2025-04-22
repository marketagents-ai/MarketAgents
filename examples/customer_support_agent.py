import asyncio
from typing import Any, Dict, List, Literal, TypedDict
from pydantic import BaseModel
import random

from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.chat import ChatEnvironment
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from minference.lite.models import (
    CallableTool,
    LLMClient,
    LLMConfig,
    ResponseFormat,
)

class PurchaseSummary(TypedDict):
    order_id: str
    items: List[str]
    total_amount: float
    order_date: str

class OrderStatus(TypedDict):
    order_id: str
    status: Literal["processing", "shipped", "delivered", "cancelled"]

def get_purchase_history() -> Dict[str, Any]:
    """Return the user's two most recent purchases."""
    return {
        "purchases": [
            {
                "order_id": "A1001",
                "items": ["Kindle Paperwhite"],
                "total_amount": 149.99,
                "order_date": "2025-03-10"
            },
            {
                "order_id": "A0987",
                "items": ["Echo Dot"],
                "total_amount": 49.99,
                "order_date": "2025-02-22"
            }
        ]
    }

async def check_order_status(order_id: str) -> OrderStatus:
    """Return a mock status for the given order_id."""
    if order_id not in {"A1001", "A0987"}:
        raise ValueError("Order not found")
    return OrderStatus(
        order_id=order_id,
        status=random.choice(
            ["processing", "shipped", "delivered", "cancelled"]
        ),
    )

async def get_store_policy() -> str:
    """Return a short excerpt of the store's return & refund policy."""
    return (
        "Returns are accepted within 30 days of delivery for items in original "
        "condition. Refunds are processed to the original payment method."
    )

async def escalate_to_human() -> dict[str, str]:
    """Provide a link to human support."""
    return {"url": "https://support.shopsmart.com/contact", "type": "escalation"}

purchase_history_tool = CallableTool.from_callable(
    get_purchase_history, name="get_purchase_history"
)
order_status_tool = CallableTool.from_callable(
    check_order_status, name="check_order_status"
)
store_policy_tool = CallableTool.from_callable(
    get_store_policy, name="get_store_policy"
)
escalate_tool = CallableTool.from_callable(
    escalate_to_human, name="escalate_to_human"
)

TOOLS = [
    purchase_history_tool,
    order_status_tool,
    store_policy_tool,
    escalate_tool,
]

support_persona = Persona(
    role="Customer Support Agent",
    persona=(
        "You are an AI customer‑support agent for XPTO Telecom. "
        "Assist customers quickly, accurately and politely using the tools."
    ),
    objectives=[
        "Resolve customer questions",
        "Use the provided tools – never hallucinate",
        "Escalate to human only if the issue cannot be solved automatically",
    ],
    skills=["Customer service", "Troubleshooting", "Order tracking"],
)


async def main() -> None:
    """
    Builds the agent, sends one sample message and prints the reply.
    Extend or loop as needed for multi‑turn interactions.
    """
    storage_utils = AgentStorageAPIUtils(
        config=AgentStorageConfig(
            model="text-embedding-3-small",
            embedding_provider="openai",
            vector_dim=256,
        )
    )

    chat_env = ChatEnvironment(name="support_chat", tools = TOOLS)

    agent = await MarketAgent.create(
        name="xpto-support",
        persona=support_persona,
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model="gpt-4o-mini",
            temperature=0.4,
            response_format=ResponseFormat.auto_tools,
        ),
        environments={"chat": chat_env},
        storage_utils=storage_utils,
    )

    agent.task = "Hi, can you check the status of my latest order?"

    reply = await agent.run_episode()
    print("Assistant:", reply)

if __name__ == "__main__":
    asyncio.run(main())