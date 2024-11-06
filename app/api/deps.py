from typing import Annotated, Generator
from fastapi import Depends
from sqlmodel import Session
from app.db.session import get_session, engine
from market_agents.inference.sql_inference import ParallelAIUtilities, RequestLimits

def get_db(session: Session = Depends(get_session)) -> Generator[Session, None, None]:
    try:
        yield session
    finally:
        session.close()

DatabaseDep = Annotated[Session, Depends(get_db)]

# Create ParallelAIUtilities with the same engine
_ai_utils = ParallelAIUtilities(
    engine=engine,
    oai_request_limits=RequestLimits(
        max_requests_per_minute=500,
        max_tokens_per_minute=200000,
        provider="openai"
    )
)

def get_ai_utils() -> ParallelAIUtilities:
    """Get the singleton AI utilities instance"""
    return _ai_utils