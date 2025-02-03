from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from market_agents.memory.storage_models import MemoryObject


class BaseRewardFunction(BaseModel):
    """
    Base class for reward functions that combine environment rewards with agent self-assessment.
    Inherit from this class to create environment-specific reward functions.
    """
    environment_weight: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Weight for environment-provided rewards"
    )
    self_eval_weight: float = Field(
        0.4, ge=0.0, le=1.0,
        description="Weight for agent's self-eval reward"
    )
    economic_weight: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Weight for economic value in reward calculation (only used if economic agent exists)"
    )

    class Config:
        arbitrary_types_allowed = True

    def compute(
        self,
        environment_reward: Optional[float],
        reflection_data: Dict[str, Any],
        economic_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute composite reward with detailed breakdown.
        Override this method for custom reward calculations.
        """
        self_reward = self._extract_self_reward(reflection_data)
        
        composite = (
            (environment_reward or 0.0) * self.environment_weight +
            self_reward * self.self_eval_weight
        )
        
        if economic_value is not None:
            composite += economic_value * self.economic_weight
            components = {
                "environment": environment_reward or 0.0,
                "self_eval": self_reward,
                "economic": economic_value
            }
        else:
            components = {
                "environment": environment_reward or 0.0,
                "self_eval": self_reward
            }
        
        return {
            "total_reward": round(composite, 4),
            "components": components,
            "weights": self.dict(include={'environment_weight', 'self_eval_weight', 'economic_weight'}),
            "type": self.__class__.__name__
        }

    def _extract_self_reward(self, reflection_data: Dict) -> float:
        """Parse self-eval reward from reflection data"""
        return float(reflection_data.get("self_reward", 0.0))
    
class RLExperience(BaseModel):
    """Minimal RL experience data that complements episodic memory"""
    state: Dict[str, Any] = Field(
        description="Agent's local observation from environment"
    )
    action: Dict[str, Any] = Field(
        description="Action taken"
    )
    reward_data: Dict[str, Any] = Field(
        description="Reward breakdown (environment, self-eval, economic)"
    )
    next_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Next observation if available"
    )
    exploration_rate: float = Field(
        description="Current exploration rate when action was taken"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )