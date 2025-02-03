from datetime import datetime
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import logging

from market_agents.verbal_rl.rl_models import BaseRewardFunction, RLExperience
from minference.enregistry import EntityRegistry

logger = logging.getLogger(__name__)
EntityRegistry()._logger = logger


class VerbalRLAgent(BaseModel):
    """Reinforcement Learning subsystem integrated with MarketAgent"""
    policy: Dict[str, Any] = Field(
        default_factory=lambda: {"exploration_rate": 0.2, "learning_rate": 0.01},
        description="RL policy parameters"
    )
    reward_function: BaseRewardFunction = Field(
        default_factory=BaseRewardFunction,
        description="Default reward calculation mechanism"
    )
    last_experience: Optional[RLExperience] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_policy(self, reward: float) -> None:
        """Basic policy update rule using reflection rewards"""
        self.policy["exploration_rate"] *= max(0.5, 1 - self.policy["learning_rate"] * reward)
        logger.debug(f"Updated exploration rate: {self.policy['exploration_rate']:.2f}")

    async def store_experience(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward_data: Dict[str, Any],
        next_state: Dict[str, Any],
        exploration_rate: float,
        created_at: datetime
    ) -> None:
        """Store experience in both short-term and long-term memory"""
        experience = RLExperience(
            state=state,
            action=action,
            reward_data=reward_data,
            next_state=next_state,
            exploration_rate=exploration_rate,
            timestamp=created_at
        )           
        self.last_experience = experience