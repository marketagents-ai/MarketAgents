# grpo_evaluation.py
# -------------------------------------------------------------
# Stand-alone GRPO mechanism & environment that IMPORTS the
# *existing* MarketAgents environment abstractions but does NOT
# modify any existing project files.
#
# Copy-paste this script anywhere inside your repo (e.g.
# market_agents/environments/mechanisms/grpo_evaluation.py) and
# import `GRPOEvaluationEnvironment` in your orchestrator.
#
# -------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Optional, Union, Type

import torch
from pydantic import BaseModel, Field

# Import the already-present environment base
from market_agents.environments.environment import (
    Mechanism,
    LocalAction,
    GlobalAction,
    LocalObservation,
    GlobalObservation,
    LocalEnvironmentStep,
    EnvironmentStep,
    ActionSpace,
    ObservationSpace,
    MultiAgentEnvironment,
)

# =============================================================
# ACTION
# =============================================================
class OutcomeAction(LocalAction):
    """
    Agents send a scalar score (e.g., pass@k success, rubric tally).
    """
    score: float = Field(..., description="Scalar outcome score")

# =============================================================
# OBSERVATIONS
# =============================================================
class _ObsPayload(BaseModel):
    scores: Dict[str, float]
    advantages: Dict[str, float]
    phase: str  # 'collecting' | 'done'


class _LocalObs(LocalObservation):
    observation: _ObsPayload


class _GlobalObs(GlobalObservation):
    observations: Dict[str, _LocalObs]


# Minimal spaces so schema tools keep working
class _ActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [OutcomeAction]


class _ObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [_LocalObs]

# =============================================================
# GRPO MECHANISM
# =============================================================
class GRPOEvaluationMechanism(Mechanism):
    """
    Waits for `group_size` OutcomeAction scores, then computes

        advantage = r_max − r_second_max

    Only the top agent in each group receives the positive advantage.
    """
    sequential: bool = False  # batched

    group_size: int
    norm_adv_by_std: bool = True
    epsilon: float = 1e-6

    # internal buffers
    _scores: Dict[str, float] = Field(default_factory=dict)
    _advantages: Dict[str, float] = Field(default_factory=dict)
    _done: bool = False

    # ---------------- core step ----------------
    def step(
        self,
        action: Union[LocalAction, GlobalAction],
        cohort_id: Optional[str] = None,
    ) -> EnvironmentStep:

        # normalise => Dict[str, OutcomeAction]
        if isinstance(action, OutcomeAction):
            incoming = {action.agent_id: action}
        else:
            incoming = {aid: act for aid, act in action.actions.items()
                        if isinstance(act, OutcomeAction)}

        # record scores
        for aid, act in incoming.items():
            self._scores.setdefault(aid, float(act.score))

        # compute advantage when we have k scores
        if len(self._scores) >= self.group_size and not self._done:
            ordered = sorted(self._scores.items(), key=lambda kv: kv[1], reverse=True)
            r_max, r_second = ordered[0][1], ordered[1][1]
            advantage = r_max - r_second
            if self.norm_adv_by_std:
                std = torch.tensor(list(self._scores.values())).std()
                advantage /= (std + self.epsilon)

            winner = ordered[0][0]
            self._advantages = {aid: (advantage if aid == winner else 0.0)
                                for aid in self._scores}
            self._done = True

        # build observation
        payload = _ObsPayload(
            scores=self._scores,
            advantages=self._advantages,
            phase="done" if self._done else "collecting",
        )
        locals_obs = {aid: _LocalObs(agent_id=aid, observation=payload)
                      for aid in incoming}
        glob_obs = _GlobalObs(observations=locals_obs)

        return EnvironmentStep(global_observation=glob_obs,
                               done=self._done,
                               info={})

    # convenience for debugging
    def get_global_state(self) -> Dict[str, float]:
        return {
            "scores": self._scores,
            "advantages": self._advantages,
            "done": self._done,
        }

# =============================================================
# ENVIRONMENT WRAPPER
# =============================================================
class GRPOEvaluationEnvironment(MultiAgentEnvironment):
    name: str = "grpo_evaluation"
    action_space: _ActionSpace = Field(default_factory=_ActionSpace)
    observation_space: _ObservationSpace = Field(default_factory=_ObservationSpace)
    mechanism: GRPOEvaluationMechanism

    def __init__(
        self,
        *,
        group_size: int,
        norm_adv_by_std: bool = True,
        **kwargs,
    ):
        mech = GRPOEvaluationMechanism(
            group_size=group_size,
            norm_adv_by_std=norm_adv_by_std,
        )
        super().__init__(name=self.name, mechanism=mech, **kwargs)