# market_agents/environments/mechanisms/evaluation.py
"""
Generic rubric-based evaluation mechanism & environment.

Workflow
--------
1. Candidate agent sends a free-form answer (CandidateAction).
2. One judge agent per rubric criterion returns a structured JudgeScore.
3. When every criterion is scored, the mechanism aggregates the weighted
   total and ends the episode.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Type, Union

import yaml
from pydantic import BaseModel, Field

from market_agents.environments.environment import (
    Mechanism,
    LocalAction, GlobalAction,
    LocalObservation, GlobalObservation,
    LocalEnvironmentStep, EnvironmentStep,
    ActionSpace, ObservationSpace, MultiAgentEnvironment
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------#
#                              ACTIONS                              #
# ------------------------------------------------------------------#
class CandidateAction(LocalAction):
    """Raw answer produced by the foundation model."""
    action: str = Field(..., description="Free-form text answer.")


class JudgeScore(BaseModel):
    """Score for a single rubric criterion."""
    criterion: str = Field(..., description="Rubric key, e.g. 'reasoning_depth'")
    level: int = Field(..., ge=0, le=3, description="Discrete level 0–3")
    notes: str = Field("", description="Short justification or evidence")


class JudgeAction(LocalAction):
    """Judge agent submits a score object."""
    action: JudgeScore


# ------------------------------------------------------------------#
#                         ACTION / OBS SPACES                       #
# ------------------------------------------------------------------#
class EvalActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [CandidateAction, JudgeAction]


class EvalObservation(BaseModel):
    phase: str = Field(..., description="'generation' | 'scoring' | 'done'")
    prompt: str
    candidate_answer: Optional[str] = None
    partial_scores: Dict[str, JudgeScore] = Field(default_factory=dict)
    aggregated: Optional[Dict[str, float]] = None


class EvalLocalObservation(LocalObservation):
    observation: EvalObservation


class EvalGlobalObservation(GlobalObservation):
    observations: Dict[str, EvalLocalObservation]


class EvalObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [EvalLocalObservation]


# ------------------------------------------------------------------#
#                             MECHANISM                             #
# ------------------------------------------------------------------#
class EvaluationMechanism(Mechanism):
    """
    Two-phase state machine:

    • generation — waits for a CandidateAction.  
    • scoring    — collects JudgeAction per criterion.  
    • done       — aggregates weighted total and stops.
    """
    sequential: bool = False

    rubric: Dict[str, Any]
    candidate_id: str
    judge_ids: List[str]
    prompt: str

    current_phase: str = "generation"
    _answer: Optional[str] = None
    _scores: Dict[str, JudgeScore] = Field(default_factory=dict)

    # ---------------- helper ---------------- #
    def _aggregate(self) -> Dict[str, float]:
        total, weight_sum = 0.0, 0.0
        for crit, spec in self.rubric["criteria"].items():
            weight = spec["weight"]
            level = self._scores.get(crit, JudgeScore(criterion=crit, level=0)).level / 3.0
            total += weight * level
            weight_sum += weight
        return {"total": total / weight_sum if weight_sum else 0.0}

    # ---------------- core step ------------- #
    def step(
        self,
        action: Union[LocalAction, GlobalAction],
        cohort_id: Optional[str] = None
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:

        # Phase 1 – candidate answer
        if self.current_phase == "generation":
            cand = (
                action if isinstance(action, CandidateAction)
                else next((a for a in action.actions.values() if isinstance(a, CandidateAction)), None)
            )
            if cand is None:
                raise ValueError("Expected CandidateAction during generation phase")
            self._answer = cand.action
            self.current_phase = "scoring"
            done = False

        # Phase 2 – judge scoring
        elif self.current_phase == "scoring":
            acts = [action] if isinstance(action, JudgeAction) else action.actions.values()
            for ja in acts:
                if isinstance(ja, JudgeAction):
                    self._scores[ja.action.criterion] = ja.action
            if len(self._scores) >= len(self.rubric["criteria"]):
                self.current_phase = "done"
            done = self.current_phase == "done"

        # Already finished
        else:
            done = True

        # Build observation
        obs_payload = EvalObservation(
            phase=self.current_phase,
            prompt=self.prompt,
            candidate_answer=self._answer,
            partial_scores=self._scores,
            aggregated=self._aggregate() if self.current_phase == "done" else None
        )

        # Return local or global step
        if isinstance(action, LocalAction):
            local_obs = EvalLocalObservation(agent_id=action.agent_id, observation=obs_payload)
            return LocalEnvironmentStep(observation=local_obs, done=done, info={})

        locs = {aid: EvalLocalObservation(agent_id=aid, observation=obs_payload)
                for aid in action.actions.keys()}
        glob = EvalGlobalObservation(observations=locs)
        return EnvironmentStep(global_observation=glob, done=done, info={})

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "phase": self.current_phase,
            "answer": self._answer,
            "scores": {k: v.model_dump() for k, v in self._scores.items()},
            "aggregated": self._aggregate() if self.current_phase == "done" else None
        }


# ------------------------------------------------------------------#
#                        ENVIRONMENT WRAPPER                        #
# ------------------------------------------------------------------#
class EvaluationEnvironment(MultiAgentEnvironment):
    """
    Plug-and-play environment wrapping EvaluationMechanism.
    Initialise with a rubric YAML, candidate+judge IDs, and the task prompt.
    """
    name: str = "evaluation"
    action_space: EvalActionSpace = Field(default_factory=EvalActionSpace)
    observation_space: EvalObservationSpace = Field(default_factory=EvalObservationSpace)
    mechanism: EvaluationMechanism

    def __init__(
        self,
        *,
        rubric_path: str,
        candidate_id: str,
        judge_ids: List[str],
        prompt: str,
        **kwargs
    ):
        with open(rubric_path, "r") as fp:
            rubric_def = yaml.safe_load(fp)

        mech = EvaluationMechanism(
            rubric=rubric_def,
            candidate_id=candidate_id,
            judge_ids=judge_ids,
            prompt=prompt
        )

        super().__init__(
            name="evaluation",
            mechanism=mech,
            **kwargs
        )