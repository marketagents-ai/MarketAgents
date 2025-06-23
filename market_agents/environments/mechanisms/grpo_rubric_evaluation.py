from __future__ import annotations
from typing import Dict, Any, List, Optional, Type, Union

import torch
import yaml
from pydantic import BaseModel, Field

from market_agents.environments.environment import (
    Mechanism,
    LocalAction, GlobalAction,
    LocalObservation, GlobalObservation,
    LocalEnvironmentStep, EnvironmentStep,
    ActionSpace, ObservationSpace, MultiAgentEnvironment
)
from market_agents.environments.mechanisms.evaluation import (
    CandidateAction, JudgeScore, JudgeAction,
    EvalActionSpace, EvalObservation, EvalLocalObservation, EvalGlobalObservation,
    EvalObservationSpace
)

# ------------------------------------------------------------------#
#                             MECHANISM                             #
# ------------------------------------------------------------------#
class GRPORubricEvaluationMechanism(Mechanism):
    """
    Combines rubric-based evaluation with GRPO advantage calculation.
    """
    sequential: bool = False

    rubric: Dict[str, Any]
    candidate_id: str
    judge_ids: List[str]
    prompt: str
    group_size: int
    norm_adv_by_std: bool = True
    epsilon: float = 1e-6

    current_phase: str = "generation"
    _answer: Optional[str] = None
    _scores: Dict[str, JudgeScore] = Field(default_factory=dict)
    _advantages: Dict[str, float] = Field(default_factory=dict)
    _done: bool = False

    # ---------------- helper ---------------- #
    def _aggregate(self) -> Dict[str, float]:
        total, weight_sum = 0.0, 0.0
        for crit, spec in self.rubric["criteria"].items():
            weight = spec["weight"]
            level = self._scores.get(crit, JudgeScore(criterion=crit, level=0)).level / 3.0
            total += weight * level
            weight_sum += weight
        return {"total": total / weight_sum if weight_sum else 0.0}

    def _calculate_advantage(self) -> None:
        """Calculates GRPO advantage for judge scores."""
        if len(self._scores) >= self.group_size:
            # Aggregate judge scores into a single score per agent
            agent_scores = {}
            for criterion, score in self._scores.items():
                agent_id = score.agent_id  # Assuming JudgeScore has agent_id
                if agent_id not in agent_scores:
                    agent_scores[agent_id] = 0.0
                agent_scores[agent_id] += score.level  # Sum levels across criteria

            ordered = sorted(agent_scores.items(), key=lambda kv: kv[1], reverse=True)
            if len(ordered) > 1:  # Need at least two agents to calculate advantage
                r_max, r_second = ordered[0][1], ordered[1][1]
                advantage = r_max - r_second
                if self.norm_adv_by_std:
                    std = torch.tensor(list(agent_scores.values())).std()
                    advantage /= (std + self.epsilon)

                winner = ordered[0][0]
                self._advantages = {aid: (advantage if aid == winner else 0.0)
                                    for aid in agent_scores}
            else:
                # Handle the case where there's only one agent
                self._advantages = {ordered[0][0]: 0.0}  # No advantage if only one agent

            self._done = True

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
                self._calculate_advantage()  # Calculate advantage after scoring
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
            aggregated=self._aggregate() if self.current_phase == "done" else None,
            advantages=self._advantages if self.current_phase == "done" else {}  # Include advantages
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
            "aggregated": self._aggregate() if self.current_phase == "done" else None,
            "advantages": self._advantages  # Include advantages in global state
        }


# ------------------------------------------------------------------#
#                        ENVIRONMENT WRAPPER                        #
# ------------------------------------------------------------------#
class GRPORubricEvaluationEnvironment(MultiAgentEnvironment):
    """
    Environment wrapping GRPORubricEvaluationMechanism.
    """
    name: str = "grpo_rubric_evaluation"
    action_space: EvalActionSpace = Field(default_factory=EvalActionSpace)
    observation_space: EvalObservationSpace = Field(default_factory=EvalObservationSpace)
    mechanism: GRPORubricEvaluationMechanism

    def __init__(
        self,
        *,
        rubric_path: str,
        candidate_id: str,
        judge_ids: List[str],
        prompt: str,
        group_size: int,
        norm_adv_by_std: bool = True,
        **kwargs
    ):
        with open(rubric_path, "r") as fp:
            rubric_def = yaml.safe_load(fp)

        mech = GRPORubricEvaluationMechanism(
            rubric=rubric_def,
            candidate_id=candidate_id,
            judge_ids=judge_ids,
            prompt=prompt,
            group_size=group_size,
            norm_adv_by_std=norm_adv_by_std
        )

        super().__init__(
            name="grpo_rubric_evaluation",
            mechanism=mech,
            **kwargs
        )