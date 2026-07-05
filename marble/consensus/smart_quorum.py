"""Smart quorum decision mechanism for memory consensus."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

from marble.consensus.majority_vote import MajorityVoteConsensus
from marble.consensus.models import (
    ConsensusDecision,
    ConsensusVote,
    MemoryProposal,
    VerificationVector,
)


class SmartQuorumConsensus(MajorityVoteConsensus):
    """
    Weighted smart-quorum consensus for MARBLE memory proposals.
    """

    def __init__(
        self,
        *,
        agent_weights: Optional[Mapping[str, float]] = None,
        honest_agents: Optional[Sequence[str]] = None,
        byzantine_agents: Optional[Sequence[str]] = None,
        epsilon_ratio: float = 0.1,
        epsilon: Optional[float] = None,
        use_dynamic_estimate: bool = False,

        confidence_threshold: float = 0.6,
        majority_threshold: float = 0.5,
        strict_majority: bool = True,
        minimum_votes: int = 1,
        hard_fail_dimensions: Sequence[str] = ("security",),
    ) -> None:
        super().__init__(
            confidence_threshold=confidence_threshold,
            majority_threshold=majority_threshold,
            strict_majority=strict_majority,
            minimum_votes=minimum_votes,
            hard_fail_dimensions=hard_fail_dimensions,
        )
        if epsilon_ratio < 0:
            raise ValueError("epsilon_ratio cannot be negative.")
        if epsilon is not None and epsilon < 0:
            raise ValueError("epsilon cannot be negative.")

        self.agent_weights = self._validate_weights(agent_weights or {})
        self.honest_agents: Set[str] = set(honest_agents or ())
        self.byzantine_agents: Set[str] = set(byzantine_agents or ())
        overlap = self.honest_agents & self.byzantine_agents
        if overlap:
            raise ValueError(
                "Agents cannot be both honest and byzantine: "
                + ", ".join(sorted(overlap))
            )
        self.epsilon_ratio = float(epsilon_ratio)
        self.epsilon = None if epsilon is None else float(epsilon)
        self.use_dynamic_estimate = use_dynamic_estimate

    def build_vote(
        self,
        proposal: MemoryProposal,
        verification: VerificationVector,
        voter_agent_id: Optional[str] = None,
    ) -> ConsensusVote:
        vote = super().build_vote(proposal, verification, voter_agent_id)
        weight = self.weight_for(vote.voter_agent_id)
        return ConsensusVote(
            proposal_id=vote.proposal_id,
            voter_agent_id=vote.voter_agent_id,
            accept=vote.accept,
            confidence_score=vote.confidence_score,
            verification=vote.verification,
            reason=vote.reason,
            weight=weight,
        )

    def decide(
        self,
        proposal: MemoryProposal,
        verifications: Iterable[VerificationVector],
    ) -> ConsensusDecision:
        votes = [self.build_vote(proposal, vector) for vector in verifications]
        accept_count = sum(1 for vote in votes if vote.accept)
        reject_count = len(votes) - accept_count
        total_weight = sum(vote.weight for vote in votes)
        accept_weight = sum(vote.weight for vote in votes if vote.accept)
        acceptance_ratio = accept_weight / total_weight if total_weight else 0.0
        quorum = self.calculate_quorum(votes)

        if len(votes) < self.minimum_votes:
            return ConsensusDecision(
                proposal_id=proposal.proposal_id,
                result="pending",
                votes=votes,
                accept_count=accept_count,
                reject_count=reject_count,
                acceptance_ratio=acceptance_ratio,
                threshold=self.majority_threshold,
                confidence_threshold=self.confidence_threshold,
                total_weight=total_weight,
                accept_weight=accept_weight,
                metadata={
                    "strategy": "smart_quorum",
                    "reason": "minimum_votes_not_met",
                    "agent_weights": dict(self.agent_weights),
                    "honest_agents": sorted(self.honest_agents),
                    "byzantine_agents": sorted(self.byzantine_agents),
                    **quorum,
                },
            )

        if self.strict_majority:
            has_weighted_majority = acceptance_ratio > self.majority_threshold
        else:
            has_weighted_majority = acceptance_ratio >= self.majority_threshold
        has_quorum_certificate = accept_weight > quorum["qc"]
        passed = has_quorum_certificate and has_weighted_majority

        return ConsensusDecision(
            proposal_id=proposal.proposal_id,
            result="pass" if passed else "fail",
            votes=votes,
            accept_count=accept_count,
            reject_count=reject_count,
            acceptance_ratio=acceptance_ratio,
            threshold=self.majority_threshold,
            confidence_threshold=self.confidence_threshold,
            total_weight=total_weight,
            accept_weight=accept_weight,
            metadata={
                "strategy": "smart_quorum",
                "has_quorum_certificate": has_quorum_certificate,
                "has_weighted_majority": has_weighted_majority,
                "agent_weights": dict(self.agent_weights),
                "honest_agents": sorted(self.honest_agents),
                "byzantine_agents": sorted(self.byzantine_agents),
                **quorum,
            },
        )

    def calculate_quorum(self, votes: Sequence[ConsensusVote]) -> Dict[str, float]:
        total_weight = sum(vote.weight for vote in votes)
        weights_by_agent = {vote.voter_agent_id: vote.weight for vote in votes}
        # 得到诚实结点的权重和拜占庭结点的权重
        honest_weight = sum(
            weights_by_agent.get(agent_id, 0.0) for agent_id in self.honest_agents
        )
        byzantine_weight = sum(
            weights_by_agent.get(agent_id, 0.0) for agent_id in self.byzantine_agents
        )
        unknown_weight = max(total_weight - honest_weight - byzantine_weight, 0.0)
        
        # 安全余量，如果没有设置则使用 epsilon_ratio * total_weight
        epsilon = self.epsilon
        if epsilon is None:
            epsilon = self.epsilon_ratio * total_weight

        if self.use_dynamic_estimate or not (self.honest_agents or self.byzantine_agents):
            max_weight = max(weights_by_agent.values(), default=0.0)
            qc = max(total_weight / 2.0 + max_weight, 2.0 * total_weight / 3.0)
            formula = "dynamic_estimate"
        else:
            qc = (honest_weight + 2.0 * byzantine_weight) / 2.0 + epsilon
            formula = "known_honest_byzantine"

        return {
            "qc": qc,
            "honest_weight": honest_weight,
            "byzantine_weight": byzantine_weight,
            "unknown_weight": unknown_weight,
            "epsilon": epsilon,
            "epsilon_ratio": self.epsilon_ratio,
            "total_weight": total_weight,
            "formula": formula,
        }

    def weight_for(self, agent_id: str) -> float:
        return self.agent_weights.get(agent_id, 1.0)

    def _validate_weights(
        self, weights: Mapping[str, float]
    ) -> Dict[str, float]:
        validated: Dict[str, float] = {}
        for agent_id, weight in weights.items():
            if float(weight) < 0:
                raise ValueError(f"Weight for {agent_id} cannot be negative.")
            validated[str(agent_id)] = float(weight)
        return validated
