"""Runnable memory proposal -> verification -> consensus workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence

from marble.consensus.majority_vote import MajorityVoteConsensus
from marble.consensus.models import (
    ConsensusDecision,
    ConsensusResult,
    MemoryProposal,
    MultiVerificationSummary,
    ProposalVerification,
    VerificationContext,
    VerificationVector,
)
from marble.consensus.proposal_builder import ProposalBuilder
from marble.consensus.weight_manager import WeightManager
from marble.consensus.verification_engine import VerificationEngine
from marble.memory.consensus_memory import ConsensusMemory


@dataclass(frozen=True)
class ConsensusWorkflowResult:
    """Full trace for one proposal processed by the consensus workflow."""

    proposal: MemoryProposal
    verifications: List[VerificationVector]
    decision: ConsensusDecision
    committed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "verifications": [vector.to_dict() for vector in self.verifications],
            "decision": self.decision.to_dict(),
            "committed": self.committed,
        }


class MemoryConsensusWorkflow:
    """Orchestrates proposal construction, verification, and consensus."""

    def __init__(
        self,
        *,
        memory: Optional[ConsensusMemory] = None,
        proposal_builder: Optional[ProposalBuilder] = None,
        verification_engine: Optional[VerificationEngine] = None,
        consensus: Optional[MajorityVoteConsensus] = None,
        weight_manager: Optional[WeightManager] = None,
        include_proposer_as_verifier: bool = False,
    ) -> None:
        self.memory = memory or ConsensusMemory()
        self.proposal_builder = proposal_builder or ProposalBuilder()
        self.verification_engine = verification_engine or VerificationEngine()
        self.consensus = consensus or MajorityVoteConsensus()
        self.weight_manager = weight_manager
        self.include_proposer_as_verifier = include_proposer_as_verifier

    def submit_agent_output(
        self,
        *,
        task_id: str,
        task_description: str,
        proposer_agent_id: str,
        output: Any,
        verifier_agent_ids: Sequence[str],
        proposal_summary: Optional[str] = None,
        parent_proposals: Optional[Sequence[str]] = None,
    ) -> ConsensusWorkflowResult:
        """Process one agent output through the full baseline flow."""
        proposal = self.proposal_builder.from_agent_output(
            task_id=task_id,
            agent_id=proposer_agent_id,
            output=output,
            proposal_summary=proposal_summary,
            parent_proposals=parent_proposals,
        )
        self.memory.store_proposal(proposal)

        verifications = self._verify_proposal(
            proposal=proposal,
            task_description=task_description,
            verifier_agent_ids=verifier_agent_ids,
        )
        self._sync_consensus_weights()
        decision = self.consensus.decide(proposal, verifications)
        finalized_proposal = self._with_consensus_result(
            proposal=proposal,
            verifications=verifications,
            decision=decision,
        )
        self.memory.store_proposal(finalized_proposal)
        self.memory.store_decision(decision)

        committed = decision.accepted
        if committed:
            self.memory.commit_proposal(finalized_proposal)
        self._update_agent_state(finalized_proposal, decision)
        return ConsensusWorkflowResult(
            proposal=finalized_proposal,
            verifications=verifications,
            decision=decision,
            committed=committed,
        )

    def run_agent_outputs(
        self,
        *,
        task_id: str,
        task_description: str,
        agent_outputs: Dict[str, Any],
    ) -> List[ConsensusWorkflowResult]:
        """Run the consensus flow over precomputed outputs keyed by agent_id."""
        agent_ids = list(agent_outputs)
        return [
            self.submit_agent_output(
                task_id=task_id,
                task_description=task_description,
                proposer_agent_id=agent_id,
                output=output,
                verifier_agent_ids=agent_ids,
            )
            for agent_id, output in agent_outputs.items()
        ]

    def run_agents(
        self,
        *,
        task_id: str,
        task_description: str,
        agents: Iterable[Any],
    ) -> List[ConsensusWorkflowResult]:
        """Run MARBLE agents and process each output through consensus."""
        agent_list = list(agents)
        agent_ids = [agent.agent_id for agent in agent_list]
        results: List[ConsensusWorkflowResult] = []
        for agent in agent_list:
            output, _communication = agent.act(task_description)
            results.append(
                self.submit_agent_output(
                    task_id=task_id,
                    task_description=task_description,
                    proposer_agent_id=agent.agent_id,
                    output=output,
                    verifier_agent_ids=agent_ids,
                )
            )
        return results

    def _verify_proposal(
        self,
        *,
        proposal: MemoryProposal,
        task_description: str,
        verifier_agent_ids: Sequence[str],
    ) -> List[VerificationVector]:
        related = self.memory.retrieve_committed(task_id=proposal.task_id)
        context = VerificationContext(
            task_id=proposal.task_id,
            task_description=task_description,
            related_proposals=related,
        )
        verifiers = self._effective_verifiers(proposal.agent_id, verifier_agent_ids)
        verifications = [
            self.verification_engine.evaluate(
                proposal=proposal,
                context=context,
                verifier_agent_id=verifier_agent_id,
            )
            for verifier_agent_id in verifiers
        ]
        for vector in verifications:
            self.memory.append_verification(proposal.proposal_id, vector)
        return verifications

    def _effective_verifiers(
        self, proposer_agent_id: str, verifier_agent_ids: Sequence[str]
    ) -> List[str]:
        if self.include_proposer_as_verifier:
            return list(dict.fromkeys(verifier_agent_ids))
        return [
            agent_id
            for agent_id in dict.fromkeys(verifier_agent_ids)
            if agent_id != proposer_agent_id
        ]

    def _with_consensus_result(
        self,
        *,
        proposal: MemoryProposal,
        verifications: Sequence[VerificationVector],
        decision: ConsensusDecision,
    ) -> MemoryProposal:
        weighted_scores = self._average_dimension_scores(verifications)
        verification = replace(
            proposal.verification,
            multi_verification=MultiVerificationSummary(
                weighted_scores=weighted_scores
            ),
            consensus_result=ConsensusResult(
                total_weight=decision.total_weight,
                vote_weight=decision.accept_weight,
                result=decision.result,
            ),
        )
        return replace(proposal, verification=verification)

    def _average_dimension_scores(
        self, verifications: Sequence[VerificationVector]
    ) -> Dict[str, float]:
        if not verifications:
            return {}
        dimensions = ("veracity", "rationality", "value", "security")
        return {
            dimension: sum(
                float(getattr(vector, dimension)) for vector in verifications
            )
            / len(verifications)
            for dimension in dimensions
        }

    def _sync_consensus_weights(self) -> None:
        if self.weight_manager is None:
            return
        set_agent_weights = getattr(self.consensus, "set_agent_weights", None)
        if set_agent_weights is None:
            return
        set_agent_weights(
            {
                str(snapshot["agent_id"]): float(snapshot["weight"])
                for snapshot in self.weight_manager.snapshots()
            }
        )

    def _update_agent_state(
        self,
        proposal: MemoryProposal,
        decision: ConsensusDecision,
    ) -> None:
        if self.weight_manager is None:
            return
        proposal_confidence = float(
            decision.metadata.get("proposal_confidence_score", 0.0)
        )
        self.weight_manager.record_proposal_confidence(
            proposal.agent_id,
            proposal_confidence,
        )
        for vote in decision.votes:
            self.weight_manager.record_vote_alignment(
                vote.voter_agent_id,
                vote.accept == decision.accepted,
            )
