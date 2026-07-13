"""Post-event memory proposal -> verification -> consensus lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence

from marble.consensus.majority_vote import MajorityVoteConsensus
from marble.consensus.models import (
    ConsensusDecision,
    ConsensusResult,
    MemoryProposal,
    MultiVerificationSummary,
    ProposalVerification,
    SelfVerificationScores,
    VERIFICATION_DIMENSIONS,
    VerificationContext,
    VerificationVector,
    to_plain_data,
    utc_now_iso,
)
from marble.consensus.proposal_builder import ProposalBuilder
from marble.consensus.verification_engine import VerificationEngine
from marble.consensus.weight_manager import WeightManager
from marble.memory.consensus_memory import ConsensusMemory


@dataclass(frozen=True)
class ConsensusWorkflowResult:
    """Full trace for one proposal processed by memory consensus."""

    proposal: MemoryProposal
    verifications: List[VerificationVector]
    decision: ConsensusDecision
    committed: bool
    status: str = "completed"
    commit_status: str = "not_attempted"
    self_verification_passed: bool = False
    self_verification: Optional[VerificationVector] = None
    verifier_records: List[Dict[str, Any]] = field(default_factory=list)
    weight_snapshot_before: List[Dict[str, Any]] = field(default_factory=list)
    weight_snapshot_after: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "verifications": [vector.to_dict() for vector in self.verifications],
            "decision": self.decision.to_dict(),
            "committed": self.committed,
            "status": self.status,
            "commit_status": self.commit_status,
            "self_verification_passed": self.self_verification_passed,
            "self_verification": (
                self.self_verification.to_dict() if self.self_verification else None
            ),
            "verifier_records": to_plain_data(self.verifier_records),
            "weight_snapshot_before": to_plain_data(self.weight_snapshot_before),
            "weight_snapshot_after": to_plain_data(self.weight_snapshot_after),
            "metadata": to_plain_data(self.metadata),
            "errors": to_plain_data(self.errors),
        }


class MemoryConsensusWorkflow:
    """Runs consensus as an independent post-event service."""

    def __init__(
        self,
        *,
        memory: Optional[ConsensusMemory] = None,
        proposal_builder: Optional[ProposalBuilder] = None,
        verification_engine: Optional[VerificationEngine] = None,
        consensus: Optional[MajorityVoteConsensus] = None,
        weight_manager: Optional[WeightManager] = None,
        include_proposer_as_verifier: bool = False,
        self_confidence_threshold: float = 0.6,
    ) -> None:
        self.memory = memory or ConsensusMemory()
        self.proposal_builder = proposal_builder or ProposalBuilder()
        self.verification_engine = verification_engine or VerificationEngine()
        self.consensus = consensus or MajorityVoteConsensus()
        self.weight_manager = weight_manager
        self.include_proposer_as_verifier = include_proposer_as_verifier
        self.self_confidence_threshold = float(self_confidence_threshold)

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsensusWorkflowResult:
        """Process one completed MARBLE event output through memory consensus."""
        lifecycle_metadata = dict(metadata or {})
        lifecycle_metadata.setdefault("created_at", utc_now_iso())
        proposal = self.proposal_builder.from_agent_output(
            task_id=task_id,
            agent_id=proposer_agent_id,
            output=output,
            proposal_summary=proposal_summary,
            parent_proposals=parent_proposals,
        )
        self.memory.store_proposal(proposal)
        weight_before = self._weight_snapshots()
        self._sync_consensus_weights_from_snapshot(weight_before)

        self_vector, self_errors = self._self_verify(
            proposal=proposal,
            task_description=task_description,
        )
        proposal = self._with_self_verification(proposal, self_vector)
        self.memory.store_proposal(proposal)
        self_passed = self_vector.confidence_score >= self.self_confidence_threshold
        expected_verifiers = self._effective_verifiers(
            proposer_agent_id,
            verifier_agent_ids,
        )
        if not self_passed:
            decision = self._self_failed_decision(proposal, self_vector)
            finalized = self._with_consensus_result(
                proposal=proposal,
                decision=decision,
                verifier_records=[],
                expected_verifier_ids=expected_verifiers,
                weight_snapshot=weight_before,
            )
            self.memory.store_proposal(finalized)
            self.memory.store_decision(decision)
            self._update_agent_state(finalized, decision)
            weight_after = self._weight_snapshots()
            return ConsensusWorkflowResult(
                proposal=finalized,
                verifications=[],
                decision=decision,
                committed=False,
                status="self_verification_failed",
                commit_status="not_attempted",
                self_verification_passed=False,
                self_verification=self_vector,
                verifier_records=[],
                weight_snapshot_before=weight_before,
                weight_snapshot_after=weight_after,
                metadata=lifecycle_metadata,
                errors=self_errors,
            )

        verifications, verifier_records, verifier_errors = self._verify_proposal(
            proposal=proposal,
            task_description=task_description,
            verifier_agent_ids=verifier_agent_ids,
            weight_snapshot=weight_before,
        )
        decision = self.consensus.decide(proposal, verifications)
        finalized = self._with_consensus_result(
            proposal=proposal,
            decision=decision,
            verifier_records=verifier_records,
            expected_verifier_ids=expected_verifiers,
            weight_snapshot=weight_before,
        )
        self.memory.store_proposal(finalized)
        self.memory.store_decision(decision)

        committed = False
        commit_status = "not_attempted"
        commit_errors: List[Dict[str, Any]] = []
        if decision.accepted:
            try:
                self.memory.commit_proposal(finalized)
                committed = True
                commit_status = "committed"
            except Exception as exc:  # pragma: no cover - defensive
                commit_status = "failed"
                commit_errors.append(
                    {"stage": "commit", "error": str(exc), "type": type(exc).__name__}
                )
        self._update_agent_state(finalized, decision)
        weight_after = self._weight_snapshots()
        return ConsensusWorkflowResult(
            proposal=finalized,
            verifications=verifications,
            decision=decision,
            committed=committed,
            status="accepted" if decision.accepted else "rejected",
            commit_status=commit_status,
            self_verification_passed=True,
            self_verification=self_vector,
            verifier_records=verifier_records,
            weight_snapshot_before=weight_before,
            weight_snapshot_after=weight_after,
            metadata=lifecycle_metadata,
            errors=self_errors + verifier_errors + commit_errors,
        )

    def run_agent_outputs(
        self,
        *,
        task_id: str,
        task_description: str,
        agent_outputs: Dict[str, Any],
    ) -> List[ConsensusWorkflowResult]:
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

    def _self_verify(
        self,
        *,
        proposal: MemoryProposal,
        task_description: str,
    ) -> tuple[VerificationVector, List[Dict[str, Any]]]:
        context = VerificationContext(
            task_id=proposal.task_id,
            task_description=task_description,
            related_proposals=self.memory.retrieve_committed(task_id=proposal.task_id),
            metadata={"verification_stage": "self"},
        )
        try:
            vector = self.verification_engine.evaluate(
                proposal=proposal,
                context=context,
                verifier_agent_id=proposal.agent_id,
            )
            return vector, []
        except Exception as exc:
            return (
                VerificationVector(
                    veracity=0,
                    rationality=0,
                    value=0,
                    security=0,
                    reasoning=f"Self verification failed: {exc}",
                    verifier_agent_id=proposal.agent_id,
                    metadata={
                        "status": "failed",
                        "stage": "self_verification",
                        "error_type": type(exc).__name__,
                    },
                ),
                [
                    {
                        "stage": "self_verification",
                        "agent_id": proposal.agent_id,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                ],
            )

    def _verify_proposal(
        self,
        *,
        proposal: MemoryProposal,
        task_description: str,
        verifier_agent_ids: Sequence[str],
        weight_snapshot: Sequence[Dict[str, Any]],
    ) -> tuple[List[VerificationVector], List[Dict[str, Any]], List[Dict[str, Any]]]:
        related = self.memory.retrieve_committed(task_id=proposal.task_id)
        context = VerificationContext(
            task_id=proposal.task_id,
            task_description=task_description,
            related_proposals=related,
            metadata={"verification_stage": "multi_agent"},
        )
        verifiers = self._effective_verifiers(proposal.agent_id, verifier_agent_ids)
        weights = self._weights_by_agent(weight_snapshot)
        verifications: List[VerificationVector] = []
        records: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for verifier_agent_id in verifiers:
            try:
                vector = self.verification_engine.evaluate(
                    proposal=proposal,
                    context=context,
                    verifier_agent_id=verifier_agent_id,
                )
                verifications.append(vector)
                self.memory.append_verification(proposal.proposal_id, vector)
                records.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "verifier_agent_id": verifier_agent_id,
                        "status": "valid",
                        "weight": weights.get(verifier_agent_id, 1.0),
                        "verification": vector.to_dict(),
                        "failure_reason": "",
                    }
                )
            except Exception as exc:
                error = {
                    "stage": "multi_agent_verification",
                    "proposal_id": proposal.proposal_id,
                    "verifier_agent_id": verifier_agent_id,
                    "status": "failed",
                    "weight": weights.get(verifier_agent_id, 1.0),
                    "failure_reason": str(exc),
                    "error_type": type(exc).__name__,
                }
                records.append(error)
                errors.append(error)
                self.memory.append_verification(proposal.proposal_id, error)
        return verifications, records, errors

    def _effective_verifiers(
        self, proposer_agent_id: str, verifier_agent_ids: Sequence[str]
    ) -> List[str]:
        ids = list(dict.fromkeys(str(agent_id) for agent_id in verifier_agent_ids))
        if self.include_proposer_as_verifier:
            return ids
        return [agent_id for agent_id in ids if agent_id != proposer_agent_id]

    def _with_self_verification(
        self,
        proposal: MemoryProposal,
        vector: VerificationVector,
    ) -> MemoryProposal:
        verification = replace(
            proposal.verification,
            self_verification=SelfVerificationScores(
                veracity_score=float(vector.veracity),
                rationality_score=float(vector.rationality),
                value_score=float(vector.value),
                security_score=float(vector.security),
            ),
        )
        return replace(proposal, verification=verification)

    def _with_consensus_result(
        self,
        *,
        proposal: MemoryProposal,
        decision: ConsensusDecision,
        verifier_records: Sequence[Dict[str, Any]],
        expected_verifier_ids: Sequence[str],
        weight_snapshot: Sequence[Dict[str, Any]],
    ) -> MemoryProposal:
        metadata_summary = decision.metadata.get("multi_verification_summary", {})
        weighted_scores = (
            dict(metadata_summary)
            if isinstance(metadata_summary, dict)
            else self._weighted_dimension_scores(decision.votes)
        )
        summary = MultiVerificationSummary(weighted_scores=weighted_scores)
        verification = replace(
            proposal.verification,
            multi_verification=summary,
            consensus_result=ConsensusResult(
                total_weight=decision.total_weight,
                vote_weight=decision.accept_weight,
                result=decision.result,
            ),
        )
        return replace(proposal, verification=verification)

    def _self_failed_decision(
        self,
        proposal: MemoryProposal,
        vector: VerificationVector,
    ) -> ConsensusDecision:
        return ConsensusDecision(
            proposal_id=proposal.proposal_id,
            result="fail",
            votes=[],
            accept_count=0,
            reject_count=0,
            acceptance_ratio=0.0,
            threshold=getattr(self.consensus, "majority_threshold", 0.5),
            confidence_threshold=self.self_confidence_threshold,
            total_weight=0.0,
            accept_weight=0.0,
            metadata={
                "strategy": "self_verification_gate",
                "reason": "self_verification_failed",
                "self_confidence_score": vector.confidence_score,
                "self_confidence_threshold": self.self_confidence_threshold,
                "proposal_confidence_score": 0.0,
                "proposal_confidence_method": "self_verification_gate",
                "multi_verification_summary": {
                    dimension: 0.0 for dimension in VERIFICATION_DIMENSIONS
                },
                "consensus_result": ConsensusResult(
                    total_weight=0.0,
                    vote_weight=0.0,
                    result="fail",
                ).to_dict(),
            },
        )

    def _sync_consensus_weights_from_snapshot(
        self,
        snapshot: Sequence[Dict[str, Any]],
    ) -> None:
        set_agent_weights = getattr(self.consensus, "set_agent_weights", None)
        if set_agent_weights is None:
            return
        set_agent_weights(self._weights_by_agent(snapshot))

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

    def _weight_snapshots(self) -> List[Dict[str, Any]]:
        if self.weight_manager is None:
            return []
        return [dict(snapshot) for snapshot in self.weight_manager.snapshots()]

    @staticmethod
    def _weights_by_agent(
        snapshot: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        return {
            str(item["agent_id"]): float(item["weight"])
            for item in snapshot
            if "agent_id" in item and "weight" in item
        }

    @staticmethod
    def _weighted_dimension_scores(votes: Sequence[Any]) -> Dict[str, float]:
        total_weight = sum(
            max(float(getattr(vote, "weight", 1.0)), 0.0) for vote in votes
        )
        if total_weight <= 0.0:
            return {dimension: 0.0 for dimension in VERIFICATION_DIMENSIONS}
        return {
            dimension: sum(
                float(getattr(vote.verification, dimension))
                * max(float(getattr(vote, "weight", 1.0)), 0.0)
                for vote in votes
            )
            / total_weight
            for dimension in VERIFICATION_DIMENSIONS
        }
