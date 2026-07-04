"""Shared-memory adapter for memory proposal infrastructure."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from marble.consensus.models import (
    ConsensusDecision,
    MemoryProposal,
    VerificationVector,
)
from marble.memory.shared_memory import SharedMemory


class ConsensusMemory(SharedMemory):
    """SharedMemory-compatible store for proposals and verification records."""

    PROPOSALS_KEY = "memory_proposals"
    VERIFICATIONS_KEY = "memory_proposal_verifications"
    DECISIONS_KEY = "memory_proposal_decisions"
    COMMITTED_KEY = "committed_memory_proposals"

    def __init__(self) -> None:
        super().__init__()
        self.storage[self.PROPOSALS_KEY] = {}
        self.storage[self.VERIFICATIONS_KEY] = {}
        self.storage[self.DECISIONS_KEY] = {}
        self.storage[self.COMMITTED_KEY] = {}

    def update(self, key: str, information: Any) -> None:
        """Store arbitrary shared memory or a proposal keyed by proposal_id."""
        if isinstance(information, MemoryProposal):
            self.store_proposal(information)
            return
        super().update(key, information)

    def store_proposal(self, proposal: MemoryProposal) -> None:
        with self.lock:
            proposals = self.storage.setdefault(self.PROPOSALS_KEY, {})
            proposals[proposal.proposal_id] = proposal

    def retrieve_proposal(self, proposal_id: str) -> Optional[MemoryProposal]:
        with self.lock:
            proposals = self.storage.get(self.PROPOSALS_KEY, {})
            proposal = proposals.get(proposal_id)
            return proposal if isinstance(proposal, MemoryProposal) else None

    def retrieve_proposals(
        self, task_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[MemoryProposal]:
        with self.lock:
            proposals = list(self.storage.get(self.PROPOSALS_KEY, {}).values())
        filtered = [
            proposal for proposal in proposals if isinstance(proposal, MemoryProposal)
        ]
        if task_id is not None:
            filtered = [
                proposal for proposal in filtered if proposal.task_id == task_id
            ]
        if agent_id is not None:
            filtered = [
                proposal for proposal in filtered if proposal.agent_id == agent_id
            ]
        return filtered

    def append_verification(
        self,
        proposal_id: str,
        verification: Union[VerificationVector, Dict[str, Any]],
    ) -> None:
        with self.lock:
            verifications = self.storage.setdefault(self.VERIFICATIONS_KEY, {})
            verifications.setdefault(proposal_id, []).append(verification)

    def retrieve_verifications(
        self, proposal_id: str
    ) -> List[Union[VerificationVector, Dict[str, Any]]]:
        with self.lock:
            return list(
                self.storage.get(self.VERIFICATIONS_KEY, {}).get(proposal_id, [])
            )

    def store_decision(self, decision: ConsensusDecision) -> None:
        with self.lock:
            decisions = self.storage.setdefault(self.DECISIONS_KEY, {})
            decisions[decision.proposal_id] = decision

    def retrieve_decision(self, proposal_id: str) -> Optional[ConsensusDecision]:
        with self.lock:
            decisions = self.storage.get(self.DECISIONS_KEY, {})
            decision = decisions.get(proposal_id)
            return decision if isinstance(decision, ConsensusDecision) else None

    def commit_proposal(self, proposal: MemoryProposal) -> None:
        with self.lock:
            committed = self.storage.setdefault(self.COMMITTED_KEY, {})
            committed[proposal.proposal_id] = proposal

    def retrieve_committed(
        self, task_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> List[MemoryProposal]:
        with self.lock:
            proposals = list(self.storage.get(self.COMMITTED_KEY, {}).values())
        committed = [
            proposal for proposal in proposals if isinstance(proposal, MemoryProposal)
        ]
        if task_id is not None:
            committed = [
                proposal for proposal in committed if proposal.task_id == task_id
            ]
        if agent_id is not None:
            committed = [
                proposal for proposal in committed if proposal.agent_id == agent_id
            ]
        return committed
