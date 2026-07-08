"""Event-driven consensus layer for MARBLE memory proposals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

from marble.consensus.workflow import ConsensusWorkflowResult, MemoryConsensusWorkflow


@dataclass(frozen=True)
class ProposalSubmission:
    """One proposal event waiting for consensus-layer scheduling."""

    task_id: str
    task_description: str
    proposer_agent_id: str
    output: Any
    proposal_summary: Optional[str] = None
    parent_proposals: Optional[Sequence[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentActivityTracker:
    """Minimal activity view used by the consensus layer.

    MARBLE's regular workflow still owns agent execution. This tracker only
    records whether an agent can be temporarily scheduled for verification.
    """

    def __init__(self, agent_ids: Iterable[str]) -> None:
        self._agent_ids = list(dict.fromkeys(str(agent_id) for agent_id in agent_ids))
        self._busy_agents: set[str] = set()

    def mark_busy(self, agent_id: str) -> None:
        self._ensure_known(agent_id)
        self._busy_agents.add(agent_id)

    def mark_idle(self, agent_id: str) -> None:
        self._ensure_known(agent_id)
        self._busy_agents.discard(agent_id)

    def is_idle(self, agent_id: str) -> bool:
        self._ensure_known(agent_id)
        return agent_id not in self._busy_agents

    def idle_agents(self) -> List[str]:
        return [
            agent_id
            for agent_id in self._agent_ids
            if agent_id not in self._busy_agents
        ]

    def busy_agents(self) -> List[str]:
        return [
            agent_id
            for agent_id in self._agent_ids
            if agent_id in self._busy_agents
        ]

    def _ensure_known(self, agent_id: str) -> None:
        if agent_id not in self._agent_ids:
            self._agent_ids.append(agent_id)


class ConsensusLayer:
    """Event-driven extension layer above MARBLE agent workflows.

    The layer schedules temporary verification work for existing idle agents. It
    does not create verifier/reviewer agents and does not call agent workflow
    methods such as ``act``.
    """

    PROPOSAL_SUBMITTED = "consensus.proposal_submitted"
    AGENT_BUSY = "consensus.agent_busy"
    AGENT_IDLE = "consensus.agent_idle"
    CONSENSUS_COMPLETED = "consensus.completed"

    def __init__(
        self,
        *,
        workflow: MemoryConsensusWorkflow,
        agent_ids: Iterable[str],
        event_bus: Optional[Any] = None,
        layer_id: str = "consensus_layer",
        min_verifiers: Optional[int] = None,
        max_verifiers: Optional[int] = None,
        notification_recipients: Optional[Sequence[str]] = None,
        auto_subscribe: bool = True,
    ) -> None:
        if max_verifiers is not None and max_verifiers <= 0:
            raise ValueError("max_verifiers must be positive when provided.")
        if min_verifiers is not None and min_verifiers <= 0:
            raise ValueError("min_verifiers must be positive when provided.")
        if (
            min_verifiers is not None
            and max_verifiers is not None
            and min_verifiers > max_verifiers
        ):
            raise ValueError("min_verifiers cannot exceed max_verifiers.")

        self.workflow = workflow
        self.layer_id = layer_id
        self.event_bus = event_bus
        self.notification_recipients = list(notification_recipients or [])
        self.agent_activity = AgentActivityTracker(agent_ids)
        self.min_verifiers = (
            min_verifiers
            if min_verifiers is not None
            else getattr(workflow.consensus, "minimum_votes", 1)
        )
        self.max_verifiers = max_verifiers
        self.pending: Deque[ProposalSubmission] = deque()
        self.completed_results: List[ConsensusWorkflowResult] = []
        self.events: List[Dict[str, Any]] = []

        if self.event_bus is not None and auto_subscribe:
            self.event_bus.subscribe(self.layer_id, self.handle_event)

    def handle_event(self, event: Dict[str, Any]) -> Optional[ConsensusWorkflowResult]:
        """Handle a MARBLE event bus event."""
        event_type = event.get("event_type")
        if event_type in {self.PROPOSAL_SUBMITTED, "proposal_submitted"}:
            return self.submit_proposal(
                task_id=str(event["task_id"]),
                task_description=str(event.get("task_description", "")),
                proposer_agent_id=str(event["proposer_agent_id"]),
                output=event.get("output"),
                proposal_summary=event.get("proposal_summary"),
                parent_proposals=event.get("parent_proposals"),
                metadata=dict(event.get("metadata", {})),
            )
        if event_type in {self.AGENT_BUSY, "agent_busy"}:
            self.mark_agent_busy(str(event["agent_id"]))
            return None
        if event_type in {self.AGENT_IDLE, "agent_idle"}:
            self.mark_agent_idle(str(event["agent_id"]))
            return None
        return None

    def submit_proposal(
        self,
        *,
        task_id: str,
        task_description: str,
        proposer_agent_id: str,
        output: Any,
        proposal_summary: Optional[str] = None,
        parent_proposals: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConsensusWorkflowResult]:
        """Queue a proposal event and process it when enough agents are idle."""
        before_count = len(self.completed_results)
        submission = ProposalSubmission(
            task_id=task_id,
            task_description=task_description,
            proposer_agent_id=proposer_agent_id,
            output=output,
            proposal_summary=proposal_summary,
            parent_proposals=parent_proposals,
            metadata=dict(metadata or {}),
        )
        self.pending.append(submission)
        self.events.append(
            {
                "event_type": self.PROPOSAL_SUBMITTED,
                "proposer_agent_id": proposer_agent_id,
                "task_id": task_id,
                "status": "queued",
            }
        )
        self.drain()
        if len(self.completed_results) > before_count:
            return self.completed_results[before_count]
        return None

    def mark_agent_busy(self, agent_id: str) -> None:
        """Mark an existing workflow agent as unavailable for verification."""
        self.agent_activity.mark_busy(agent_id)
        self.events.append(
            {"event_type": self.AGENT_BUSY, "agent_id": agent_id}
        )

    def mark_agent_idle(self, agent_id: str) -> None:
        """Mark an existing workflow agent as available and retry the queue."""
        self.agent_activity.mark_idle(agent_id)
        self.events.append(
            {"event_type": self.AGENT_IDLE, "agent_id": agent_id}
        )
        self.drain()

    def drain(self) -> None:
        """Process queued proposals while enough idle agents are available."""
        while self.pending:
            submission = self.pending[0]
            verifier_agent_ids = self._select_verifiers(
                proposer_agent_id=submission.proposer_agent_id
            )
            if len(verifier_agent_ids) < self.min_verifiers:
                self.events.append(
                    {
                        "event_type": "consensus.proposal_waiting",
                        "proposer_agent_id": submission.proposer_agent_id,
                        "task_id": submission.task_id,
                        "idle_agent_count": len(verifier_agent_ids),
                        "required_verifier_count": self.min_verifiers,
                    }
                )
                break
            self.pending.popleft()
            self._process_submission(submission, verifier_agent_ids)

    def _select_verifiers(self, *, proposer_agent_id: str) -> List[str]:
        candidates = self.agent_activity.idle_agents()
        if not self.workflow.include_proposer_as_verifier:
            candidates = [
                agent_id for agent_id in candidates if agent_id != proposer_agent_id
            ]
        if self.max_verifiers is not None:
            return candidates[: self.max_verifiers]
        return candidates

    def _process_submission(
        self,
        submission: ProposalSubmission,
        verifier_agent_ids: Sequence[str],
    ) -> None:
        for agent_id in verifier_agent_ids:
            self.agent_activity.mark_busy(agent_id)
        self.events.append(
            {
                "event_type": "consensus.verification_started",
                "proposer_agent_id": submission.proposer_agent_id,
                "task_id": submission.task_id,
                "verifier_agent_ids": list(verifier_agent_ids),
            }
        )
        try:
            result = self.workflow.submit_agent_output(
                task_id=submission.task_id,
                task_description=submission.task_description,
                proposer_agent_id=submission.proposer_agent_id,
                output=submission.output,
                verifier_agent_ids=verifier_agent_ids,
                proposal_summary=submission.proposal_summary,
                parent_proposals=submission.parent_proposals,
            )
            self.completed_results.append(result)
            completion_event = {
                "event_type": self.CONSENSUS_COMPLETED,
                "proposal_id": result.proposal.proposal_id,
                "proposer_agent_id": result.proposal.agent_id,
                "task_id": result.proposal.task_id,
                "result": result.decision.result,
                "committed": result.committed,
                "verifier_agent_ids": list(verifier_agent_ids),
            }
            self.events.append(completion_event)
            if self.event_bus is not None and self.notification_recipients:
                self.event_bus.publish(
                    {
                        **completion_event,
                        "recipients": list(self.notification_recipients),
                    }
                )
        finally:
            for agent_id in verifier_agent_ids:
                self.agent_activity.mark_idle(agent_id)
