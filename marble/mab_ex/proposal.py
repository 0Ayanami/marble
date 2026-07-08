"""Proposal workflow construction for experiment runs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from marble.consensus.majority_vote import MajorityVoteConsensus
from marble.consensus.layer import ConsensusLayer
from marble.consensus.smart_quorum import SmartQuorumConsensus
from marble.consensus.verification_engine import (
    HeuristicProposalEvaluator,
    LLMProposalEvaluator,
    VerificationEngine,
)
from marble.consensus.weight_manager import WeightManager
from marble.consensus.workflow import MemoryConsensusWorkflow
from marble.consensus.models import (
    MemoryProposal,
    VerificationContext,
    VerificationVector,
)
from marble.mab_ex.config import ProposalExperimentConfig
from marble.memory.consensus_memory import ConsensusMemory


class TrustAllProposalEvaluator:
    """Verification evaluator used only when verification is disabled."""

    def evaluate(
        self,
        proposal: MemoryProposal,
        context: VerificationContext,
        verifier_agent_id: Optional[str] = None,
    ) -> VerificationVector:
        _ = proposal, context
        return VerificationVector(
            veracity=1,
            rationality=1,
            value=1,
            security=1,
            reasoning="Verification engine disabled by experiment config.",
            verifier_agent_id=verifier_agent_id,
            metadata={"evaluator": "disabled"},
        )


def build_memory_consensus_workflow(
    *,
    memory: ConsensusMemory,
    proposal_config: ProposalExperimentConfig,
    consensus_config: Dict[str, Any],
    agent_models: Optional[Dict[str, str]] = None,
    agent_capability_coefficients: Optional[Dict[str, float]] = None,
    agent_ids: Optional[list[str]] = None,
) -> MemoryConsensusWorkflow:
    """Build the reusable MARBLE memory consensus workflow from config."""
    verification_engine = VerificationEngine(
        evaluator=_build_proposal_evaluator(
            proposal_config,
            agent_models=agent_models,
        )
    )
    consensus = _build_consensus(consensus_config.get("algorithm", {}))
    weight_manager = (
        WeightManager(**proposal_config.weight_manager)
        if proposal_config.weight_manager_enabled
        else None
    )
    if weight_manager is not None:
        for agent_id in agent_ids or []:
            coefficient = (agent_capability_coefficients or {}).get(agent_id, 1.0)
            weight_manager.register_agent(
                agent_id,
                capability_coefficient=coefficient,
            )
    return MemoryConsensusWorkflow(
        memory=memory,
        verification_engine=verification_engine,
        consensus=consensus,
        weight_manager=weight_manager,
        include_proposer_as_verifier=proposal_config.include_proposer_as_verifier,
    )


def build_consensus_layer(
    *,
    memory: ConsensusMemory,
    proposal_config: ProposalExperimentConfig,
    consensus_config: Dict[str, Any],
    agent_ids: list[str],
    agent_models: Optional[Dict[str, str]] = None,
    event_bus: Optional[Any] = None,
) -> ConsensusLayer:
    """Build the event-driven consensus layer extension."""
    algorithm_config = consensus_config.get("algorithm", {})
    agent_capability_coefficients = {
        str(agent_id): float(weight)
        for agent_id, weight in algorithm_config.get("agent_weights", {}).items()
    }
    workflow = build_memory_consensus_workflow(
        memory=memory,
        proposal_config=proposal_config,
        consensus_config=consensus_config,
        agent_models=agent_models,
        agent_capability_coefficients=agent_capability_coefficients,
        agent_ids=agent_ids,
    )
    scheduler_config = algorithm_config.get(
        "scheduler",
        consensus_config.get("scheduler", {}),
    )
    return ConsensusLayer(
        workflow=workflow,
        agent_ids=agent_ids,
        event_bus=event_bus,
        min_verifiers=scheduler_config.get(
            "min_verifiers",
            algorithm_config.get("minimum_votes"),
        ),
        max_verifiers=scheduler_config.get("max_verifiers"),
    )


def _build_proposal_evaluator(
    proposal_config: ProposalExperimentConfig,
    *,
    agent_models: Optional[Dict[str, str]] = None,
) -> Any:
    verification_config = dict(proposal_config.verification)
    if not proposal_config.verification_enabled:
        return TrustAllProposalEvaluator()
    verification_type = verification_config.get("type", "heuristic")
    if verification_type == "llm":
        fallback = (
            HeuristicProposalEvaluator()
            if verification_config.get("fallback_to_heuristic", True)
            else None
        )
        return LLMProposalEvaluator(
            model=verification_config.get("model"),
            agent_models=agent_models,
            base_url=verification_config.get("base_url"),
            env_path=verification_config.get("env_path", ".env"),
            include_commented_env=verification_config.get(
                "include_commented_env",
                True,
            ),
            fallback_evaluator=fallback,
        )
    return HeuristicProposalEvaluator(
        dimension_weights=verification_config.get("dimension_weights")
    )


def _build_consensus(algorithm_config: Dict[str, Any]) -> MajorityVoteConsensus:
    strategy = algorithm_config.get(
        "strategy",
        algorithm_config.get("type", "majority_vote"),
    )
    common_kwargs = {
        "confidence_threshold": algorithm_config.get("confidence_threshold", 0.6),
        "majority_threshold": algorithm_config.get("majority_threshold", 0.5),
        "strict_majority": algorithm_config.get("strict_majority", True),
        "minimum_votes": algorithm_config.get("minimum_votes", 1),
        "hard_fail_dimensions": algorithm_config.get(
            "hard_fail_dimensions",
            ("security",),
        ),
    }
    if strategy == "smart_quorum":
        return SmartQuorumConsensus(
            agent_weights=algorithm_config.get("agent_weights", {}),
            honest_agents=algorithm_config.get("honest_agents", []),
            byzantine_agents=algorithm_config.get("byzantine_agents", []),
            epsilon_ratio=algorithm_config.get("epsilon_ratio", 0.1),
            epsilon=algorithm_config.get("epsilon"),
            use_dynamic_estimate=algorithm_config.get(
                "use_dynamic_estimate",
                False,
            ),
            fisher_ida=algorithm_config.get("fisher_ida"),
            alpha_initial=algorithm_config.get("alpha_initial", 0.6),
            beta_initial=algorithm_config.get("beta_initial", 0.4),
            **common_kwargs,
        )
    return MajorityVoteConsensus(**common_kwargs)

