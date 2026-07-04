"""Consensus-memory infrastructure for MARBLE."""

from marble.memory.consensus_memory import ConsensusMemory
from marble.consensus.models import (
    DEFAULT_DIMENSION_WEIGHTS,
    VERIFICATION_DIMENSIONS,
    ConsensusDecision,
    ConsensusResult,
    ConsensusVote,
    KeyDecision,
    MemoryProposal,
    MemoryProposalBody,
    MemoryProposalHeader,
    MultiVerificationSummary,
    ProposalAction,
    ProposalDataReference,
    ProposalObservation,
    ProposalThoughts,
    ProposalVerification,
    SelfVerificationScores,
    VerificationContext,
    VerificationVector,
)
from marble.consensus.majority_vote import MajorityVoteConsensus
from marble.consensus.proposal_builder import (
    HashSignatureProvider,
    ProposalBuilder,
    SignatureProvider,
)
from marble.consensus.verification_engine import (
    HeuristicProposalEvaluator,
    LLMProposalEvaluator,
    ProposalEvaluator,
    VerificationEngine,
    load_consensus_env,
)
from marble.consensus.weight_manager import AgentWeightState, WeightManager
from marble.consensus.workflow import (
    ConsensusWorkflowResult,
    MemoryConsensusWorkflow,
)

__all__ = [
    "AgentWeightState",
    "ConsensusMemory",
    "ConsensusDecision",
    "ConsensusResult",
    "ConsensusVote",
    "ConsensusWorkflowResult",
    "DEFAULT_DIMENSION_WEIGHTS",
    "HashSignatureProvider",
    "HeuristicProposalEvaluator",
    "KeyDecision",
    "LLMProposalEvaluator",
    "MemoryProposal",
    "MemoryProposalBody",
    "MemoryProposalHeader",
    "MultiVerificationSummary",
    "MajorityVoteConsensus",
    "MemoryConsensusWorkflow",
    "ProposalAction",
    "ProposalBuilder",
    "ProposalDataReference",
    "ProposalEvaluator",
    "ProposalObservation",
    "ProposalThoughts",
    "ProposalVerification",
    "SelfVerificationScores",
    "SignatureProvider",
    "VERIFICATION_DIMENSIONS",
    "VerificationContext",
    "VerificationEngine",
    "VerificationVector",
    "WeightManager",
    "load_consensus_env",
]
