"""MultiAgentBench experiment runner infrastructure for MARBLE."""

__all__ = [
    "AgentExperimentConfig",
    "AgentCountSweepRunner",
    "BenchmarkExperimentConfig",
    "CollaborationExperimentConfig",
    "ExperimentConfig",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentSweepResult",
    "ExperimentSweepRunner",
    "ExperimentSettings",
    "FisherIDAExperimentConfig",
    "ProposalExperimentConfig",
]


def __getattr__(name: str) -> object:
    if name in {
        "AgentExperimentConfig",
        "BenchmarkExperimentConfig",
        "CollaborationExperimentConfig",
        "ExperimentConfig",
        "ExperimentSettings",
        "FisherIDAExperimentConfig",
        "ProposalExperimentConfig",
    }:
        from marble.mab_ex import config

        return getattr(config, name)
    if name in {
        "AgentCountSweepRunner",
        "ExperimentRunResult",
        "ExperimentRunner",
        "ExperimentSweepResult",
        "ExperimentSweepRunner",
    }:
        from marble.mab_ex import runner

        return getattr(runner, name)
    raise AttributeError(name)
