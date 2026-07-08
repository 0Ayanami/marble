"""Configuration contracts for MultiAgentBench experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


def _as_dict(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass
class FisherIDAExperimentConfig:
    """Reserved Fisher-IDA optimizer configuration for Smart Quorum."""

    enabled: bool = False
    alpha_initial: float = 0.6
    beta_initial: float = 0.4
    M: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FisherIDAExperimentConfig":
        data = _as_dict(data)
        return cls(
            enabled=bool(data.get("enabled", False)),
            alpha_initial=float(data.get("alpha_initial", data.get("alpha", 0.6))),
            beta_initial=float(data.get("beta_initial", data.get("beta", 0.4))),
            M=int(data.get("M", data.get("rounds", 1))),
            parameters=_as_dict(data.get("parameters")),
        )

    def validate(self) -> None:
        if not 0.0 <= self.alpha_initial <= 1.0:
            raise ValueError("proposal.fisher_ida.alpha_initial must be in [0, 1].")
        if not 0.0 <= self.beta_initial <= 1.0:
            raise ValueError("proposal.fisher_ida.beta_initial must be in [0, 1].")
        if abs((self.alpha_initial + self.beta_initial) - 1.0) > 1e-6:
            raise ValueError(
                "proposal.fisher_ida.alpha_initial + beta_initial must equal 1."
            )
        if self.M <= 0:
            raise ValueError("proposal.fisher_ida.M must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha_initial": self.alpha_initial,
            "beta_initial": self.beta_initial,
            "M": self.M,
            "parameters": self.parameters,
        }


@dataclass
class ExperimentSettings:
    """Top-level metadata and output settings."""

    name: str = "mab_experiment"
    output_root: str = "experiments"
    run_id: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSettings":
        data = _as_dict(data)
        return cls(
            name=str(data.get("name", "mab_experiment")),
            output_root=str(data.get("output_root", "experiments")),
            run_id=data.get("run_id"),
            seed=data.get("seed"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "output_root": self.output_root,
            "run_id": self.run_id,
            "seed": self.seed,
        }


@dataclass
class AgentExperimentConfig:
    """Agent count, identity, model, and capability configuration."""

    total: int
    model: Any = "gpt-3.5-turbo"
    capability_coefficient: float = 1.0
    model_capability_coefficients: Dict[str, float] = field(default_factory=dict)
    profiles: List[str] = field(default_factory=list)
    agent_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    num_byzantine: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentExperimentConfig":
        data = _as_dict(data)
        model_groups = _as_dict(data.get("models"))
        capabilities = _as_dict(
            data.get("model_capability_coefficients", data.get("model_capabilities"))
        )
        total = int(data.get("total", data.get("agent_count", 0)))
        return cls(
            total=total,
            model=data.get("model", data.get("llm", model_groups.get("default", "gpt-3.5-turbo"))),
            capability_coefficient=float(
                data.get(
                    "capability_coefficient",
                    data.get("default_capability_coefficient", 1.0),
                )
            ),
            model_capability_coefficients={
                str(model): float(coefficient)
                for model, coefficient in capabilities.items()
            },
            profiles=list(data.get("profiles", [])),
            agent_overrides={
                str(agent_id): dict(config)
                for agent_id, config in _as_dict(data.get("agent_overrides")).items()
            },
            num_byzantine=int(
                data.get("num_byzantine", data.get("byzantine", data.get("byzantine_count", 0)))
            ),
        )

    def validate(self) -> None:
        if self.total <= 0:
            raise ValueError("agents.total must be positive.")
        if self.num_byzantine != 0:
            raise ValueError("MultiAgentBench experiments in mab_ex require agents.num_byzantine = 0.")
        coefficients = {
            "agents.capability_coefficient": self.capability_coefficient,
            **{
                f"agents.model_capability_coefficients.{model}": coefficient
                for model, coefficient in self.model_capability_coefficients.items()
            },
        }
        for name, coefficient in coefficients.items():
            if float(coefficient) <= 0:
                raise ValueError(f"{name} must be positive.")
        for agent_id, override in self.agent_overrides.items():
            if "capability_coefficient" in override:
                if float(override["capability_coefficient"]) <= 0:
                    raise ValueError(
                        f"agents.agent_overrides.{agent_id}.capability_coefficient "
                        "must be positive."
                    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "model": self.model,
            "capability_coefficient": self.capability_coefficient,
            "model_capability_coefficients": self.model_capability_coefficients,
            "profiles": self.profiles,
            "agent_overrides": self.agent_overrides,
            "num_byzantine": self.num_byzantine,
        }


@dataclass
class CollaborationExperimentConfig:
    """Pre-proposal collaboration mode configuration."""

    mode: str = "role_based"
    rounds: int = 1
    max_tokens: int = 192
    temperature: float = 0.0
    disable_environment_actions: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CollaborationExperimentConfig":
        data = _as_dict(data)
        return cls(
            mode=str(data.get("mode", "role_based")),
            rounds=int(data.get("rounds", 1)),
            max_tokens=int(data.get("max_tokens", data.get("agent_max_tokens", 192))),
            temperature=float(data.get("temperature", 0.0)),
            disable_environment_actions=bool(
                data.get("disable_environment_actions", False)
            ),
            options=_as_dict(data.get("options")),
        )

    def validate(self) -> None:
        if self.rounds < 0:
            raise ValueError("collaboration.rounds cannot be negative.")
        if self.max_tokens <= 0:
            raise ValueError("collaboration.max_tokens must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "rounds": self.rounds,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "disable_environment_actions": self.disable_environment_actions,
            "options": self.options,
        }


@dataclass
class ProposalExperimentConfig:
    """Proposal builder, verification, consensus, and weight settings."""

    enabled: bool = True
    builder_enabled: bool = True
    verification_enabled: bool = True
    weight_manager_enabled: bool = True
    max_rounds: int = 1
    include_proposer_as_verifier: bool = False
    verification: Dict[str, Any] = field(default_factory=dict)
    consensus: Dict[str, Any] = field(default_factory=dict)
    weight_manager: Dict[str, Any] = field(default_factory=dict)
    fisher_ida: FisherIDAExperimentConfig = field(
        default_factory=FisherIDAExperimentConfig
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProposalExperimentConfig":
        data = _as_dict(data)
        return cls(
            enabled=bool(data.get("enabled", True)),
            builder_enabled=bool(
                data.get("builder_enabled", data.get("enable_proposal_builder", True))
            ),
            verification_enabled=bool(
                data.get(
                    "verification_enabled",
                    data.get("enable_verification_engine", True),
                )
            ),
            weight_manager_enabled=bool(
                data.get(
                    "weight_manager_enabled",
                    data.get("enable_weight_manager", True),
                )
            ),
            max_rounds=int(data.get("max_rounds", 1)),
            include_proposer_as_verifier=bool(
                data.get("include_proposer_as_verifier", False)
            ),
            verification=_as_dict(data.get("verification")),
            consensus=_as_dict(data.get("consensus", data.get("algorithm"))),
            weight_manager=_as_dict(data.get("weight_manager")),
            fisher_ida=FisherIDAExperimentConfig.from_dict(
                _as_dict(data.get("fisher_ida", data.get("fisher_ida_optimization")))
            ),
        )

    def validate(self) -> None:
        if self.max_rounds <= 0:
            raise ValueError("proposal.max_rounds must be positive.")
        self.fisher_ida.validate()

    def to_engine_consensus_config(
        self,
        *,
        task_id: str,
        agent_weights: Optional[Dict[str, float]] = None,
        agent_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        algorithm = dict(self.consensus)
        if "strategy" not in algorithm and "type" not in algorithm:
            algorithm["strategy"] = "majority_vote"
        algorithm["fisher_ida"] = self.fisher_ida.to_dict()
        algorithm.setdefault("alpha_initial", self.fisher_ida.alpha_initial)
        algorithm.setdefault("beta_initial", self.fisher_ida.beta_initial)
        if agent_weights and algorithm.get("strategy") == "smart_quorum":
            algorithm.setdefault("agent_weights", agent_weights)
        if agent_ids is not None:
            algorithm.setdefault("honest_agents", list(agent_ids))
            algorithm.setdefault("byzantine_agents", [])

        verification = dict(self.verification)
        if not self.verification_enabled:
            verification["type"] = "disabled"
        else:
            verification.setdefault("type", "heuristic")

        return {
            "task_id": task_id,
            "algorithm": algorithm,
            "verification": verification,
            "enable_weight_manager": self.weight_manager_enabled,
            "weight_manager": dict(self.weight_manager),
            "include_proposer_as_verifier": self.include_proposer_as_verifier,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "builder_enabled": self.builder_enabled,
            "verification_enabled": self.verification_enabled,
            "weight_manager_enabled": self.weight_manager_enabled,
            "max_rounds": self.max_rounds,
            "include_proposer_as_verifier": self.include_proposer_as_verifier,
            "verification": self.verification,
            "consensus": self.consensus,
            "weight_manager": self.weight_manager,
            "fisher_ida": self.fisher_ida.to_dict(),
        }


@dataclass
class BenchmarkExperimentConfig:
    """Benchmark selection and adapter configuration."""

    type: str = "multiagentbench"
    environment: str = "research"
    dataset: Optional[str] = None
    task_id: Optional[Any] = None
    task: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None
    defaults: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkExperimentConfig":
        data = _as_dict(data)
        return cls(
            type=str(data.get("type", "multiagentbench")),
            environment=str(data.get("environment", data.get("scenario", "research"))),
            dataset=data.get("dataset"),
            task_id=data.get("task_id", data.get("task")),
            task=_as_dict(data.get("inline_task", data.get("task_config"))),
            output_dir=data.get("output_dir"),
            defaults=_as_dict(data.get("defaults")),
            options=_as_dict(data.get("options")),
        )

    def validate(self) -> None:
        if not self.dataset and not self.task:
            raise ValueError("benchmark.dataset or benchmark.inline_task is required.")
        if self.dataset and not Path(str(self.dataset)).exists():
            raise FileNotFoundError(f"benchmark.dataset not found: {self.dataset}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "environment": self.environment,
            "dataset": self.dataset,
            "task_id": self.task_id,
            "inline_task": self.task,
            "output_dir": self.output_dir,
            "defaults": self.defaults,
            "options": self.options,
        }


@dataclass
class SweepExperimentConfig:
    """Optional sweep expansion for agent-count experiments."""

    agent_counts: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepExperimentConfig":
        data = _as_dict(data)
        return cls(agent_counts=[int(value) for value in _as_list(data.get("agent_counts"))])

    def validate(self) -> None:
        for count in self.agent_counts:
            if count <= 0:
                raise ValueError("sweep.agent_counts values must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return {"agent_counts": list(self.agent_counts)}


@dataclass
class ExperimentConfig:
    """Validated, centralized configuration for one MultiAgentBench run."""

    experiment: ExperimentSettings
    agents: AgentExperimentConfig
    collaboration: CollaborationExperimentConfig
    proposal: ProposalExperimentConfig
    benchmark: BenchmarkExperimentConfig
    sweep: SweepExperimentConfig = field(default_factory=SweepExperimentConfig)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        raw = dict(data)
        config = cls(
            experiment=ExperimentSettings.from_dict(_as_dict(data.get("experiment"))),
            agents=AgentExperimentConfig.from_dict(_as_dict(data.get("agents"))),
            collaboration=CollaborationExperimentConfig.from_dict(
                _as_dict(data.get("collaboration"))
            ),
            proposal=ProposalExperimentConfig.from_dict(_as_dict(data.get("proposal"))),
            benchmark=BenchmarkExperimentConfig.from_dict(_as_dict(data.get("benchmark"))),
            sweep=SweepExperimentConfig.from_dict(_as_dict(data.get("sweep"))),
            raw=raw,
        )
        config.validate()
        return config

    @classmethod
    def load(cls, file_path: str) -> "ExperimentConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError("Experiment config must be a YAML mapping.")
        return cls.from_dict(data)

    def validate(self) -> None:
        self.agents.validate()
        self.collaboration.validate()
        self.proposal.validate()
        self.benchmark.validate()
        self.sweep.validate()

    def with_agent_count(self, count: int) -> "ExperimentConfig":
        data = self.to_dict()
        data["agents"]["total"] = int(count)
        data["experiment"]["run_id"] = f"{self.experiment.run_id or self.experiment.name}_agent{count}"
        data["benchmark"]["output_dir"] = str(
            Path(self.benchmark.output_dir or self.experiment.output_root)
            / f"agent{count}"
        )
        return ExperimentConfig.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment.to_dict(),
            "agents": self.agents.to_dict(),
            "collaboration": self.collaboration.to_dict(),
            "proposal": self.proposal.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "sweep": self.sweep.to_dict(),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
