"""Benchmark adapters for the MultiAgentBench experiment runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type

import yaml

from marble.configs.config import Config
from marble.mab_ex.config import ExperimentConfig
from marble.mab_ex.environment import build_environment_config
from marble.mab_ex.metrics import standardize_task_metrics


DEFAULT_ROLE_PROFILES = [
    "Focus on literature review and identifying gaps in the task domain.",
    "Focus on practical experimental design and reproducible evaluation.",
    "Focus on benchmark metrics, scoring, and result analysis.",
    "Focus on data provenance, source checking, and factual consistency.",
    "Focus on coordination efficiency and communication overhead.",
    "Focus on memory retrieval quality and information usefulness.",
    "Focus on method novelty and expected contribution.",
    "Focus on clarity, structure, and final-answer quality.",
]


@dataclass
class BenchmarkCase:
    """One benchmark task normalized from an adapter-specific source."""

    benchmark_type: str
    task_id: str
    raw: Dict[str, Any]
    dataset_path: Optional[str] = None


class BenchmarkAdapter:
    """Adapter interface for benchmark-specific loading and evaluation."""

    name: str

    def list_case_ids(self, config: ExperimentConfig) -> List[str]:
        return [self.load_case(config).task_id]

    def load_case(self, config: ExperimentConfig) -> BenchmarkCase:
        raise NotImplementedError

    def build_engine_config(
        self,
        *,
        case: BenchmarkCase,
        config: ExperimentConfig,
        proposal_output_path: Path,
    ) -> Config:
        raise NotImplementedError

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        config: ExperimentConfig,
        engine: Any,
        proposal_records: List[Dict[str, Any]],
        committed_memory: List[Dict[str, Any]],
        runtime_seconds: float,
        estimated_tokens: int,
        trace_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


def _load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [data] if isinstance(data, dict) else list(data)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [data] if isinstance(data, dict) else list(data)
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records
    raise ValueError(f"Unsupported benchmark dataset format: {path.suffix}")


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _parse_task_id_selection(
    selection: Any,
    records: List[Dict[str, Any]],
) -> List[str]:
    ids = [
        str(record.get("task_id", record.get("id", index + 1)))
        for index, record in enumerate(records)
    ]
    if selection is None:
        return ids
    if isinstance(selection, (list, tuple)):
        selected: List[str] = []
        for item in selection:
            selected.extend(_parse_task_id_selection(item, records))
        return list(dict.fromkeys(selected))

    text = str(selection).strip()
    if text.lower() in {"all", "*"}:
        return ids
    if "," in text:
        selected = []
        for item in text.split(","):
            selected.extend(_parse_task_id_selection(item.strip(), records))
        return list(dict.fromkeys(selected))
    if "-" in text:
        start_text, end_text = text.split("-", 1)
        if start_text.strip().isdigit() and end_text.strip().isdigit():
            start = int(start_text)
            end = int(end_text)
            step = 1 if start <= end else -1
            wanted = {str(value) for value in range(start, end + step, step)}
            return [task_id for task_id in ids if task_id in wanted]
    return [text]


def _fill_empty_mapping(
    current: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(current)
    for key, default_value in defaults.items():
        if merged.get(key) in (None, ""):
            merged[key] = default_value
    return merged


def _fully_connected_relationships(agent_ids: Iterable[str]) -> List[List[str]]:
    ids = list(agent_ids)
    relationships: List[List[str]] = []
    for source_index, source in enumerate(ids):
        for target in ids[source_index + 1 :]:
            relationships.append([source, target, "collaborate with"])
    return relationships


class MultiAgentBenchAdapter:
    """Adapter for MultiAgentBench JSONL/YAML task records."""

    name = "multiagentbench"

    def list_case_ids(self, config: ExperimentConfig) -> List[str]:
        benchmark = config.benchmark
        if benchmark.task:
            raw = dict(benchmark.task)
            return [str(_first_non_empty(benchmark.task_id, raw.get("task_id"), "1"))]

        assert benchmark.dataset is not None
        dataset_path = Path(benchmark.dataset)
        records = _load_records(dataset_path)
        if not records:
            raise ValueError(f"No benchmark records found in {dataset_path}.")
        selected = _parse_task_id_selection(benchmark.task_id, records)
        available = {
            str(record.get("task_id", record.get("id", index + 1)))
            for index, record in enumerate(records)
        }
        missing = [task_id for task_id in selected if task_id not in available]
        if missing:
            raise ValueError(
                "Task id(s) not found in "
                f"{dataset_path}: {', '.join(missing)}"
            )
        return selected

    def load_case(self, config: ExperimentConfig) -> BenchmarkCase:
        benchmark = config.benchmark
        if benchmark.task:
            raw = dict(benchmark.task)
            task_id = str(_first_non_empty(benchmark.task_id, raw.get("task_id"), "1"))
            return BenchmarkCase(
                benchmark_type=self.name,
                task_id=task_id,
                raw=raw,
                dataset_path=None,
            )

        assert benchmark.dataset is not None
        dataset_path = Path(benchmark.dataset)
        records = _load_records(dataset_path)
        if not records:
            raise ValueError(f"No benchmark records found in {dataset_path}.")

        selected = None
        if benchmark.task_id is not None:
            wanted_ids = _parse_task_id_selection(benchmark.task_id, records)
            if len(wanted_ids) != 1:
                raise ValueError(
                    "benchmark.task_id selects multiple tasks. Use the sweep "
                    "runner or CLI --all-tasks for multi-task experiments."
                )
            wanted = wanted_ids[0]
            for record in records:
                if str(record.get("task_id", record.get("id", ""))) == wanted:
                    selected = record
                    break
            if selected is None:
                raise ValueError(
                    f"Task id {benchmark.task_id} not found in {dataset_path}."
                )
        else:
            selected = records[0]
        task_id = str(selected.get("task_id", selected.get("id", "1")))
        return BenchmarkCase(
            benchmark_type=self.name,
            task_id=task_id,
            raw=dict(selected),
            dataset_path=str(dataset_path),
        )

    def build_engine_config(
        self,
        *,
        case: BenchmarkCase,
        config: ExperimentConfig,
        proposal_output_path: Path,
    ) -> Config:
        raw = case.raw
        environment_name = str(
            _first_non_empty(
                config.benchmark.environment,
                raw.get("scenario"),
                config.benchmark.options.get("scenario"),
                "research",
            )
        )
        agent_configs = self._build_agents(raw, config)
        agent_ids = [str(agent["agent_id"]) for agent in agent_configs]
        agent_capability_coefficients = self._agent_capability_coefficients(
            agent_configs,
            config=config,
        )
        relationships = self._build_relationships(raw, agent_ids)

        environment = build_environment_config(
            environment_name=environment_name,
            raw_environment=raw.get("environment", {}),
            defaults=config.benchmark.defaults.get("environment", {}),
        )

        task = raw.get("task", {})
        if not isinstance(task, dict):
            task = {"content": str(task)}
        task = dict(task)
        task.setdefault("task_id", case.task_id)
        task.setdefault("id", case.task_id)

        metrics = raw.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metrics = dict(metrics)
        if metrics.get("evaluate_llm") in (None, ""):
            metrics["evaluate_llm"] = config.agents.model

        memory = raw.get("memory", {})
        if not isinstance(memory, dict):
            memory = {}
        memory = dict(memory)
        if config.proposal.enabled:
            memory["type"] = "ConsensusMemory"
        else:
            memory = _fill_empty_mapping(memory, {"type": "SharedMemory"})

        coordinate_mode = str(
            _first_non_empty(
                raw.get("coordinate_mode"),
                config.benchmark.defaults.get("coordinate_mode"),
                "graph",
            )
        )

        engine_data: Dict[str, Any] = {
            "environment": environment,
            "agents": agent_configs,
            "relationships": relationships,
            "memory": memory,
            "coordinate_mode": coordinate_mode,
            "task": task,
            "metrics": metrics,
            "engine_planner": raw.get("engine_planner", {}),
            "llm": config.agents.model,
            "consensus": config.proposal.to_engine_consensus_config(
                task_id=case.task_id,
                agent_weights=agent_capability_coefficients,
                agent_ids=agent_ids,
            ),
            "output": {
                "file_path": str(proposal_output_path),
                "format": "jsonl",
            },
        }
        return Config(engine_data)

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        config: ExperimentConfig,
        engine: Any,
        proposal_records: List[Dict[str, Any]],
        committed_memory: List[Dict[str, Any]],
        runtime_seconds: float,
        estimated_tokens: int,
        trace_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        decisions = []
        for record in proposal_records:
            decision = record.get("decision")
            if isinstance(decision, dict):
                decisions.append(decision)
        accepted = [item for item in decisions if item.get("result") == "pass"]
        rejected = [item for item in decisions if item.get("result") == "fail"]
        final_answer = _final_answer_from_committed(committed_memory)
        final_answer_source = "committed_memory"
        if not final_answer:
            final_answer = _final_answer_from_proposal_records(proposal_records)
            final_answer_source = "collaboration_outputs"
        kpi_error = None
        if config.benchmark.options.get("evaluate_kpi", False):
            try:
                engine.evaluator.evaluate_kpi(engine.task, final_answer)
            except Exception as exc:  # pragma: no cover - provider/runtime dependent
                kpi_error = str(exc)
        evaluator_metrics = engine.evaluator.get_metrics()
        effectiveness = _effectiveness_metrics(
            engine.evaluator.metrics,
            kpi_error=kpi_error,
        )
        efficiency = _efficiency_metrics(
            proposal_records=proposal_records,
            runtime_seconds=runtime_seconds,
            trace_metrics=trace_metrics or {},
        )
        agent_capabilities = engine.config.consensus.get("algorithm", {}).get(
            "agent_weights",
            {},
        )
        result = {
            "benchmark_type": self.name,
            "dataset": case.dataset_path,
            "task_id": case.task_id,
            "environment": config.benchmark.environment,
            "task": engine.task,
            "workflow": config.collaboration.mode,
            "collaboration_mode": config.collaboration.mode,
            "consensus": config.proposal.enabled,
            "proposal_enabled": config.proposal.enabled,
            "final_answer": final_answer,
            "final_answer_source": final_answer_source,
            "benchmark_score": None,
            "accuracy": None,
            "effectiveness": effectiveness,
            "efficiency": efficiency,
            "success_rate": evaluator_metrics["success_rate"],
            "completion_rate": evaluator_metrics["success_rate"],
            "token_consumption": evaluator_metrics["total_tokens"],
            "estimated_collaboration_tokens": estimated_tokens,
            "cost": None,
            "runtime_seconds": runtime_seconds,
            "proposal_count": len(proposal_records),
            "committed_count": len(committed_memory),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "acceptance_rate": len(accepted) / len(decisions) if decisions else 0.0,
            "agent_capability_coefficients": agent_capabilities,
            "evaluator_source": "marble_evaluator_fallback",
            "benchmark_score_note": (
                "No repository-local MultiAgentBench official evaluator was found. "
                "Research tasks are scored with MARBLE evaluator completion/KPI "
                "signals and normalized into the standard MAB metric schema."
            ),
            "evaluator_metrics": engine.evaluator.metrics,
        }
        result["standard_metrics"] = standardize_task_metrics(
            task_id=case.task_id,
            environment=config.benchmark.environment,
            workflow=config.collaboration.mode,
            consensus=config.proposal.enabled,
            agent_number=config.agents.total,
            result=result,
        )
        return result

    def _build_agents(
        self,
        raw: Mapping[str, Any],
        config: ExperimentConfig,
    ) -> List[Dict[str, Any]]:
        raw_agents = list(raw.get("agents", []))
        agents: List[Dict[str, Any]] = []
        for index in range(config.agents.total):
            if index < len(raw_agents):
                agent_config = dict(raw_agents[index])
            else:
                agent_config = {
                    "agent_id": f"agent_{index + 1:02d}",
                    "type": "BaseAgent",
                    "profile": DEFAULT_ROLE_PROFILES[
                        index % len(DEFAULT_ROLE_PROFILES)
                    ],
                }
            agent_config.setdefault("agent_id", f"agent_{index + 1:02d}")
            agent_config.setdefault("type", "BaseAgent")
            if config.agents.profiles and index < len(config.agents.profiles):
                agent_config["profile"] = config.agents.profiles[index]
            agent_config.setdefault(
                "profile",
                DEFAULT_ROLE_PROFILES[index % len(DEFAULT_ROLE_PROFILES)],
            )
            override = config.agents.agent_overrides.get(str(agent_config["agent_id"]))
            if override:
                agent_config.update(override)
            agent_config.setdefault("llm", config.agents.model)
            agents.append(agent_config)
        return agents

    def _agent_capability_coefficients(
        self,
        agent_configs: List[Dict[str, Any]],
        *,
        config: ExperimentConfig,
    ) -> Dict[str, float]:
        coefficients: Dict[str, float] = {}
        for agent_config in agent_configs:
            agent_id = str(agent_config["agent_id"])
            if "capability_coefficient" in agent_config:
                coefficient = float(agent_config["capability_coefficient"])
            else:
                model = str(agent_config.get("llm", config.agents.model))
                coefficient = float(
                    config.agents.model_capability_coefficients.get(
                        model,
                        config.agents.capability_coefficient,
                    )
                )
            if coefficient <= 0:
                raise ValueError(
                    f"Capability coefficient for {agent_id} must be positive."
                )
            coefficients[agent_id] = coefficient
        return coefficients

    def _build_relationships(
        self,
        raw: Mapping[str, Any],
        agent_ids: List[str],
    ) -> List[List[str]]:
        selected = set(agent_ids)
        relationships = []
        for relationship in raw.get("relationships", []):
            if len(relationship) != 3:
                continue
            source, target, rel_type = relationship
            if source in selected and target in selected:
                relationships.append([source, target, rel_type])
        return relationships or _fully_connected_relationships(agent_ids)


def _final_answer_from_committed(committed_memory: List[Dict[str, Any]]) -> str:
    if not committed_memory:
        return ""
    lines = []
    for index, proposal in enumerate(committed_memory, start=1):
        header = proposal.get("header", {})
        observations = proposal.get("body", {}).get("observations", [])
        descriptions = [
            str(item.get("description", ""))
            for item in observations
            if item.get("description")
        ]
        content = " ".join(descriptions) or str(header.get("proposal_summary", ""))
        lines.append(f"{index}. {header.get('agent_id', 'unknown')}: {content}")
    return "\n".join(lines)


def _final_answer_from_proposal_records(
    proposal_records: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    for record in proposal_records:
        agent_outputs = record.get("agent_outputs")
        if isinstance(agent_outputs, dict):
            for agent_id, output in agent_outputs.items():
                content = str(output).replace("Result from the model:", "").strip()
                if content:
                    lines.append(f"{agent_id}: {content}")
        proposal = record.get("proposal")
        if isinstance(proposal, dict):
            header = proposal.get("header", {})
            body = proposal.get("body", {})
            observations = body.get("observations", [])
            descriptions = [
                str(item.get("description", ""))
                for item in observations
                if isinstance(item, dict) and item.get("description")
            ]
            content = " ".join(descriptions) or str(
                header.get("proposal_summary", "")
            )
            if content:
                lines.append(f"{header.get('agent_id', 'unknown')}: {content}")
    return "\n".join(lines)


def _effectiveness_metrics(
    evaluator_metrics: Dict[str, Any],
    *,
    kpi_error: Optional[str] = None,
) -> Dict[str, Any]:
    total_milestones = int(evaluator_metrics.get("total_milestones") or 0)
    agent_kpis = {
        str(agent_id): int(count)
        for agent_id, count in dict(evaluator_metrics.get("agent_kpis") or {}).items()
    }
    agent_kpi_ratios = {
        agent_id: (count / total_milestones if total_milestones else None)
        for agent_id, count in agent_kpis.items()
    }
    accomplished = max(agent_kpis.values(), default=0)
    return {
        "metric": "KPI",
        "definition": "n_j / M, accomplished milestones per agent over total milestones",
        "total_milestones_M": total_milestones,
        "agent_accomplished_milestones_nj": agent_kpis,
        "agent_kpi_ratios_nj_over_M": agent_kpi_ratios,
        "kpi_accomplishment_ratio": (
            accomplished / total_milestones if total_milestones else None
        ),
        "kpi_error": kpi_error,
    }


def _efficiency_metrics(
    *,
    proposal_records: List[Dict[str, Any]],
    runtime_seconds: float,
    trace_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    collaboration_events: List[Dict[str, Any]] = []
    consensus_events: List[Dict[str, Any]] = []
    qc_evolvement: List[Dict[str, Any]] = []
    voting_agents_count_evolvement: List[Dict[str, Any]] = []

    for index, record in enumerate(proposal_records, start=1):
        collaboration_events.extend(record.get("collaboration_events") or [])
        consensus_events.extend(record.get("consensus_layer_events") or [])
        decision = record.get("decision") or {}
        metadata = decision.get("metadata") or {}
        proposal_id = decision.get("proposal_id") or metadata.get("proposal_id")
        qc_evolvement.append(
            {
                "proposal_index": index,
                "proposal_id": proposal_id,
                "result": decision.get("result"),
                "qc": metadata.get("qc"),
                "accept_weight": decision.get("accept_weight"),
                "total_weight": decision.get("total_weight"),
            }
        )
        votes = decision.get("votes") or []
        voting_agents_count_evolvement.append(
            {
                "proposal_index": index,
                "proposal_id": proposal_id,
                "voting_agents_count": len(votes),
                "voting_agent_ids": [
                    vote.get("voter_agent_id")
                    for vote in votes
                    if isinstance(vote, dict)
                ],
            }
        )

    message_events = [
        event
        for event in collaboration_events
        if isinstance(event, dict) and event.get("message") not in (None, "")
    ]
    proposal_message_count = sum(
        1 for event in message_events if event.get("phase") == "proposal"
    )
    message_count = len(message_events)
    interaction_count = len(collaboration_events) + len(consensus_events)
    proposal_count = len(proposal_records)
    consensus_seconds = float(trace_metrics.get("consensus_seconds") or 0.0)
    collaboration_seconds = float(trace_metrics.get("collaboration_seconds") or 0.0)
    return {
        "task_completion_time_seconds": runtime_seconds,
        "extra_time_cost_seconds": consensus_seconds,
        "collaboration_time_seconds": collaboration_seconds,
        "consensus_time_seconds": consensus_seconds,
        "message_count": message_count,
        "proposal_message_count": proposal_message_count,
        "extra_message_count": max(message_count - proposal_count, 0),
        "interaction_count": interaction_count,
        "extra_interaction_count": max(interaction_count - proposal_count, 0),
        "qc_evolvement": qc_evolvement,
        "voting_agents_count_evolvement": voting_agents_count_evolvement,
        "round_timing": list(trace_metrics.get("rounds") or []),
    }


class BenchmarkAdapterRegistry:
    """Registry for benchmark adapters selected by experiment config."""

    def __init__(self) -> None:
        self._adapters: Dict[str, Type[BenchmarkAdapter]] = {}

    def register(self, name: str, adapter_cls: Type[BenchmarkAdapter]) -> None:
        if not name:
            raise ValueError("Benchmark adapter name cannot be empty.")
        self._adapters[name] = adapter_cls

    def create(self, name: str) -> BenchmarkAdapter:
        normalized = name.replace("-", "_").lower()
        try:
            return self._adapters[normalized]()
        except KeyError as exc:
            available = ", ".join(sorted(self._adapters))
            raise ValueError(
                f"Unknown benchmark adapter '{name}'. Available: {available}"
            ) from exc

    def names(self) -> List[str]:
        return sorted(self._adapters)


BENCHMARK_ADAPTERS = BenchmarkAdapterRegistry()
BENCHMARK_ADAPTERS.register("multiagentbench", MultiAgentBenchAdapter)
BENCHMARK_ADAPTERS.register("multi_agent_bench", MultiAgentBenchAdapter)
