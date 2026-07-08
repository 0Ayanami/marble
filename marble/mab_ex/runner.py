"""Unified MultiAgentBench experiment runner entry point."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from marble.engine import Engine
from marble.mab_ex.benchmarks import BENCHMARK_ADAPTERS
from marble.mab_ex.collaboration import COLLABORATION_MODES
from marble.mab_ex.config import ExperimentConfig
from marble.mab_ex.metrics import aggregate_group_metrics
from marble.mab_ex.proposal import build_consensus_layer
from marble.mab_ex.results import ExperimentPaths, ExperimentResultWriter
from marble.memory.consensus_memory import ConsensusMemory


@dataclass(frozen=True)
class ExperimentRunResult:
    """Paths and high-level statistics from one experiment run."""

    paths: ExperimentPaths
    benchmark_result: Dict[str, Any]
    proposal_records: List[Dict[str, Any]]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": self.paths.to_dict(),
            "benchmark_result": self.benchmark_result,
            "proposal_record_count": len(self.proposal_records),
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class ExperimentSweepResult:
    """Aggregate result for a benchmark sweep over multiple cases."""

    sweep_dir: Path
    summary_path: Path
    metrics_path: Path
    progress_path: Path
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sweep_dir": str(self.sweep_dir),
            "summary_path": str(self.summary_path),
            "metrics_path": str(self.metrics_path),
            "progress_path": str(self.progress_path),
            "task_count": len(self.results),
            "results": self.results,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class AgentCountSweepResult:
    """Aggregate result for a sweep over configured agent counts."""

    results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"agent_count_results": self.results}


class ExperimentRunner:
    """Coordinates benchmark loading, collaboration, proposal, and saving."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @classmethod
    def from_config_path(cls, config_path: str) -> "ExperimentRunner":
        return cls(ExperimentConfig.load(config_path))

    def run(self) -> ExperimentRunResult:
        start = time.perf_counter()
        writer = ExperimentResultWriter(self.config)
        writer.save_config()
        os.environ["MARBLE_LOG_PATH"] = str(writer.paths.logs_dir / "app.log")

        adapter = BENCHMARK_ADAPTERS.create(self.config.benchmark.type)
        case = adapter.load_case(self.config)
        engine_config = adapter.build_engine_config(
            case=case,
            config=self.config,
            proposal_output_path=writer.paths.proposal_result_path,
        )
        engine = Engine(engine_config)
        if self.config.collaboration.disable_environment_actions:
            engine.environment.action_handler_descriptions = {}

        collaboration_mode = COLLABORATION_MODES.create(
            self.config.collaboration.mode
        )

        proposal_records: List[Dict[str, Any]] = []
        trace_metrics: Dict[str, Any] = {
            "collaboration_seconds": 0.0,
            "consensus_seconds": 0.0,
            "rounds": [],
        }
        total_estimated_tokens = 0
        committed_memory: List[Dict[str, Any]] = []
        if self.config.proposal.enabled:
            proposal_records, total_estimated_tokens, trace_metrics = self._run_proposals(
                engine=engine,
                case_task_id=case.task_id,
                collaboration_mode=collaboration_mode,
            )
            if isinstance(engine.memory, ConsensusMemory):
                committed_memory = [
                    proposal.to_dict()
                    for proposal in engine.memory.retrieve_committed(
                        task_id=case.task_id
                    )
                ]
        else:
            collaboration_start = time.perf_counter()
            collaboration_result = collaboration_mode.run(
                engine=engine,
                config=self.config.collaboration,
                round_index=0,
            )
            collaboration_seconds = time.perf_counter() - collaboration_start
            trace_metrics["collaboration_seconds"] = collaboration_seconds
            trace_metrics["rounds"].append(
                {
                    "round": 1,
                    "collaboration_seconds": collaboration_seconds,
                    "consensus_seconds": 0.0,
                    "agent_output_count": len(collaboration_result.agent_outputs),
                }
            )
            total_estimated_tokens = collaboration_result.estimated_tokens
            proposal_records = [
                {
                    "round": 1,
                    "proposal_enabled": False,
                    "agent_outputs": collaboration_result.agent_outputs,
                    "collaboration": collaboration_result.to_dict(),
                    "collaboration_events": collaboration_result.events,
                }
            ]

        engine.evaluator.metrics["task_completion"].append(
            1 if committed_memory or not self.config.proposal.enabled else 0
        )
        engine.evaluator.metrics["token_consumption"].append(total_estimated_tokens)
        engine.evaluator.finalize()

        runtime_seconds = time.perf_counter() - start
        benchmark_result = adapter.evaluate(
            case=case,
            config=self.config,
            engine=engine,
            proposal_records=proposal_records,
            committed_memory=committed_memory,
            runtime_seconds=runtime_seconds,
            estimated_tokens=total_estimated_tokens,
            trace_metrics=trace_metrics,
        )
        benchmark_result["paths"] = writer.paths.to_dict()
        benchmark_result["experiment_name"] = self.config.experiment.name
        benchmark_result["agent_count"] = self.config.agents.total

        metrics = benchmark_result["standard_metrics"]
        summary = self._summary_payload(
            benchmark_result=benchmark_result,
            metrics=metrics,
        )
        writer.save_proposal_results(proposal_records)
        writer.save_benchmark_result(benchmark_result)
        writer.save_metrics(metrics)
        writer.save_summary(summary)
        return ExperimentRunResult(
            paths=writer.paths,
            benchmark_result=benchmark_result,
            proposal_records=proposal_records,
            metrics=metrics,
        )

    def _run_proposals(
        self,
        *,
        engine: Engine,
        case_task_id: str,
        collaboration_mode: Any,
    ) -> tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        if not isinstance(engine.memory, ConsensusMemory):
            engine.memory = ConsensusMemory()
            for agent in engine.graph.get_all_agents():
                agent.shared_memory = engine.memory

        records: List[Dict[str, Any]] = []
        estimated_tokens = 0
        trace_metrics: Dict[str, Any] = {
            "collaboration_seconds": 0.0,
            "consensus_seconds": 0.0,
            "rounds": [],
        }
        consensus_layer = None
        if self.config.proposal.builder_enabled:
            agent_models = {
                str(agent_config["agent_id"]): str(agent_config.get("llm", ""))
                for agent_config in engine.config.agents
                if agent_config.get("llm")
            }
            consensus_layer = build_consensus_layer(
                memory=engine.memory,
                proposal_config=self.config.proposal,
                consensus_config=engine.config.consensus,
                agent_ids=[
                    agent.agent_id for agent in engine.graph.get_all_agents()
                ],
                agent_models=agent_models,
            )

        for round_index in range(self.config.proposal.max_rounds):
            collaboration_start = time.perf_counter()
            collaboration_result = collaboration_mode.run(
                engine=engine,
                config=self.config.collaboration,
                round_index=round_index,
            )
            collaboration_seconds = time.perf_counter() - collaboration_start
            trace_metrics["collaboration_seconds"] += collaboration_seconds
            estimated_tokens += collaboration_result.estimated_tokens
            if consensus_layer is None:
                trace_metrics["rounds"].append(
                    {
                        "round": round_index + 1,
                        "collaboration_seconds": collaboration_seconds,
                        "consensus_seconds": 0.0,
                        "agent_output_count": len(collaboration_result.agent_outputs),
                    }
                )
                records.append(
                    {
                        "round": round_index + 1,
                        "proposal_builder_enabled": False,
                        "agent_outputs": collaboration_result.agent_outputs,
                        "collaboration": collaboration_result.to_dict(),
                        "collaboration_events": collaboration_result.events,
                    }
                )
                continue

            consensus_start = time.perf_counter()
            result_start = len(consensus_layer.completed_results)
            event_start = len(consensus_layer.events)
            for agent_id, output in collaboration_result.agent_outputs.items():
                consensus_layer.submit_proposal(
                    task_id=case_task_id,
                    task_description=engine.task,
                    proposer_agent_id=agent_id,
                    output=output,
                    metadata={
                        "round": round_index + 1,
                        "collaboration_mode": self.config.collaboration.mode,
                    },
                )
            consensus_layer.drain()
            consensus_seconds = time.perf_counter() - consensus_start
            trace_metrics["consensus_seconds"] += consensus_seconds
            trace_metrics["rounds"].append(
                {
                    "round": round_index + 1,
                    "collaboration_seconds": collaboration_seconds,
                    "consensus_seconds": consensus_seconds,
                    "agent_output_count": len(collaboration_result.agent_outputs),
                    "completed_proposals": (
                        len(consensus_layer.completed_results) - result_start
                    ),
                }
            )
            consensus_events = consensus_layer.events[event_start:]
            results = consensus_layer.completed_results[result_start:]
            for result in results:
                result_data = result.to_dict()
                result_data["round"] = round_index + 1
                result_data["collaboration_mode"] = self.config.collaboration.mode
                result_data["collaboration_transcript"] = (
                    collaboration_result.transcript
                )
                result_data["timing"] = {
                    "round": round_index + 1,
                    "collaboration_seconds": collaboration_seconds,
                    "consensus_seconds": consensus_seconds,
                }
                result_data["collaboration_events"] = collaboration_result.events
                result_data["consensus_layer_events"] = consensus_events
                records.append(result_data)
        return records, estimated_tokens, trace_metrics

    def _summary_payload(
        self,
        *,
        benchmark_result: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "experiment_name": self.config.experiment.name,
            "task_id": benchmark_result.get("task_id"),
            "environment": self.config.benchmark.environment,
            "workflow": self.config.collaboration.mode,
            "consensus": self.config.proposal.enabled,
            "agent_number": self.config.agents.total,
            "task_success_rate": metrics.get("task_success_rate"),
            "overall_score": metrics.get("overall_score"),
            "average_task_score": metrics.get("average_task_score"),
            "completion_rate": metrics.get("completion_rate"),
            "proposal_count": benchmark_result.get("proposal_count"),
            "committed_count": benchmark_result.get("committed_count"),
            "runtime_seconds": benchmark_result.get("runtime_seconds"),
        }


class ExperimentSweepRunner:
    """Runs the configured experiment across the selected benchmark cases."""

    def __init__(self, config: ExperimentConfig, *, ignore_task_id: bool = False) -> None:
        self.config = config
        self.ignore_task_id = ignore_task_id

    @classmethod
    def from_config_path(
        cls,
        config_path: str,
        *,
        ignore_task_id: bool = False,
    ) -> "ExperimentSweepRunner":
        return cls(ExperimentConfig.load(config_path), ignore_task_id=ignore_task_id)

    def run_all_cases(self) -> ExperimentSweepResult:
        adapter = BENCHMARK_ADAPTERS.create(self.config.benchmark.type)
        list_config = self._config_without_task_id() if self.ignore_task_id else self.config
        case_ids = adapter.list_case_ids(list_config)
        sweep_dir = self._create_sweep_dir()
        progress_path = sweep_dir / "sweep_progress.jsonl"
        summary_path = sweep_dir / "summary.json"
        metrics_path = sweep_dir / "metrics.json"

        results: List[Dict[str, Any]] = self._load_progress(progress_path)
        task_metrics: List[Dict[str, Any]] = self._load_task_metrics(results)
        completed_ids = {str(result.get("task_id")) for result in results}
        started_at = time.time()
        for index, case_id in enumerate(case_ids, start=1):
            if str(case_id) in completed_ids:
                continue
            case_config = self._config_for_case(case_id)
            run_result = ExperimentRunner(case_config).run()
            benchmark = run_result.benchmark_result
            task_metrics.append(run_result.metrics)
            record = {
                "index": index,
                "task_id": case_id,
                "run_dir": str(run_result.paths.run_dir),
                "proposal_result_path": str(run_result.paths.proposal_result_path),
                "benchmark_result_path": str(run_result.paths.benchmark_result_path),
                "metrics_path": str(run_result.paths.metrics_path),
                "summary_path": str(run_result.paths.summary_path),
                "success_rate": benchmark.get("success_rate"),
                "overall_score": run_result.metrics.get("overall_score"),
                "task_score": (run_result.metrics.get("task_scores") or {}).get(str(case_id)),
                "completion_rate": run_result.metrics.get("completion_rate"),
                "proposal_count": benchmark.get("proposal_count"),
                "committed_count": benchmark.get("committed_count"),
                "accepted_count": benchmark.get("accepted_count"),
                "rejected_count": benchmark.get("rejected_count"),
                "acceptance_rate": benchmark.get("acceptance_rate"),
                "token_consumption": benchmark.get("token_consumption"),
                "runtime_seconds": benchmark.get("runtime_seconds"),
                "kpi_accomplishment_ratio": benchmark.get("effectiveness", {}).get(
                    "kpi_accomplishment_ratio"
                ),
                "extra_time_cost_seconds": benchmark.get("efficiency", {}).get(
                    "extra_time_cost_seconds"
                ),
                "message_count": benchmark.get("efficiency", {}).get(
                    "message_count"
                ),
            }
            results.append(record)
            completed_ids.add(str(case_id))
            with progress_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._write_summary(
                summary_path=summary_path,
                metrics_path=metrics_path,
                case_ids=case_ids,
                results=results,
                task_metrics=task_metrics,
                started_at=started_at,
                completed=False,
            )

        group_metrics = self._write_summary(
            summary_path=summary_path,
            metrics_path=metrics_path,
            case_ids=case_ids,
            results=results,
            task_metrics=task_metrics,
            started_at=started_at,
            completed=True,
        )
        return ExperimentSweepResult(
            sweep_dir=sweep_dir,
            summary_path=summary_path,
            metrics_path=metrics_path,
            progress_path=progress_path,
            results=results,
            metrics=group_metrics,
        )

    def _config_for_case(self, case_id: str) -> ExperimentConfig:
        data = copy.deepcopy(self.config.to_dict())
        data["benchmark"]["task_id"] = case_id
        base_run_id = (
            self.config.experiment.run_id
            or self.config.experiment.name
            or "experiment"
        )
        safe_case_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(case_id)
        )
        data["experiment"]["run_id"] = f"{base_run_id}_task_{safe_case_id}"
        return ExperimentConfig.from_dict(data)

    def _create_sweep_dir(self) -> Path:
        output_root = Path(
            self.config.benchmark.output_dir or self.config.experiment.output_root
        )
        base_run_id = (
            self.config.experiment.run_id
            or self.config.experiment.name
            or "experiment"
        )
        base = output_root / f"{base_run_id}_sweep"
        if not base.exists():
            base.mkdir(parents=True)
            return base
        if self._is_incomplete_sweep(base):
            return base

        suffix = 1
        while True:
            candidate = output_root / f"{base_run_id}_sweep_{suffix:02d}"
            if not candidate.exists():
                candidate.mkdir(parents=True)
                return candidate
            if self._is_incomplete_sweep(candidate):
                return candidate
            suffix += 1

    def _is_incomplete_sweep(self, sweep_dir: Path) -> bool:
        summary_path = sweep_dir / "summary.json"
        if not summary_path.exists():
            return True
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return True
        return not bool(summary.get("completed", False))

    def _load_progress(self, progress_path: Path) -> List[Dict[str, Any]]:
        if not progress_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        deduped: Dict[str, Dict[str, Any]] = {}
        for record in records:
            deduped[str(record.get("task_id"))] = record
        return list(deduped.values())

    def _load_task_metrics(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        for result in results:
            metrics_path = result.get("metrics_path")
            if not metrics_path:
                continue
            path = Path(str(metrics_path))
            if path.exists():
                metrics.append(json.loads(path.read_text(encoding="utf-8")))
        return metrics

    def _write_summary(
        self,
        *,
        summary_path: Path,
        metrics_path: Path,
        case_ids: List[str],
        results: List[Dict[str, Any]],
        task_metrics: List[Dict[str, Any]],
        started_at: float,
        completed: bool,
    ) -> Dict[str, Any]:
        completed_count = len(results)
        group_metrics = aggregate_group_metrics(
            environment=self.config.benchmark.environment,
            workflow=self.config.collaboration.mode,
            consensus=self.config.proposal.enabled,
            agent_number=self.config.agents.total,
            task_metrics=task_metrics,
        )
        payload = {
            "experiment_name": self.config.experiment.name,
            "environment": self.config.benchmark.environment,
            "workflow": self.config.collaboration.mode,
            "consensus": self.config.proposal.enabled,
            "agent_number": self.config.agents.total,
            "benchmark_type": self.config.benchmark.type,
            "dataset": self.config.benchmark.dataset,
            "completed": completed,
            "task_count": len(case_ids),
            "completed_task_count": completed_count,
            "remaining_task_count": len(case_ids) - completed_count,
            "task_success_rate": group_metrics["task_success_rate"],
            "overall_score": group_metrics["overall_score"],
            "average_task_score": group_metrics["average_task_score"],
            "completion_rate": group_metrics["completion_rate"],
            "total_token_consumption": sum(
                int(result.get("token_consumption") or 0) for result in results
            ),
            "total_runtime_seconds": time.time() - started_at,
            "results": results,
        }
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        metrics_path.write_text(
            json.dumps(group_metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return group_metrics

    def _config_without_task_id(self) -> ExperimentConfig:
        data = copy.deepcopy(self.config.to_dict())
        data["benchmark"]["task_id"] = None
        return ExperimentConfig.from_dict(data)


class AgentCountSweepRunner:
    """Runs the same task sweep for each configured agent count."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @classmethod
    def from_config_path(cls, config_path: str) -> "AgentCountSweepRunner":
        return cls(ExperimentConfig.load(config_path))

    def run(self) -> AgentCountSweepResult:
        counts = self.config.sweep.agent_counts or [self.config.agents.total]
        results: List[Dict[str, Any]] = []
        for count in counts:
            count_config = self.config.with_agent_count(count)
            sweep = ExperimentSweepRunner(count_config).run_all_cases()
            results.append(
                {
                    "agent_number": count,
                    "sweep_dir": str(sweep.sweep_dir),
                    "summary_path": str(sweep.summary_path),
                    "metrics_path": str(sweep.metrics_path),
                    "metrics": sweep.metrics,
                }
            )
        return AgentCountSweepResult(results=results)
