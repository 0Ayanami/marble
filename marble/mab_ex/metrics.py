"""Standardized MultiAgentBench metric payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def task_score_from_result(result: Mapping[str, Any]) -> float:
    """Derive a task score from the best available evaluator signal."""
    effectiveness = result.get("effectiveness") or {}
    if isinstance(effectiveness, Mapping):
        kpi_score = effectiveness.get("kpi_accomplishment_ratio")
        if kpi_score is not None:
            return _as_float(kpi_score)
    if result.get("benchmark_score") is not None:
        return _as_float(result.get("benchmark_score"))
    return _as_float(result.get("success_rate"))


def standardize_task_metrics(
    *,
    task_id: str,
    environment: str,
    workflow: str,
    consensus: bool,
    agent_number: int,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the required per-task metric artifact."""
    task_score = task_score_from_result(result)
    completion_rate = _as_float(result.get("success_rate"))
    task_success_rate = _as_float(result.get("success_rate"))
    overall_score = task_score
    average_task_score = task_score
    metric_values = {
        "task_success_rate": task_success_rate,
        "overall_score": overall_score,
        "average_task_score": average_task_score,
        "completion_rate": completion_rate,
        "task_score": task_score,
    }
    effectiveness = result.get("effectiveness") or {}
    if isinstance(effectiveness, Mapping):
        kpi_score = effectiveness.get("kpi_accomplishment_ratio")
        if kpi_score is not None:
            metric_values["kpi_accomplishment_ratio"] = _as_float(kpi_score)
    records = [
        {
            "task_id": str(task_id),
            "environment": environment,
            "workflow": workflow,
            "consensus": consensus,
            "agent_number": agent_number,
            "metric": metric,
            "score": score,
        }
        for metric, score in metric_values.items()
    ]
    return {
        "task_id": str(task_id),
        "environment": environment,
        "workflow": workflow,
        "consensus": consensus,
        "agent_number": agent_number,
        "evaluator_source": result.get("evaluator_source", "marble_evaluator_fallback"),
        "task_success_rate": task_success_rate,
        "overall_score": overall_score,
        "average_task_score": average_task_score,
        "completion_rate": completion_rate,
        "task_scores": {str(task_id): task_score},
        "records": records,
    }


def aggregate_group_metrics(
    *,
    environment: str,
    workflow: str,
    consensus: bool,
    agent_number: int,
    task_metrics: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-task metrics for one experiment group."""
    metrics = list(task_metrics)
    task_scores = {
        str(metric["task_id"]): _as_float(
            (metric.get("task_scores") or {}).get(str(metric["task_id"]))
        )
        for metric in metrics
    }
    success_rates = [_as_float(metric.get("task_success_rate")) for metric in metrics]
    completion_rates = [_as_float(metric.get("completion_rate")) for metric in metrics]
    scores = list(task_scores.values())
    average_score = sum(scores) / len(scores) if scores else 0.0
    average_success = (
        sum(success_rates) / len(success_rates) if success_rates else 0.0
    )
    average_completion = (
        sum(completion_rates) / len(completion_rates) if completion_rates else 0.0
    )
    records: List[Dict[str, Any]] = []
    for metric in metrics:
        records.extend(list(metric.get("records") or []))
    return {
        "environment": environment,
        "workflow": workflow,
        "consensus": consensus,
        "agent_number": agent_number,
        "task_count": len(metrics),
        "task_success_rate": average_success,
        "overall_score": average_score,
        "average_task_score": average_score,
        "completion_rate": average_completion,
        "task_scores": task_scores,
        "records": records,
    }
