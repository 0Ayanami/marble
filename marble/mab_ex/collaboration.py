"""Configurable collaboration adapters for MultiAgentBench experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from marble.mab_ex.config import CollaborationExperimentConfig


def estimate_tokens(*texts: str) -> int:
    """Cheap deterministic token estimate used for experiment accounting."""
    return sum(len(text or "") for text in texts) // 4


@dataclass
class CollaborationRunResult:
    """Outputs from one collaboration/proposal-generation round."""

    agent_outputs: Dict[str, str]
    transcript: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    estimated_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_outputs": dict(self.agent_outputs),
            "transcript": self.transcript,
            "events": list(self.events),
            "estimated_tokens": self.estimated_tokens,
        }


class CollaborationMode:
    """Interface for pre-consensus collaboration strategies."""

    name: str
    coordinate_mode: str

    def run(
        self,
        *,
        engine: Any,
        config: CollaborationExperimentConfig,
        round_index: int,
    ) -> CollaborationRunResult:
        """Generate per-agent outputs for a proposal round."""
        static_result = self._static_result(
            engine=engine,
            config=config,
            round_index=round_index,
        )
        if static_result is not None:
            return static_result

        engine.current_iteration = 0
        engine.coordinate_mode = self.coordinate_mode
        engine.config.coordination_mode = self.coordinate_mode
        if self.coordinate_mode == "groupchat":
            engine.groupchat_coordinate()
        elif self.coordinate_mode == "role_based":
            engine.role_based_coordinate()
        else:
            raise ValueError(f"Unsupported collaboration coordinate mode: {self.name}")

        summary = getattr(engine, "last_run_result", {}) or {}
        agent_outputs = _extract_agent_outputs(summary)
        transcript = str(summary.get("group_transcript", ""))
        events = _events_from_summary(
            summary=summary,
            mode=self.name,
            round_index=round_index,
        )
        estimated_tokens = estimate_tokens(
            transcript,
            *[str(output) for output in agent_outputs.values()],
        )
        return CollaborationRunResult(
            agent_outputs=agent_outputs,
            transcript=transcript,
            events=events,
            estimated_tokens=estimated_tokens,
        )

    def _static_result(
        self,
        *,
        engine: Any,
        config: CollaborationExperimentConfig,
        round_index: int,
    ) -> Optional[CollaborationRunResult]:
        static_outputs = config.options.get("static_agent_outputs", {})
        if not isinstance(static_outputs, dict) or not static_outputs:
            return None

        agent_outputs: Dict[str, str] = {}
        events: List[Dict[str, Any]] = []
        for agent in engine.graph.get_all_agents():
            if agent.agent_id not in static_outputs:
                continue
            output = "Result from the model:" + str(static_outputs[agent.agent_id]) + "\n"
            agent_outputs[agent.agent_id] = output
            events.append(
                {
                    "round": round_index + 1,
                    "phase": "proposal",
                    "mode": self.name,
                    "agent_id": agent.agent_id,
                    "message": output,
                }
            )
        return CollaborationRunResult(
            agent_outputs=agent_outputs,
            transcript="\n".join(
                f"{agent_id}: {output}" for agent_id, output in agent_outputs.items()
            ),
            events=events,
            estimated_tokens=estimate_tokens(*agent_outputs.values()),
        )


class RoleBasedCollaborationMode(CollaborationMode):
    """Role-based proposal generation through Engine.role_based_coordinate."""

    name = "role_based"
    coordinate_mode = "role_based"


class GroupchatCollaborationMode(CollaborationMode):
    """Shared transcript generation through Engine.groupchat_coordinate."""

    name = "groupchat"
    coordinate_mode = "groupchat"


def _extract_agent_outputs(summary: Dict[str, Any]) -> Dict[str, str]:
    direct_outputs = summary.get("agent_outputs")
    if isinstance(direct_outputs, dict):
        return {
            str(agent_id): str(output)
            for agent_id, output in direct_outputs.items()
            if output is not None
        }

    outputs: Dict[str, str] = {}
    iterations = summary.get("iterations", [])
    if not isinstance(iterations, list):
        return outputs
    for iteration in iterations:
        if not isinstance(iteration, dict):
            continue
        task_results = iteration.get("task_results", [])
        if not isinstance(task_results, list):
            continue
        for item in task_results:
            if not isinstance(item, dict):
                continue
            if "agent_id" in item and "result" in item:
                outputs[str(item["agent_id"])] = str(item["result"])
                continue
            for agent_id, result in item.items():
                outputs[str(agent_id)] = str(result)
    return outputs


def _events_from_summary(
    *,
    summary: Dict[str, Any],
    mode: str,
    round_index: int,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    iterations = summary.get("iterations", [])
    if not isinstance(iterations, list):
        return events
    for iteration in iterations:
        if not isinstance(iteration, dict):
            continue
        task_results = iteration.get("task_results", [])
        if not isinstance(task_results, list):
            continue
        for item in task_results:
            if not isinstance(item, dict):
                continue
            if "agent_id" in item and "result" in item:
                events.append(
                    {
                        "round": round_index + 1,
                        "phase": "proposal",
                        "mode": mode,
                        "agent_id": str(item["agent_id"]),
                        "message": str(item["result"]),
                    }
                )
                continue
            for agent_id, result in item.items():
                events.append(
                    {
                        "round": round_index + 1,
                        "phase": "proposal",
                        "mode": mode,
                        "agent_id": str(agent_id),
                        "message": str(result),
                    }
                )
    return events


class CollaborationModeRegistry:
    """Registry for collaboration modes selected by experiment config."""

    def __init__(self) -> None:
        self._modes: Dict[str, Type[CollaborationMode]] = {}

    def register(self, name: str, mode_cls: Type[CollaborationMode]) -> None:
        if not name:
            raise ValueError("Collaboration mode name cannot be empty.")
        self._modes[name] = mode_cls

    def create(self, name: str) -> CollaborationMode:
        normalized = name.replace("-", "_").lower()
        try:
            return self._modes[normalized]()
        except KeyError as exc:
            available = ", ".join(sorted(self._modes))
            raise ValueError(
                f"Unknown collaboration mode '{name}'. Available: {available}"
            ) from exc

    def names(self) -> List[str]:
        return sorted(self._modes)


COLLABORATION_MODES = CollaborationModeRegistry()
COLLABORATION_MODES.register("role_based", RoleBasedCollaborationMode)
COLLABORATION_MODES.register("magentic_one", RoleBasedCollaborationMode)
COLLABORATION_MODES.register("direct", RoleBasedCollaborationMode)
COLLABORATION_MODES.register("groupchat", GroupchatCollaborationMode)
COLLABORATION_MODES.register("mad", GroupchatCollaborationMode)
COLLABORATION_MODES.register("selector_groupchat", GroupchatCollaborationMode)
COLLABORATION_MODES.register("multi_agent_discussion", GroupchatCollaborationMode)
