"""Configurable collaboration modes for MultiAgentBench proposal generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from marble.llms.model_prompting import model_prompting
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

    def run(
        self,
        *,
        engine: Any,
        config: CollaborationExperimentConfig,
        round_index: int,
    ) -> CollaborationRunResult:
        """Generate per-agent outputs for a proposal round."""
        raise NotImplementedError


def _agent_label(agent: Any) -> str:
    return f"{agent.agent_id}: {getattr(agent, 'profile', '')}"


def _peer_descriptions(agent: Any, agents: List[Any]) -> List[str]:
    return [_agent_label(peer) for peer in agents if peer.agent_id != agent.agent_id]


def _static_output(
    config: CollaborationExperimentConfig,
    agent_id: str,
) -> Optional[str]:
    static_outputs = config.options.get("static_agent_outputs", {})
    if not isinstance(static_outputs, dict) or agent_id not in static_outputs:
        return None
    return "Result from the model:" + str(static_outputs[agent_id]) + "\n"


def _message_content(message: Any) -> str:
    return str(getattr(message, "content", "") or "")


def _build_proposal_prompt(
    *,
    agent: Any,
    task: str,
    agents: List[Any],
    transcript: str = "",
    mode_name: str,
) -> str:
    transcript_section = (
        f"\n\nShared {mode_name} transcript:\n{transcript}\n" if transcript else ""
    )
    return (
        f"You are {agent.agent_id}: {getattr(agent, 'profile', '')}\n"
        f"Task: {task}\n"
        f"Other agents: {_peer_descriptions(agent, agents)}\n"
        f"{transcript_section}\n"
        "Propose one concise memory entry for this task. The memory should be "
        "verifiable, useful, safe, and suitable for the configured MARBLE "
        "proposal consensus workflow. Return only the proposed memory content."
    )


class RoleBasedCollaborationMode:
    """Role-based proposal generation using existing MARBLE agents/profiles."""

    name = "role_based"

    def run(
        self,
        *,
        engine: Any,
        config: CollaborationExperimentConfig,
        round_index: int,
    ) -> CollaborationRunResult:
        agents = engine.graph.get_all_agents()
        agent_outputs: Dict[str, str] = {}
        events: List[Dict[str, Any]] = []
        estimated_tokens = 0
        for agent in agents:
            static = _static_output(config, agent.agent_id)
            if static is not None:
                output = static
            elif config.options.get("use_agent_act", True):
                output, communication = agent.act(engine.task)
                if communication:
                    events.append(
                        {
                            "round": round_index + 1,
                            "phase": "communication",
                            "mode": self.name,
                            "agent_id": agent.agent_id,
                            "message": communication,
                        }
                    )
            else:
                prompt = _build_proposal_prompt(
                    agent=agent,
                    task=engine.task,
                    agents=agents,
                    mode_name=self.name,
                )
                response = model_prompting(
                    llm_model=agent.llm,
                    messages=[{"role": "user", "content": prompt}],
                    return_num=1,
                    max_token_num=config.max_tokens,
                    temperature=config.temperature,
                    top_p=None,
                    stream=None,
                )[0]
                content = _message_content(response)
                output = "Result from the model:" + content + "\n"
                estimated_tokens += estimate_tokens(prompt, content)

            estimated_tokens += estimate_tokens(output)
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
            events=events,
            estimated_tokens=estimated_tokens,
        )


class MultiAgentDiscussionMode:
    """M.A.D. mode: shared discussion transcript before proposal generation."""

    name = "mad"

    def run(
        self,
        *,
        engine: Any,
        config: CollaborationExperimentConfig,
        round_index: int,
    ) -> CollaborationRunResult:
        agents = engine.graph.get_all_agents()
        transcript_lines: List[str] = []
        events: List[Dict[str, Any]] = []
        estimated_tokens = 0
        for discussion_round in range(config.rounds):
            transcript = "\n".join(transcript_lines)
            for agent in agents:
                prompt = self._build_discussion_prompt(
                    agent=agent,
                    task=engine.task,
                    agents=agents,
                    transcript=transcript,
                    proposal_round=round_index,
                    discussion_round=discussion_round,
                )
                response = model_prompting(
                    llm_model=agent.llm,
                    messages=[{"role": "user", "content": prompt}],
                    return_num=1,
                    max_token_num=config.max_tokens,
                    temperature=config.temperature,
                    top_p=None,
                    stream=None,
                )[0]
                message = _message_content(response)
                estimated_tokens += estimate_tokens(prompt, message)
                transcript_line = f"{agent.agent_id}: {message}"
                transcript_lines.append(transcript_line)
                events.append(
                    {
                        "round": round_index + 1,
                        "discussion_round": discussion_round + 1,
                        "phase": "discussion",
                        "mode": self.name,
                        "agent_id": agent.agent_id,
                        "message": message,
                    }
                )

        transcript = "\n".join(transcript_lines)
        agent_outputs: Dict[str, str] = {}
        proposal_events: List[Dict[str, Any]] = []
        for agent in agents:
            static = _static_output(config, agent.agent_id)
            if static is not None:
                output = static
            elif config.options.get("use_agent_act", True):
                proposal_task = (
                    f"{engine.task}\n\n"
                    f"Shared Multi-Agent Discussion transcript:\n{transcript}\n\n"
                    "Based on the discussion, propose one concise memory entry "
                    "that is verifiable, useful, and safe."
                )
                output, communication = agent.act(proposal_task)
                if communication:
                    proposal_events.append(
                        {
                            "round": round_index + 1,
                            "phase": "communication",
                            "mode": self.name,
                            "agent_id": agent.agent_id,
                            "message": communication,
                        }
                    )
            else:
                prompt = _build_proposal_prompt(
                    agent=agent,
                    task=engine.task,
                    agents=agents,
                    transcript=transcript,
                    mode_name=self.name,
                )
                response = model_prompting(
                    llm_model=agent.llm,
                    messages=[{"role": "user", "content": prompt}],
                    return_num=1,
                    max_token_num=config.max_tokens,
                    temperature=config.temperature,
                    top_p=None,
                    stream=None,
                )[0]
                content = _message_content(response)
                output = "Result from the model:" + content + "\n"
                estimated_tokens += estimate_tokens(prompt, content)
            estimated_tokens += estimate_tokens(output)
            agent_outputs[agent.agent_id] = output
            proposal_events.append(
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
            transcript=transcript,
            events=events + proposal_events,
            estimated_tokens=estimated_tokens,
        )

    def _build_discussion_prompt(
        self,
        *,
        agent: Any,
        task: str,
        agents: List[Any],
        transcript: str,
        proposal_round: int,
        discussion_round: int,
    ) -> str:
        return (
            f"You are {agent.agent_id}: {getattr(agent, 'profile', '')}\n"
            f"Task: {task}\n"
            f"Other agents: {_peer_descriptions(agent, agents)}\n"
            f"Proposal round: {proposal_round + 1}\n"
            f"Discussion round: {discussion_round + 1}\n\n"
            f"Current shared transcript:\n{transcript or '(empty)'}\n\n"
            "Participate in a structured Multi-Agent Discussion. Add one concise "
            "contribution that helps the group build useful, verifiable, and safe "
            "memory proposals later. Do not produce a final memory proposal yet. "
            "Return only your next message to the group in at most 80 words."
        )


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
COLLABORATION_MODES.register("mad", MultiAgentDiscussionMode)
COLLABORATION_MODES.register("groupchat", MultiAgentDiscussionMode)
COLLABORATION_MODES.register("selector_groupchat", MultiAgentDiscussionMode)
COLLABORATION_MODES.register("multi_agent_discussion", MultiAgentDiscussionMode)
