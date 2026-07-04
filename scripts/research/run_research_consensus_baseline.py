"""Run the majority-vote memory consensus baseline in ResearchEnvironment."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marble.configs.config import Config
from marble.engine import Engine
from marble.llms.model_prompting import load_provider_env


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_config(output_path: Path) -> Config:
    model = os.environ.get("MODEL", "deepseek-v4-flash")
    base_url = os.environ.get("BASE_URL", "https://api.deepseek.com")
    return Config(
        {
            "environment": {
                "type": "Research",
                "name": "Research Consensus Baseline Environment",
                "max_iterations": 1,
            },
            "agents": [
                {
                    "agent_id": "researcher_literature",
                    "profile": (
                        "Focus on literature review and identifying gaps in "
                        "multi-agent memory systems."
                    ),
                    "type": "BaseAgent",
                    "llm": model,
                },
                {
                    "agent_id": "researcher_security",
                    "profile": (
                        "Focus on safety, prompt-injection risks, and memory "
                        "poisoning in multi-agent systems."
                    ),
                    "type": "BaseAgent",
                    "llm": model,
                },
                {
                    "agent_id": "researcher_methods",
                    "profile": (
                        "Focus on practical experimental design and baseline "
                        "consensus mechanisms."
                    ),
                    "type": "BaseAgent",
                    "llm": model,
                },
            ],
            "relationships": [
                ["researcher_literature", "researcher_security", "collaborate with"],
                ["researcher_literature", "researcher_methods", "collaborate with"],
                ["researcher_security", "researcher_methods", "collaborate with"],
            ],
            "memory": {"type": "ConsensusMemory"},
            "coordinate_mode": "consensus",
            "task": {
                "task_id": "research_consensus_baseline",
                "content": (
                    "In ResearchEnvironment, propose one concise memory that helps "
                    "design a research baseline for consensus-based memory in "
                    "multi-agent systems. Focus on verifiable, useful, and safe "
                    "research information. Avoid external tool calls unless they "
                    "are strictly necessary."
                ),
            },
            "metrics": {"evaluate_llm": model},
            "engine_planner": {},
            "consensus": {
                "task_id": "research_consensus_baseline",
                "algorithm": {
                    "confidence_threshold": 0.6,
                    "majority_threshold": 0.5,
                    "strict_majority": True,
                    "minimum_votes": 1,
                    "hard_fail_dimensions": ["security"],
                },
                "verification": {
                    "type": "llm",
                    "model": model,
                    "base_url": base_url,
                    "fallback_to_heuristic": True,
                },
                "enable_weight_manager": True,
                "include_proposer_as_verifier": False,
            },
            "output": {"file_path": str(output_path), "format": "jsonl"},
        }
    )


def _save_explicit_result(engine: Engine, output_path: Path) -> Path:
    result_path = output_path.with_suffix(".summary.json")
    memory = engine.memory
    proposals = memory.retrieve_proposals(task_id="research_consensus_baseline")
    committed = memory.retrieve_committed(task_id="research_consensus_baseline")
    decisions = [
        memory.retrieve_decision(proposal.proposal_id) for proposal in proposals
    ]
    summary: Dict[str, Any] = {
        "environment": engine.environment.__class__.__name__,
        "model": os.environ.get("MODEL"),
        "base_url": os.environ.get("BASE_URL"),
        "task_id": "research_consensus_baseline",
        "proposal_count": len(proposals),
        "committed_count": len(committed),
        "proposals": [proposal.to_dict() for proposal in proposals],
        "decisions": [
            decision.to_dict() for decision in decisions if decision is not None
        ],
        "committed_memory": [proposal.to_dict() for proposal in committed],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_path


def main() -> None:
    os.environ["MARBLE_LLM_PROVIDER"] = "deepseek"
    load_provider_env(preferred_provider="deepseek")

    timestamp = _timestamp()
    output_path = Path("result") / f"research_consensus_baseline_{timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = _build_config(output_path)
    engine = Engine(config)
    # Keep this baseline focused on memory consensus. ResearchEnvironment is used,
    # while external research tools are disabled to avoid optional dependency noise.
    engine.environment.action_handler_descriptions = {}
    engine.start()
    summary_path = _save_explicit_result(engine, output_path)
    print(f"Saved Engine JSONL: {output_path}")
    print(f"Saved explicit summary: {summary_path}")


if __name__ == "__main__":
    main()
