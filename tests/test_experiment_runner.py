import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from marble.experiments.config import ExperimentConfig
from marble.experiments.benchmarks import BENCHMARK_ADAPTERS
from marble.experiments.runner import ExperimentRunner, ExperimentSweepRunner


def _base_experiment_config(output_root: Path) -> dict:
    return {
        "experiment": {
            "name": "unit_experiment",
            "output_root": str(output_root),
            "run_id": "run_001",
        },
        "agents": {
            "total": 3,
            "honest": 2,
            "byzantine": 1,
            "model": "dummy-model",
            "byzantine_strategy": "prompt_injection",
        },
        "collaboration": {
            "mode": "role_based",
            "rounds": 1,
            "max_tokens": 64,
            "disable_environment_actions": True,
            "options": {
                "static_agent_outputs": {
                    "agent1": "Verified useful memory about the benchmark.",
                    "agent2": "Safe reproducible proposal for the task.",
                }
            },
        },
        "proposal": {
            "enabled": True,
            "builder_enabled": True,
            "verification_enabled": True,
            "weight_manager_enabled": True,
            "max_rounds": 1,
            "verification": {"type": "heuristic"},
            "consensus": {
                "strategy": "majority_vote",
                "confidence_threshold": 0.6,
                "majority_threshold": 0.5,
                "strict_majority": True,
                "minimum_votes": 1,
                "hard_fail_dimensions": ["security"],
            },
        },
        "benchmark": {
            "type": "multiagentbench",
            "inline_task": {
                "scenario": "research",
                "task_id": 7,
                "environment": {
                    "type": "Base",
                    "name": "Unit Environment",
                    "max_iterations": 1,
                },
                "task": {"content": "Design one safe consensus benchmark."},
                "agents": [
                    {
                        "agent_id": "agent1",
                        "type": "BaseAgent",
                        "profile": "Methodologist",
                    },
                    {
                        "agent_id": "agent2",
                        "type": "BaseAgent",
                        "profile": "Evaluator",
                    },
                    {
                        "agent_id": "agent3",
                        "type": "BaseAgent",
                        "profile": "Adversary",
                    },
                ],
                "relationships": [
                    ["agent1", "agent2", "collaborate with"],
                    ["agent1", "agent3", "collaborate with"],
                    ["agent2", "agent3", "collaborate with"],
                ],
            },
        },
    }


def test_experiment_config_validates_agent_counts(tmp_path: Path) -> None:
    data = _base_experiment_config(tmp_path)
    data["agents"]["honest"] = 1

    with pytest.raises(ValueError, match="honest \\+ agents.byzantine"):
        ExperimentConfig.from_dict(data)


def test_experiment_runner_saves_reproducible_artifacts(tmp_path: Path) -> None:
    config = ExperimentConfig.from_dict(_base_experiment_config(tmp_path))

    result = ExperimentRunner(config).run()

    assert result.paths.run_dir.exists()
    assert result.paths.config_path.exists()
    assert result.paths.proposal_result_path.exists()
    assert result.paths.benchmark_result_path.exists()
    assert result.paths.logs_dir.exists()
    assert result.paths.figures_dir.exists()

    proposal_lines = result.paths.proposal_result_path.read_text(
        encoding="utf-8"
    ).splitlines()
    proposal_records = [json.loads(line) for line in proposal_lines]
    assert len(proposal_records) == 3
    assert all("consensus_layer_events" in record for record in proposal_records)
    assert {record["decision"]["result"] for record in proposal_records} == {
        "fail",
        "pass",
    }

    benchmark_result = json.loads(
        result.paths.benchmark_result_path.read_text(encoding="utf-8")
    )
    assert benchmark_result["proposal_count"] == 3
    assert benchmark_result["committed_count"] == 2
    assert benchmark_result["byzantine_agent_count"] == 1
    assert benchmark_result["malicious_proposal_count"] == 1
    assert benchmark_result["malicious_block_rate"] == 1.0
    assert benchmark_result["attack_dataset"]["enabled"] is False
    assert benchmark_result["attack_dataset"]["type"] == "none"
    assert benchmark_result["paths"]["config_path"].endswith("config.yaml")


def test_benchmark_adapter_assigns_honest_and_byzantine_models(
    tmp_path: Path,
) -> None:
    config_data = _base_experiment_config(tmp_path)
    config_data["agents"]["model"] = "default-model"
    config_data["agents"]["honest_model"] = "strong-model"
    config_data["agents"]["byzantine_model"] = "weak-model"
    config_data["agents"]["agent_overrides"] = {
        "agent2": {"llm": "agent-specific-model"}
    }
    config = ExperimentConfig.from_dict(config_data)
    adapter = BENCHMARK_ADAPTERS.create(config.benchmark.type)
    case = adapter.load_case(config)

    engine_config = adapter.build_engine_config(
        case=case,
        config=config,
        proposal_output_path=tmp_path / "proposal_result.jsonl",
    )
    llms = {
        str(agent["agent_id"]): agent["llm"]
        for agent in engine_config.agents
    }

    assert llms["agent1"] == "strong-model"
    assert llms["agent2"] == "agent-specific-model"
    assert llms["agent3"] == "weak-model"


def test_benchmark_adapter_assigns_capability_coefficients(
    tmp_path: Path,
) -> None:
    config_data = _base_experiment_config(tmp_path)
    config_data["agents"]["model"] = "default-model"
    config_data["agents"]["honest_model"] = "strong-model"
    config_data["agents"]["byzantine_model"] = "weak-model"
    config_data["agents"]["honest_capability_coefficient"] = 1.2
    config_data["agents"]["byzantine_capability_coefficient"] = 0.7
    config_data["proposal"]["consensus"]["strategy"] = "smart_quorum"
    config = ExperimentConfig.from_dict(config_data)
    adapter = BENCHMARK_ADAPTERS.create(config.benchmark.type)
    case = adapter.load_case(config)

    engine_config = adapter.build_engine_config(
        case=case,
        config=config,
        proposal_output_path=tmp_path / "proposal_result.jsonl",
    )
    weights = engine_config.consensus["algorithm"]["agent_weights"]

    assert weights["agent1"] == 1.2
    assert weights["agent2"] == 1.2
    assert weights["agent3"] == 0.7


def test_experiment_sweep_runner_saves_progress_and_summary(tmp_path: Path) -> None:
    config_data = _base_experiment_config(tmp_path)
    config_data["experiment"]["run_id"] = "sweep_run"
    first_task = dict(config_data["benchmark"]["inline_task"])
    second_task = dict(config_data["benchmark"]["inline_task"])
    first_task["task_id"] = 1
    second_task["task_id"] = 2
    second_task["task"] = {"content": "Design a second safe consensus benchmark."}
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(task) for task in [first_task, second_task]),
        encoding="utf-8",
    )
    config_data["benchmark"].pop("inline_task")
    config_data["benchmark"]["dataset"] = str(dataset_path)
    config_data["benchmark"]["task_id"] = "1-2"

    result = ExperimentSweepRunner(
        ExperimentConfig.from_dict(config_data)
    ).run_all_cases()

    assert result.sweep_dir.exists()
    assert result.progress_path.exists()
    assert result.summary_path.exists()
    assert len(result.results) == 2
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["task_count"] == 2
    assert summary["completed_task_count"] == 2
    for record in result.results:
        assert Path(record["proposal_result_path"]).exists()
        assert Path(record["benchmark_result_path"]).exists()


def test_experiment_cli_runs_static_smoke_config(tmp_path: Path) -> None:
    config_data = _base_experiment_config(tmp_path)
    config_data["experiment"]["run_id"] = "cli_run"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "marble.experiments.cli",
            "--config",
            str(config_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    run_dir = Path(payload["paths"]["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "proposal_result.jsonl").exists()
    assert (run_dir / "benchmark_result.json").exists()
