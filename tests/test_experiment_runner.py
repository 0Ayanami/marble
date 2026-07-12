import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from marble.mab_ex.benchmarks import BENCHMARK_ADAPTERS
from marble.mab_ex.config import ExperimentConfig
from marble.mab_ex.runner import (
    AgentCountSweepRunner,
    ExperimentRunner,
    ExperimentSweepRunner,
)


def _base_experiment_config(output_root: Path) -> dict:
    return {
        "experiment": {
            "name": "unit_experiment",
            "output_root": str(output_root),
            "run_id": "run_001",
        },
        "agents": {
            "total": 3,
            "model": "dummy-model",
            "capability_coefficient": 8.96,
            "model_capability_coefficients": {"dummy-model": 8.96},
            "num_byzantine": 0,
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
                    "agent3": "Clear summary for the task.",
                    "agent_03": "Generated fallback agent contribution.",
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
            "environment": "research",
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
                        "profile": "Summarizer",
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


def test_experiment_config_supports_byzantine_model_groups(tmp_path: Path) -> None:
    data = _base_experiment_config(tmp_path)
    data["agents"]["num_byzantine"] = 1
    data["agents"]["models"] = {
        "honest": "honest-model",
        "byzantine": "byzantine-model",
    }
    data["agents"]["model_capability_coefficients"] = {
        "honest-model": 8.96,
        "byzantine-model": 2.0,
    }
    data["proposal"]["consensus"]["strategy"] = "smart_quorum"

    config = ExperimentConfig.from_dict(data)
    adapter = BENCHMARK_ADAPTERS.create(config.benchmark.type)
    case = adapter.load_case(config)
    engine_config = adapter.build_engine_config(
        case=case,
        config=config,
        proposal_output_path=tmp_path / "proposal_result.jsonl",
    )

    llms = {str(agent["agent_id"]): agent["llm"] for agent in engine_config.agents}
    roles = {
        str(agent["agent_id"]): agent["consensus_role"]
        for agent in engine_config.agents
    }
    algorithm = engine_config.consensus["algorithm"]

    assert llms == {
        "agent1": "honest-model",
        "agent2": "honest-model",
        "agent3": "byzantine-model",
    }
    assert roles["agent3"] == "byzantine"
    assert algorithm["honest_agents"] == ["agent1", "agent2"]
    assert algorithm["byzantine_agents"] == ["agent3"]
    assert algorithm["agent_weights"]["agent1"] == 8.96
    assert algorithm["agent_weights"]["agent3"] == 2.0


def test_experiment_config_loads_benchmark_config_path_and_consensus_switch(
    tmp_path: Path,
) -> None:
    data = _base_experiment_config(tmp_path)
    benchmark_config_path = tmp_path / "benchmark.yaml"
    inline_task = data["benchmark"].pop("inline_task")
    benchmark_config_path.write_text(
        yaml.safe_dump(
            {
                "environment": "research",
                "inline_task": inline_task,
                "task_id": 7,
                "options": {"evaluate_kpi": True},
                "metrics": {"effectiveness": ["KPI"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data["benchmark"] = {
        "type": "multiagentbench",
        "config_path": str(benchmark_config_path),
        "output_dir": str(tmp_path / "runs"),
    }
    data["consensus"] = {"enabled": False}
    data["proposal"].pop("enabled", None)

    config = ExperimentConfig.from_dict(data)

    assert config.proposal.enabled is False
    assert config.benchmark.environment == "research"
    assert config.benchmark.task_id == 7
    assert config.benchmark.options["evaluate_kpi"] is True
    assert config.benchmark.metrics["effectiveness"] == ["KPI"]


def test_benchmark_metrics_kpi_request_enables_kpi_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_model_prompting(**_: object) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                content=json.dumps(
                    [
                        {
                            "milestone": "safe consensus benchmark drafted",
                            "agents": ["agent1", "agent2"],
                        }
                    ]
                )
            )
        ]

    monkeypatch.setattr("marble.evaluator.evaluator.model_prompting", fake_model_prompting)
    config_data = _base_experiment_config(tmp_path)
    config_data["benchmark"]["options"] = {"evaluate_kpi": False}
    config_data["benchmark"]["metrics"] = {"effectiveness": ["KPI"]}
    config = ExperimentConfig.from_dict(config_data)

    result = ExperimentRunner(config).run()
    benchmark_result = json.loads(
        result.paths.benchmark_result_path.read_text(encoding="utf-8")
    )

    assert benchmark_result["effectiveness"]["metric"] == "KPI"
    assert benchmark_result["kpi_requested"] is True
    assert benchmark_result["kpi_evaluated"] is True
    assert benchmark_result["effectiveness"]["kpi_evaluation_count"] == 1
    assert benchmark_result["effectiveness"]["kpi_status"] == "evaluated"
    assert benchmark_result["effectiveness"]["total_milestones_M"] == 1
    assert benchmark_result["effectiveness"]["kpi_accomplishment_ratio"] == 1.0


def test_experiment_config_rejects_too_many_byzantine_agents(
    tmp_path: Path,
) -> None:
    data = _base_experiment_config(tmp_path)
    data["agents"]["num_byzantine"] = 4

    with pytest.raises(ValueError, match="cannot exceed"):
        ExperimentConfig.from_dict(data)


def test_experiment_runner_saves_reproducible_artifacts(tmp_path: Path) -> None:
    config = ExperimentConfig.from_dict(_base_experiment_config(tmp_path))

    result = ExperimentRunner(config).run()

    assert result.paths.run_dir.exists()
    assert result.paths.config_path.exists()
    assert result.paths.proposal_result_path.exists()
    assert result.paths.benchmark_result_path.exists()
    assert result.paths.metrics_path.exists()
    assert result.paths.summary_path.exists()
    assert result.paths.logs_dir.exists()

    proposal_lines = result.paths.proposal_result_path.read_text(
        encoding="utf-8"
    ).splitlines()
    proposal_records = [json.loads(line) for line in proposal_lines]
    assert len(proposal_records) == 3
    assert all("consensus_layer_events" in record for record in proposal_records)
    assert {record["decision"]["result"] for record in proposal_records} == {"pass"}

    benchmark_result = json.loads(
        result.paths.benchmark_result_path.read_text(encoding="utf-8")
    )
    assert benchmark_result["proposal_count"] == 3
    assert benchmark_result["committed_count"] == 3
    assert "effectiveness" in benchmark_result
    assert "efficiency" in benchmark_result
    assert benchmark_result["effectiveness"]["metric"] == "KPI"
    assert "qc_evolvement" in benchmark_result["efficiency"]
    assert "voting_agents_count_evolvement" in benchmark_result["efficiency"]
    assert benchmark_result["paths"]["config_path"].endswith("config.yaml")

    metrics = json.loads(result.paths.metrics_path.read_text(encoding="utf-8"))
    assert metrics["task_id"] == "7"
    assert metrics["environment"] == "research"
    assert metrics["workflow"] == "role_based"
    assert metrics["consensus"] is True
    assert metrics["agent_number"] == 3
    assert "overall_score" in metrics
    assert "average_task_score" in metrics
    assert "completion_rate" in metrics
    assert metrics["task_scores"]["7"] == metrics["overall_score"]


def test_benchmark_adapter_assigns_models_and_capabilities(
    tmp_path: Path,
) -> None:
    config_data = _base_experiment_config(tmp_path)
    config_data["agents"]["model"] = "default-model"
    config_data["agents"]["model_capability_coefficients"] = {
        "default-model": 8.96,
        "agent-specific-model": 9.5,
    }
    config_data["agents"]["agent_overrides"] = {
        "agent2": {"llm": "agent-specific-model"}
    }
    config_data["proposal"]["consensus"]["strategy"] = "smart_quorum"
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
    weights = engine_config.consensus["algorithm"]["agent_weights"]

    assert llms["agent1"] == "default-model"
    assert llms["agent2"] == "agent-specific-model"
    assert llms["agent3"] == "default-model"
    assert weights["agent1"] == 8.96
    assert weights["agent2"] == 9.5
    assert weights["agent3"] == 8.96


def test_benchmark_adapter_maps_collaboration_modes_to_engine_coordinates(
    tmp_path: Path,
) -> None:
    config_data = _base_experiment_config(tmp_path)
    config = ExperimentConfig.from_dict(config_data)
    adapter = BENCHMARK_ADAPTERS.create(config.benchmark.type)
    case = adapter.load_case(config)

    role_engine_config = adapter.build_engine_config(
        case=case,
        config=config,
        proposal_output_path=tmp_path / "role_result.jsonl",
    )

    assert role_engine_config.coordination_mode == "role_based"
    assert role_engine_config.engine_planner["role_based"]["base_mode"] == "tree"
    assert any(
        relationship[2] == "parent"
        for relationship in role_engine_config.relationships
    )

    config_data["collaboration"]["mode"] = "groupchat"
    config_data["collaboration"]["rounds"] = 2
    groupchat_config = ExperimentConfig.from_dict(config_data)
    groupchat_case = adapter.load_case(groupchat_config)
    groupchat_engine_config = adapter.build_engine_config(
        case=groupchat_case,
        config=groupchat_config,
        proposal_output_path=tmp_path / "groupchat_result.jsonl",
    )

    assert groupchat_engine_config.coordination_mode == "groupchat"
    assert groupchat_engine_config.engine_planner["groupchat"]["rounds"] == 2


def test_consensus_and_weight_manager_defaults_are_configured(
    tmp_path: Path,
) -> None:
    config_data = _base_experiment_config(tmp_path)
    config_data["proposal"]["consensus"]["strategy"] = "smart_quorum"
    config = ExperimentConfig.from_dict(config_data)
    adapter = BENCHMARK_ADAPTERS.create(config.benchmark.type)
    case = adapter.load_case(config)

    engine_config = adapter.build_engine_config(
        case=case,
        config=config,
        proposal_output_path=tmp_path / "proposal_result.jsonl",
    )

    algorithm = engine_config.consensus["algorithm"]
    weight_manager = engine_config.consensus["weight_manager"]
    assert algorithm["epsilon_ratio"] == 1
    assert algorithm["epsilon"] is None
    assert weight_manager == {
        "alpha": 0.6,
        "beta": 0.4,
        "gamma": 5.0,
        "theta": 0.5,
        "proposal_window": 20,
        "vote_window": 30,
        "initial_vc": 0.5,
        "initial_hc": 0.5,
    }


def test_legacy_consensus_alpha_beta_are_migrated_to_weight_manager(
    tmp_path: Path,
) -> None:
    config_data = _base_experiment_config(tmp_path)
    config_data["proposal"]["consensus"]["strategy"] = "smart_quorum"
    config_data["proposal"]["consensus"]["alpha"] = 0.5
    config_data["proposal"]["consensus"]["beta"] = 0.5

    config = ExperimentConfig.from_dict(config_data)

    assert config.proposal.weight_manager["alpha"] == 0.5
    assert config.proposal.weight_manager["beta"] == 0.5


def test_experiment_sweep_runner_saves_progress_summary_and_metrics(
    tmp_path: Path,
) -> None:
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
    assert result.metrics_path.exists()
    assert len(result.results) == 2
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["task_count"] == 2
    assert summary["completed_task_count"] == 2
    assert "overall_score" in summary
    assert "average_task_score" in metrics
    assert metrics["task_scores"].keys() == {"1", "2"}
    for record in result.results:
        assert Path(record["proposal_result_path"]).exists()
        assert Path(record["benchmark_result_path"]).exists()
        assert Path(record["metrics_path"]).exists()


def test_agent_count_sweep_runner_expands_config(tmp_path: Path) -> None:
    config_data = _base_experiment_config(tmp_path)
    config_data["experiment"]["run_id"] = "agent_sweep"
    config_data["benchmark"]["output_dir"] = str(tmp_path / "groupchat")
    config_data["sweep"] = {"agent_counts": [2, 3]}
    config_data["agents"]["total"] = 2
    config_data["benchmark"]["inline_task"]["agents"] = (
        config_data["benchmark"]["inline_task"]["agents"][:2]
    )

    result = AgentCountSweepRunner(
        ExperimentConfig.from_dict(config_data)
    ).run()

    payload = result.to_dict()
    assert [item["agent_number"] for item in payload["agent_count_results"]] == [2, 3]
    for item in payload["agent_count_results"]:
        assert Path(item["summary_path"]).exists()
        assert Path(item["metrics_path"]).exists()


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
            "marble.mab_ex.cli",
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
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "summary.json").exists()
