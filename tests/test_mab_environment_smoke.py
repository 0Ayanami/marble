import json
from pathlib import Path

import pytest

from marble.mab_ex.config import ExperimentConfig
from marble.mab_ex.runner import ExperimentRunner


SMOKE_CONFIGS = [
    "minecraft_groupchat.yaml",
    "minecraft_role_based.yaml",
    "minecraft_groupchat_consensus.yaml",
    "minecraft_role_based_consensus.yaml",
    "database_groupchat.yaml",
    "database_role_based.yaml",
    "database_groupchat_consensus.yaml",
    "database_role_based_consensus.yaml",
    "coding_challenge_groupchat.yaml",
    "coding_challenge_role_based.yaml",
    "coding_challenge_groupchat_consensus.yaml",
    "coding_challenge_role_based_consensus.yaml",
]


@pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
def test_mab_environment_smoke_configs_start_one_task(
    tmp_path: Path,
    config_name: str,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "marble"
        / "configs"
        / "experiment_configs"
        / "benchmarks"
        / "marble_configs"
        / "mab_smoke"
        / config_name
    )
    config_data = ExperimentConfig.load(str(config_path)).to_dict()
    config_data["benchmark"]["output_dir"] = str(tmp_path / config_name[:-5])
    config = ExperimentConfig.from_dict(config_data)

    result = ExperimentRunner(config).run()

    benchmark_result = json.loads(
        result.paths.benchmark_result_path.read_text(encoding="utf-8")
    )
    summary = json.loads(result.paths.summary_path.read_text(encoding="utf-8"))

    assert result.paths.config_path.exists()
    assert result.paths.proposal_result_path.exists()
    assert result.paths.metrics_path.exists()
    assert benchmark_result["task_id"] == "1"
    assert benchmark_result["environment"] == config.benchmark.environment
    assert benchmark_result["experiment_group"] == config.experiment.group
    assert summary["experiment_group"] == config.experiment.group
    assert benchmark_result["consensus"] is config.proposal.enabled
    if config.proposal.enabled:
        assert benchmark_result["committed_count"] > 0
    else:
        assert benchmark_result["committed_count"] == 0
