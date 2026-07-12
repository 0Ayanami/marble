import yaml
from types import SimpleNamespace

from marble.agent.base_agent import BaseAgent
from marble.configs.config import Config
from marble.engine.engine import Engine


def load_config(config_path: str) -> Config:
    """Load YAML configuration and return as a Config object."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def main() -> None:
    # Load configuration
    config = load_config("config.yaml")

    # Initialize and start the Engine
    engine = Engine(config)
    engine.start()


if __name__ == "__main__":
    main()


def _engine_config(tmp_path, coordinate_mode="groupchat", engine_planner=None):
    return Config(
        {
            "environment": {
                "type": "Base",
                "name": "Unit Environment",
                "max_iterations": 1,
            },
            "agents": [
                {"agent_id": "agent1", "type": "BaseAgent", "profile": "Planner"},
                {"agent_id": "agent2", "type": "BaseAgent", "profile": "Reviewer"},
            ],
            "relationships": [["agent1", "agent2", "parent"]],
            "memory": {"type": "SharedMemory"},
            "coordinate_mode": coordinate_mode,
            "task": {"content": "Solve a tiny task."},
            "metrics": {},
            "engine_planner": engine_planner or {},
            "llm": "dummy-model",
            "output": {"file_path": str(tmp_path / "result.jsonl")},
        }
    )


def test_engine_groupchat_coordinate_records_transcript(tmp_path, monkeypatch):
    def fake_act(self, task):
        return f"{self.agent_id} handled task", None

    monkeypatch.setattr(BaseAgent, "act", fake_act)
    engine = Engine(
        _engine_config(
            tmp_path,
            coordinate_mode="groupchat",
            engine_planner={"groupchat": {"rounds": 1}},
        )
    )
    engine.planner.summarize_output = lambda summary, task, output_format: SimpleNamespace(
        content=summary
    )
    engine.planner.update_progress = lambda summary: None
    engine.planner.decide_next_step = lambda results: False

    engine.start()

    assert engine.last_run_result["coordination_mode"] == "groupchat"
    assert engine.last_run_result["agent_outputs"] == {
        "agent1": "agent1 handled task",
        "agent2": "agent2 handled task",
    }
    assert "agent1: agent1 handled task" in engine.last_run_result["group_transcript"]


def test_engine_role_based_dispatches_to_configured_base_mode(tmp_path):
    engine = Engine(
        _engine_config(
            tmp_path,
            coordinate_mode="role_based",
            engine_planner={"role_based": {"base_mode": "tree"}},
        )
    )
    called = {}

    def fake_tree_coordinate():
        called["base_mode"] = "tree"
        engine._write_to_jsonl(
            {
                "task": engine.task,
                "coordination_mode": engine.coordinate_mode,
                "iterations": [],
            }
        )

    engine.tree_coordinate = fake_tree_coordinate

    engine.start()

    assert called == {"base_mode": "tree"}
    assert engine.last_run_result["coordination_mode"] == "role_based"
    assert engine.last_run_result["role_based_base_mode"] == "tree"
