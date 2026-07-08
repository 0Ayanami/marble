"""Command-line entry point for MARBLE experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MARBLE experiment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run the configured experiment across every task in the benchmark dataset.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        from marble.mab_ex.benchmarks import BENCHMARK_ADAPTERS
        from marble.mab_ex.config import ExperimentConfig
        from marble.mab_ex.runner import (
            AgentCountSweepRunner,
            ExperimentRunner,
            ExperimentSweepRunner,
        )

        config = ExperimentConfig.load(args.config)
        adapter = BENCHMARK_ADAPTERS.create(config.benchmark.type)
        selected_case_ids = adapter.list_case_ids(config)
        should_task_sweep = args.all_tasks or (
            config.benchmark.task_id is not None and len(selected_case_ids) > 1
        )
        should_agent_sweep = bool(config.sweep.agent_counts)
        if should_agent_sweep:
            result = AgentCountSweepRunner(config).run()
        elif should_task_sweep:
            result = ExperimentSweepRunner(
                config,
                ignore_task_id=args.all_tasks,
            ).run_all_cases()
        elif should_agent_sweep:
            result = AgentCountSweepRunner(config).run()
        else:
            result = ExperimentRunner(config).run()
    except Exception:
        logging.exception("Experiment failed.")
        sys.exit(1)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

