"""Result directory and artifact writers for experiment runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from marble.mab_ex.config import ExperimentConfig


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem paths for one experiment run."""

    run_dir: Path
    config_path: Path
    proposal_result_path: Path
    benchmark_result_path: Path
    metrics_path: Path
    summary_path: Path
    logs_dir: Path
    figures_dir: Path

    def to_dict(self) -> Dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "config_path": str(self.config_path),
            "proposal_result_path": str(self.proposal_result_path),
            "benchmark_result_path": str(self.benchmark_result_path),
            "metrics_path": str(self.metrics_path),
            "summary_path": str(self.summary_path),
            "logs_dir": str(self.logs_dir),
            "figures_dir": str(self.figures_dir),
        }


class ExperimentResultWriter:
    """Creates run directories and writes canonical experiment artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.paths = self._create_paths(config)

    def save_config(self) -> Path:
        self.paths.config_path.write_text(
            yaml.safe_dump(
                self.config.to_dict(),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return self.paths.config_path

    def save_proposal_results(
        self,
        records: Iterable[Dict[str, Any]],
    ) -> Path:
        with self.paths.proposal_result_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return self.paths.proposal_result_path

    def save_benchmark_result(self, result: Dict[str, Any]) -> Path:
        self.paths.benchmark_result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return self.paths.benchmark_result_path

    def save_metrics(self, metrics: Dict[str, Any]) -> Path:
        self.paths.metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return self.paths.metrics_path

    def save_summary(self, summary: Dict[str, Any]) -> Path:
        self.paths.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return self.paths.summary_path

    def _create_paths(self, config: ExperimentConfig) -> ExperimentPaths:
        output_root = Path(
            config.benchmark.output_dir or config.experiment.output_root
        )
        run_id = config.experiment.run_id or self._timestamp()
        run_dir = self._unique_run_dir(output_root, run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        logs_dir = run_dir / "logs"
        figures_dir = run_dir / "figures"
        logs_dir.mkdir()
        figures_dir.mkdir()
        return ExperimentPaths(
            run_dir=run_dir,
            config_path=run_dir / "config.yaml",
            proposal_result_path=run_dir / "proposal_result.jsonl",
            benchmark_result_path=run_dir / "benchmark_result.json",
            metrics_path=run_dir / "metrics.json",
            summary_path=run_dir / "summary.json",
            logs_dir=logs_dir,
            figures_dir=figures_dir,
        )

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _unique_run_dir(self, output_root: Path, run_id: str) -> Path:
        candidate = output_root / run_id
        if not candidate.exists():
            return candidate
        suffix = 1
        while True:
            candidate = output_root / f"{run_id}_{suffix:02d}"
            if not candidate.exists():
                return candidate
            suffix += 1


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

