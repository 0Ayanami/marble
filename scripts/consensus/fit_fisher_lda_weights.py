"""Fit Fisher LDA alpha/beta groups for consensus weight experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marble.consensus import fit_fisher_lda_quality_weights


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_samples(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if suffix == ".jsonl":
        samples: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    samples.append(json.loads(stripped))
        return samples
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        if not isinstance(data, list):
            raise ValueError("JSON training data must be a list or contain samples.")
        return [dict(item) for item in data]
    raise ValueError("Supported sample formats are .json, .jsonl, and .csv.")


def normalize_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(sample)
    if "vc" not in normalized and "verified_confidence" in normalized:
        normalized["vc"] = normalized["verified_confidence"]
    if "hc" not in normalized and "historical_confidence" in normalized:
        normalized["hc"] = normalized["historical_confidence"]
    if "label" not in normalized and "agent_type" in normalized:
        normalized["label"] = normalized["agent_type"]
    for key in ("vc", "hc"):
        if key in normalized:
            normalized[key] = float(normalized[key])
    return normalized


def filter_samples(
    samples: Iterable[Dict[str, Any]],
    environments: Iterable[str],
    *,
    group_key: str,
) -> List[Dict[str, Any]]:
    wanted = {environment for environment in environments if environment}
    if not wanted:
        return list(samples)
    return [
        sample
        for sample in samples
        if str(sample.get(group_key, sample.get("env", ""))) in wanted
    ]


def group_samples(
    samples: Iterable[Dict[str, Any]],
    *,
    group_key: str,
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        group = str(sample.get(group_key, sample.get("env", "all")))
        groups.setdefault(group, []).append(sample)
    return groups


def fit_groups(
    samples: Iterable[Dict[str, Any]],
    *,
    group_key: str,
    regularization: float,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for group, group_samples_list in sorted(
        group_samples(samples, group_key=group_key).items()
    ):
        normalized = [normalize_sample(sample) for sample in group_samples_list]
        result = fit_fisher_lda_quality_weights(
            normalized,
            regularization=regularization,
        )
        result_data = result.to_dict()
        result_data["group"] = group
        results[group] = result_data
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Fisher LDA alpha/beta parameters from vc/hc samples."
    )
    parser.add_argument(
        "--samples",
        required=True,
        help="Training sample file in JSON, JSONL, or CSV format.",
    )
    parser.add_argument(
        "--group-key",
        default="environment",
        help="Sample field used to fit separate alpha/beta groups.",
    )
    parser.add_argument(
        "--environment",
        action="append",
        default=[],
        help="Only fit samples for this environment. Can be repeated.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Diagonal regularization for the within-class scatter matrix.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. Defaults to benchmark_result/fisher_lda_weights_<timestamp>.json.",
    )
    args = parser.parse_args()

    sample_path = Path(args.samples)
    samples = [normalize_sample(sample) for sample in load_samples(sample_path)]
    samples = filter_samples(
        samples,
        args.environment,
        group_key=args.group_key,
    )
    if not samples:
        raise ValueError("No samples matched the requested filters.")

    output_path = (
        Path(args.output)
        if args.output
        else Path("benchmark_result") / f"fisher_lda_weights_{_timestamp()}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_path": str(sample_path),
        "group_key": args.group_key,
        "environments": args.environment,
        "regularization": args.regularization,
        "groups": fit_groups(
            samples,
            group_key=args.group_key,
            regularization=args.regularization,
        ),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Fisher LDA weights: {output_path}")


if __name__ == "__main__":
    main()
