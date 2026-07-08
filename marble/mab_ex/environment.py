"""Environment adapters for MultiAgentBench experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Type


def _merge_empty(current: Mapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(current)
    for key, value in defaults.items():
        if merged.get(key) in (None, ""):
            merged[key] = value
    return merged


@dataclass(frozen=True)
class EnvironmentAdapter:
    """Build a MARBLE environment config for one MultiAgentBench environment."""

    name: str
    marble_type: str
    display_name: str
    default_max_iterations: int = 1
    options: Dict[str, Any] = field(default_factory=dict)
    runnable: bool = True

    def build_config(
        self,
        *,
        raw_environment: Mapping[str, Any],
        defaults: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base = {
            "type": self.marble_type,
            "name": self.display_name,
            "max_iterations": self.default_max_iterations,
        }
        base.update(self.options)
        base.update(defaults)
        return _merge_empty(raw_environment, base)


class EnvironmentAdapterRegistry:
    """Registry for benchmark environment adapters."""

    def __init__(self) -> None:
        self._adapters: Dict[str, EnvironmentAdapter] = {}

    def register(self, adapter: EnvironmentAdapter) -> None:
        self._adapters[adapter.name.replace("-", "_").lower()] = adapter

    def create(self, name: str) -> EnvironmentAdapter:
        normalized = name.replace("-", "_").lower()
        aliases = {
            "research_environment": "research",
            "coding": "coding_challenge",
            "codingchallenge": "coding_challenge",
            "db": "database",
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return self._adapters[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._adapters))
            raise ValueError(
                f"Unknown MAB environment '{name}'. Available: {available}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._adapters)


ENVIRONMENT_ADAPTERS = EnvironmentAdapterRegistry()
ENVIRONMENT_ADAPTERS.register(
    EnvironmentAdapter(
        name="research",
        marble_type="Research",
        display_name="Research Collaboration Environment",
    )
)
ENVIRONMENT_ADAPTERS.register(
    EnvironmentAdapter(
        name="minecraft",
        marble_type="Minecraft",
        display_name="Minecraft Environment",
        runnable=False,
    )
)
ENVIRONMENT_ADAPTERS.register(
    EnvironmentAdapter(
        name="database",
        marble_type="DB",
        display_name="Database Environment",
        runnable=False,
    )
)
ENVIRONMENT_ADAPTERS.register(
    EnvironmentAdapter(
        name="coding_challenge",
        marble_type="Coding",
        display_name="Coding Challenge Environment",
        runnable=False,
    )
)


def build_environment_config(
    *,
    environment_name: str,
    raw_environment: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a MARBLE engine environment config from a MAB environment name."""
    return ENVIRONMENT_ADAPTERS.create(environment_name).build_config(
        raw_environment=raw_environment,
        defaults=defaults,
    )
