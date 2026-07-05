__all__ = [
    "BaseEnvironment",
    "DBEnvironment",
    "WebEnvironment",
    "ResearchEnvironment",
    "CodingEnvironment",
    "MinecraftEnvironment",
    "WebEnvironment",
    "WorldSimulationEnvironment",
]


def __getattr__(name: str) -> object:
    if name == "BaseEnvironment":
        from .base_env import BaseEnvironment

        return BaseEnvironment
    if name == "CodingEnvironment":
        from .coding_env import CodingEnvironment

        return CodingEnvironment
    if name == "DBEnvironment":
        from .db_env import DBEnvironment

        return DBEnvironment
    if name == "MinecraftEnvironment":
        from .minecraft_env import MinecraftEnvironment

        return MinecraftEnvironment
    if name == "ResearchEnvironment":
        from .research_env import ResearchEnvironment

        return ResearchEnvironment
    if name == "WebEnvironment":
        from .web_env import WebEnvironment

        return WebEnvironment
    if name == "WorldSimulationEnvironment":
        from .world_env import WorldSimulationEnvironment

        return WorldSimulationEnvironment
    raise AttributeError(name)
