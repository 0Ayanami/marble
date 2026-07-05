__all__ = [
    "Engine",
]


def __getattr__(name: str) -> object:
    if name == "Engine":
        from .engine import Engine

        return Engine
    raise AttributeError(name)
