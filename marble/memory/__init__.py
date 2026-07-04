from .base_memory import BaseMemory
from .long_term_memory import LongTermMemory
from .shared_memory import SharedMemory
from .short_term_memory import ShortTermMemory

__all__ = [
    "BaseMemory",
    "ConsensusMemory",
    "SharedMemory",
    "LongTermMemory",
    "ShortTermMemory",
]


def __getattr__(name: str) -> object:
    if name == "ConsensusMemory":
        from marble.memory.consensus_memory import ConsensusMemory

        return ConsensusMemory
    raise AttributeError(name)
