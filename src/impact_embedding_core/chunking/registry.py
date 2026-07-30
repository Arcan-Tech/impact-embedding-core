from typing import Any

from impact_embedding_core.chunking.base import ChunkingStrategy
from impact_embedding_core.chunking.recursive import RecursiveChunkingStrategy
from impact_embedding_core.chunking.sliding_window import SlidingWindowChunkingStrategy

# "sentence"/"semantic" are intentionally not registered: they would need
# NLTK/spaCy, or langchain-experimental plus embedding calls during chunking.
_STRATEGIES: dict[str, type[ChunkingStrategy]] = {
    "sliding-window": SlidingWindowChunkingStrategy,
    "recursive": RecursiveChunkingStrategy,
}


class UnknownChunkingStrategyError(ValueError):
    """Raised when a chunking strategy name is not registered."""


def build_chunking_strategy(name: str, **kwargs: Any) -> ChunkingStrategy:
    try:
        strategy_class = _STRATEGIES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_STRATEGIES))
        raise UnknownChunkingStrategyError(
            f"Unknown chunking strategy '{name}'. Supported strategies: {supported}",
        ) from exc
    return strategy_class(**kwargs)
