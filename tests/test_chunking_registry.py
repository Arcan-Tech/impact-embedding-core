import pytest

from impact_embedding_core.chunking import (
    RecursiveChunkingStrategy,
    SlidingWindowChunkingStrategy,
    UnknownChunkingStrategyError,
    build_chunking_strategy,
)
from impact_embedding_core.tokenization import TiktokenTokenizer


def _build(name):
    return build_chunking_strategy(
        name,
        tokenizer=TiktokenTokenizer(),
        max_tokens_per_chunk=20,
        overlap_ratio=0.1,
    )


def test_registry_builds_the_recursive_strategy_by_name():
    assert isinstance(_build("recursive"), RecursiveChunkingStrategy)


def test_registry_builds_the_sliding_window_strategy_by_name():
    assert isinstance(_build("sliding-window"), SlidingWindowChunkingStrategy)


def test_registry_rejects_an_unknown_strategy_listing_the_supported_ones():
    with pytest.raises(UnknownChunkingStrategyError) as exc_info:
        _build("semantic")

    message = str(exc_info.value)
    assert "semantic" in message
    assert "recursive" in message
    assert "sliding-window" in message
