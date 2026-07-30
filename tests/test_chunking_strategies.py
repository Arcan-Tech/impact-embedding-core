import pytest

from impact_embedding_core.chunking import (
    InvalidChunkingConfigError,
    RecursiveChunkingStrategy,
    SlidingWindowChunkingStrategy,
)
from impact_embedding_core.tokenization import TiktokenTokenizer

# Two paragraphs, each ~15 tokens, separated by a blank line. With a 20-token
# limit, the natural split point is the paragraph break.
_FIRST_PARAGRAPH = ("alpha " * 15).strip()
_SECOND_PARAGRAPH = ("beta " * 15).strip()
_TWO_PARAGRAPHS = f"{_FIRST_PARAGRAPH}\n\n{_SECOND_PARAGRAPH}"

_LONG_TEXT = ("word " * 200).strip()


def _recursive(max_tokens_per_chunk=20, overlap_ratio=0.0):
    return RecursiveChunkingStrategy(
        TiktokenTokenizer(), max_tokens_per_chunk, overlap_ratio
    )


def _sliding(max_tokens_per_chunk=20, overlap_ratio=0.0):
    return SlidingWindowChunkingStrategy(
        TiktokenTokenizer(), max_tokens_per_chunk, overlap_ratio
    )


@pytest.mark.parametrize("build_strategy", [_recursive, _sliding])
def test_strategy_returns_short_text_as_a_single_unchanged_chunk(build_strategy):
    chunks = build_strategy().chunk("hello world")

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].token_count == 2


@pytest.mark.parametrize("build_strategy", [_recursive, _sliding])
def test_strategy_returns_one_empty_chunk_for_empty_text(build_strategy):
    # Splitters return nothing for empty input; emitting a chunk anyway is what
    # guarantees one embedding per source text.
    chunks = build_strategy().chunk("")

    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].token_count == 0


@pytest.mark.parametrize("build_strategy", [_recursive, _sliding])
def test_strategy_keeps_every_chunk_within_the_token_limit(build_strategy):
    chunks = build_strategy().chunk(_LONG_TEXT)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 20 for chunk in chunks)


@pytest.mark.parametrize("build_strategy", [_recursive, _sliding])
def test_strategy_annotates_each_chunk_with_its_own_token_count(build_strategy):
    tokenizer = TiktokenTokenizer()

    chunks = build_strategy().chunk(_LONG_TEXT)

    assert all(
        chunk.token_count == tokenizer.count_tokens(chunk.text) for chunk in chunks
    )


def test_recursive_prefers_paragraph_boundary_over_mid_content_cut():
    recursive_chunks = _recursive().chunk(_TWO_PARAGRAPHS)
    sliding_chunks = _sliding().chunk(_TWO_PARAGRAPHS)

    # Recursive keeps each paragraph intact: no chunk mixes alpha and beta.
    assert all(
        not ("alpha" in chunk.text and "beta" in chunk.text)
        for chunk in recursive_chunks
    )
    # The naive sliding window, blind to the boundary, produces a chunk that
    # spans the paragraph break.
    assert any(
        "alpha" in chunk.text and "beta" in chunk.text for chunk in sliding_chunks
    )


@pytest.mark.parametrize("build_strategy", [_recursive, _sliding])
def test_strategy_overlaps_consecutive_chunks_when_ratio_is_positive(build_strategy):
    without_overlap = build_strategy(overlap_ratio=0.0).chunk(_LONG_TEXT)
    with_overlap = build_strategy(overlap_ratio=0.5).chunk(_LONG_TEXT)

    # Repeated content across boundaries takes more chunks to cover one text.
    assert len(with_overlap) > len(without_overlap)


@pytest.mark.parametrize("build_strategy", [_recursive, _sliding])
@pytest.mark.parametrize("overlap_ratio", [1.0, 1.5, -0.1])
def test_strategy_rejects_an_overlap_ratio_outside_the_unit_interval(
    build_strategy, overlap_ratio
):
    # A ratio of 1 or more means the window never advances.
    with pytest.raises(InvalidChunkingConfigError, match="overlap_ratio"):
        build_strategy(overlap_ratio=overlap_ratio)


@pytest.mark.parametrize("build_strategy", [_recursive, _sliding])
@pytest.mark.parametrize("max_tokens_per_chunk", [0, -1])
def test_strategy_rejects_a_non_positive_chunk_size(
    build_strategy, max_tokens_per_chunk
):
    with pytest.raises(InvalidChunkingConfigError, match="max_tokens_per_chunk"):
        build_strategy(max_tokens_per_chunk=max_tokens_per_chunk)
