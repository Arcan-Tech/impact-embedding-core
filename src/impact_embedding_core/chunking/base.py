from dataclasses import dataclass
from typing import Protocol

from impact_embedding_core.tokenization import Tokenizer


@dataclass(frozen=True)
class Chunk:
    """A piece of a source text. ``token_count`` is its pooling weight."""

    text: str
    token_count: int


class ChunkingStrategy(Protocol):
    """Splits a text into chunks that individually fit the embedding limit."""

    def chunk(self, text: str) -> list[Chunk]: ...


class InvalidChunkingConfigError(ValueError):
    """Raised when a chunking strategy is constructed with invalid parameters."""


def resolve_overlap_tokens(max_tokens_per_chunk: int, overlap_ratio: float) -> int:
    """Validate the overlap ratio and derive the overlap in tokens.

    A ratio >= 1 would mean the window never advances, so it is rejected here
    rather than looping forever inside the splitter.
    """
    if max_tokens_per_chunk <= 0:
        raise InvalidChunkingConfigError(
            f"max_tokens_per_chunk must be positive, got {max_tokens_per_chunk}",
        )
    if not 0 <= overlap_ratio < 1:
        raise InvalidChunkingConfigError(
            f"overlap_ratio must be in [0, 1), got {overlap_ratio}",
        )
    return round(max_tokens_per_chunk * overlap_ratio)


def build_chunks(pieces: list[str], text: str, tokenizer: Tokenizer) -> list[Chunk]:
    """Wrap raw splitter output as ``Chunk``s, guaranteeing at least one.

    Splitters return nothing for empty or whitespace-only text; emitting a
    chunk anyway is what keeps every source text mapped to exactly one pooled
    embedding downstream.
    """
    effective_pieces = pieces or [text]
    return [
        Chunk(text=piece, token_count=tokenizer.count_tokens(piece))
        for piece in effective_pieces
    ]
