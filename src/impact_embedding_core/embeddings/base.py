from dataclasses import dataclass
from typing import Protocol


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class RetryableEmbeddingError(EmbeddingError):
    """Raised for transient embedding failures that are safe to retry.

    A subclass so callers catching ``EmbeddingError`` keep working.
    """


@dataclass(frozen=True)
class Embedding:
    input: str
    vector: list[float]


@dataclass(frozen=True)
class ChunkEmbedding:
    """One chunk's vector, at its 0-based position within the source text.

    Maps onto SemanticEmbeddingChunk in the graph. No chunk text is kept.
    """

    index: int
    token_count: int
    vector: list[float]


@dataclass(frozen=True)
class RecordEmbedding:
    """The embedding output for one text.

    ``pooled`` compares against SemanticEmbedding.value in the graph, the
    ordered ``chunks`` against SemanticEmbeddingChunk.value. A single-chunk text
    has ``pooled == chunks[0].vector``.
    """

    pooled: list[float]
    chunks: list[ChunkEmbedding]


class EmbeddingConfig(Protocol):
    """The provider limits a service needs, whoever the provider is."""

    provider: str
    max_inputs: int
    max_input_tokens: int

    def __str__(self) -> str: ...


class EmbeddingClient(Protocol):
    """Turns texts into vectors.

    The seam that keeps services provider-agnostic: another provider means
    another implementation of this Protocol, with no service change.
    """

    config: EmbeddingConfig

    def generate_embeddings(self, texts: list[str]) -> list[Embedding]: ...
