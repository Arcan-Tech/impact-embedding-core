"""Shared text-to-vector pipeline.

Tokenization, chunking, pooling and embedding-generation logic, kept in one
place so that multiple consumers cannot silently drift apart on how a piece
of text turns into a vector. See README.md for the contract this package is
the source of truth for.

This module re-exports the entry points most consumers need: the chunk ->
embed -> pool orchestrator, the Scaleway client, the result types, and the
error types its public API can raise. ``ChunkingService`` and
``EmbeddingService`` take no configuration beyond their required
collaborators (an embedding client, in EmbeddingService's case) — chunking
strategy, chunk size, overlap, encoding and pooling strategy are fixed to the
library's standard pipeline, not caller-selectable, so consumers cannot
silently diverge on them. ``ScalewayEmbeddingClient`` is the exception: its
configuration stays fully caller-supplied because credentials and endpoint
have no sensible library default.

Lower-level building blocks (individual chunking strategies, the strategy
registry, ``ChunkingConfig``, the tokenizer) live in their own submodules
(``impact_embedding_core.chunking``, ``.embeddings``, ``.tokenization``) and
are reachable from there, but are not part of this curated surface.
"""

from impact_embedding_core.chunking import Chunk
from impact_embedding_core.embeddings import ChunkEmbedding
from impact_embedding_core.embeddings import Embedding
from impact_embedding_core.embeddings import EmbeddingClient
from impact_embedding_core.embeddings import EmbeddingConfig
from impact_embedding_core.embeddings import EmbeddingError
from impact_embedding_core.embeddings import RecordEmbedding
from impact_embedding_core.embeddings import RetryableEmbeddingError
from impact_embedding_core.embeddings import pool_embeddings
from impact_embedding_core.embeddings.clients import ScalewayEmbeddingClient
from impact_embedding_core.embeddings.clients import ScalewayEmbeddingConfig
from impact_embedding_core.services import ChunkingService
from impact_embedding_core.services import EmbeddingService

__all__ = [
    "Chunk",
    "ChunkEmbedding",
    "ChunkingService",
    "Embedding",
    "EmbeddingClient",
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingService",
    "RecordEmbedding",
    "RetryableEmbeddingError",
    "ScalewayEmbeddingClient",
    "ScalewayEmbeddingConfig",
    "pool_embeddings",
]
