"""EmbeddingService
-----------------
Turns text into vectors: chunk, embed each chunk, pool into one vector per
text.

Runs the same pipeline for every consumer, so vectors produced here are
directly comparable regardless of who produced them. Depends only on the
EmbeddingClient protocol and ChunkingService, never on a concrete provider.
"""

import logging
from dataclasses import dataclass

from impact_embedding_core.chunking import Chunk
from impact_embedding_core.embeddings import (
    DEFAULT_POOLING_STRATEGY,
    ChunkEmbedding,
    EmbeddingClient,
    RecordEmbedding,
    pool_embeddings,
)
from impact_embedding_core.services.chunking import ChunkingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TextChunk:
    """A chunk tagged with the index of the text it was split from."""

    text_index: int
    chunk: Chunk


class EmbeddingService:
    """Produces pooled and per-chunk embeddings for arbitrary text."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        chunking_service: ChunkingService,
        embeddings_batch: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        embedding_client:  Any implementation of the EmbeddingClient protocol.
        chunking_service:  Splits texts before they are embedded.
        embeddings_batch:  Inputs per API request. Defaults to, and is capped at,
                           the provider's max_inputs.

        Pooling is always ``DEFAULT_POOLING_STRATEGY`` ('mean'): it is the only
        strategy that keeps the pooled vector's norm consistent with the
        model's own unit vectors, and letting one consumer override it while
        the other doesn't is exactly the silent cross-repo drift this package
        exists to prevent.
        """
        self.embedding_client = embedding_client
        self.chunking_service = chunking_service
        self.embeddings_batch = embeddings_batch

    def embed_text(self, text: str) -> RecordEmbedding:
        """Embed a single text, chunking it first when it is too long."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[RecordEmbedding]:
        """Embed several texts at once, reusing API requests across them.

        Every text maps to exactly one RecordEmbedding, aligned by position.
        """
        if not texts:
            return []

        text_chunks = self._chunk_texts(texts)
        logger.debug(
            "Chunked %s texts into %s chunks",
            len(texts),
            len(text_chunks),
        )
        vectors = self._embed_chunk_texts(
            [text_chunk.chunk.text for text_chunk in text_chunks]
        )
        return self._pool_by_text(texts, text_chunks, vectors)

    def _chunk_texts(self, texts: list[str]) -> list[_TextChunk]:
        return [
            _TextChunk(text_index=index, chunk=chunk)
            for index, chunks in enumerate(self.chunking_service.chunk_many(texts))
            for chunk in chunks
        ]

    def _embed_chunk_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._batches(texts, self._embeddings_batch_size()):
            embeddings = self.embedding_client.generate_embeddings(batch)
            vectors.extend(embedding.vector for embedding in embeddings)
        return vectors

    def _embeddings_batch_size(self) -> int:
        """Inputs per request: max_inputs, narrowed by embeddings_batch if set."""
        max_inputs = self.embedding_client.config.max_inputs
        if self.embeddings_batch is None:
            return max_inputs
        return min(self.embeddings_batch, max_inputs)

    def _pool_by_text(
        self,
        texts: list[str],
        text_chunks: list[_TextChunk],
        vectors: list[list[float]],
    ) -> list[RecordEmbedding]:
        vectors_by_text: list[list[list[float]]] = [[] for _ in texts]
        weights_by_text: list[list[float]] = [[] for _ in texts]
        chunks_by_text: list[list[ChunkEmbedding]] = [[] for _ in texts]
        for text_chunk, vector in zip(text_chunks, vectors, strict=True):
            text_index = text_chunk.text_index
            token_count = text_chunk.chunk.token_count
            # Chunks arrive in text order, so appending preserves their position.
            chunks_by_text[text_index].append(
                ChunkEmbedding(
                    index=len(vectors_by_text[text_index]),
                    token_count=token_count,
                    vector=vector,
                )
            )
            vectors_by_text[text_index].append(vector)
            weights_by_text[text_index].append(float(token_count))

        return [
            RecordEmbedding(
                pooled=pool_embeddings(
                    vectors_by_text[index],
                    weights_by_text[index],
                    DEFAULT_POOLING_STRATEGY,
                ),
                chunks=chunks_by_text[index],
            )
            for index in range(len(texts))
        ]

    def _batches(self, items: list, batch_size: int) -> list[list]:
        return [
            items[index : index + batch_size]
            for index in range(0, len(items), batch_size)
        ]
