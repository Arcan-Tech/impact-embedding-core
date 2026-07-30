from dataclasses import dataclass

from impact_embedding_core.chunking import Chunk
from impact_embedding_core.embeddings import Embedding, pool_embeddings
from impact_embedding_core.services.embedding import EmbeddingService


@dataclass(frozen=True)
class FakeEmbeddingConfig:
    """Satisfies the EmbeddingConfig protocol without naming any provider."""

    provider: str = "fake"
    max_inputs: int = 10
    max_input_tokens: int = 1000


class FakeEmbeddingClient:
    """Returns a distinct, deterministic vector per input, recording batches."""

    def __init__(self, max_inputs: int = 10):
        self.config = FakeEmbeddingConfig(max_inputs=max_inputs)
        self.batches: list[list[str]] = []
        self._next_value = 1.0

    def generate_embeddings(self, texts: list[str]) -> list[Embedding]:
        self.batches.append(list(texts))
        embeddings = []
        for text in texts:
            embeddings.append(Embedding(input=text, vector=[self._next_value, 0.0]))
            self._next_value += 1.0
        return embeddings


class FakeChunkingService:
    """Splits on '|' so tests control chunk boundaries exactly."""

    def chunk_many(self, texts: list[str]) -> list[list[Chunk]]:
        return [
            [Chunk(text=piece, token_count=len(piece)) for piece in text.split("|")]
            for text in texts
        ]


def _service(client=None, **kwargs):
    return EmbeddingService(
        client or FakeEmbeddingClient(),
        FakeChunkingService(),
        **kwargs,
    )


def test_embedding_service_works_with_any_client_satisfying_the_protocol():
    # The IoC guarantee: nothing in the service knows about Scaleway.
    client = FakeEmbeddingClient()

    result = _service(client).embed_text("hello")

    assert result.pooled == [1.0, 0.0]
    assert client.batches == [["hello"]]


def test_embedding_service_returns_one_chunk_embedding_per_chunk():
    result = _service().embed_text("aa|bbb|c")

    assert [chunk.index for chunk in result.chunks] == [0, 1, 2]
    assert [chunk.token_count for chunk in result.chunks] == [2, 3, 1]
    assert [chunk.vector for chunk in result.chunks] == [
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ]


def test_embedding_service_pools_chunk_vectors_weighted_by_token_count():
    result = _service().embed_text("aa|bbb|c")

    expected = pool_embeddings(
        [chunk.vector for chunk in result.chunks],
        [float(chunk.token_count) for chunk in result.chunks],
        "mean",
    )
    assert result.pooled == expected


def test_embedding_service_returns_the_chunk_vector_itself_for_a_single_chunk():
    result = _service().embed_text("single")

    assert len(result.chunks) == 1
    assert result.pooled == result.chunks[0].vector


def test_embedding_service_aligns_results_positionally_with_the_inputs():
    results = _service().embed_texts(["a|b", "c", "d|e|f"])

    assert [len(result.chunks) for result in results] == [2, 1, 3]
    assert results[1].pooled == [3.0, 0.0]


def test_embedding_service_returns_nothing_for_no_texts():
    client = FakeEmbeddingClient()

    assert _service(client).embed_texts([]) == []
    assert client.batches == []


def test_embedding_service_splits_chunks_across_requests_of_the_configured_size():
    client = FakeEmbeddingClient()

    _service(client, embeddings_batch=2).embed_text("a|b|c|d|e")

    assert [len(batch) for batch in client.batches] == [2, 2, 1]
    assert [text for batch in client.batches for text in batch] == list("abcde")


def test_embedding_service_defaults_the_batch_size_to_the_provider_limit():
    client = FakeEmbeddingClient(max_inputs=3)

    _service(client).embed_text("a|b|c|d")

    assert [len(batch) for batch in client.batches] == [3, 1]


def test_embedding_service_never_exceeds_the_provider_batch_limit():
    client = FakeEmbeddingClient(max_inputs=2)

    _service(client, embeddings_batch=100).embed_text("a|b|c")

    assert [len(batch) for batch in client.batches] == [2, 1]
