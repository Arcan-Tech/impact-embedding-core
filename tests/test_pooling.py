import math

import pytest

from impact_embedding_core.embeddings import EmbeddingError, pool_embeddings


def test_pooling_rejects_an_empty_list_of_vectors():
    with pytest.raises(EmbeddingError, match="empty list of vectors"):
        pool_embeddings([], [], "mean")


def test_pooling_rejects_mismatched_vectors_and_weights():
    with pytest.raises(EmbeddingError, match="same length"):
        pool_embeddings([[1.0], [0.0]], [1.0], "mean")


def test_pooling_rejects_vectors_of_differing_dimensionality():
    with pytest.raises(EmbeddingError, match="differing dimensionality"):
        pool_embeddings([[1.0, 0.0], [1.0]], [1.0, 1.0], "mean")


@pytest.mark.parametrize("strategy", ["mean", "max"])
def test_pooling_returns_a_single_vector_unchanged(strategy):
    # Texts fitting in one chunk keep exactly the vector the model returned.
    vector = [0.3, 0.4, 0.5]

    assert pool_embeddings([vector], [7.0], strategy) == vector


def test_mean_pooling_weights_chunks_by_their_token_count():
    # weights 1 and 3 -> weighted average = [2.25, 4.5] = 2.25 * [1, 2].
    # L2-normalized -> [1, 2] / sqrt(5).
    pooled = pool_embeddings([[0.0, 0.0], [3.0, 6.0]], [1.0, 3.0], "mean")

    root_five = 5.0**0.5
    assert pooled == [pytest.approx(1 / root_five), pytest.approx(2 / root_five)]


def test_mean_pooling_returns_a_unit_vector():
    pooled = pool_embeddings([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0], "mean")

    assert math.isclose(math.sqrt(sum(value**2 for value in pooled)), 1.0)


def test_mean_pooling_rejects_a_non_positive_total_weight():
    with pytest.raises(EmbeddingError, match="non-positive total weight"):
        pool_embeddings([[1.0], [1.0]], [0.0, 0.0], "mean")


def test_mean_pooling_returns_a_degenerate_vector_unscaled():
    # Opposite unit vectors cancel out: nothing to normalize by.
    pooled = pool_embeddings([[1.0, 0.0], [-1.0, 0.0]], [1.0, 1.0], "mean")

    assert pooled == [0.0, 0.0]


def test_max_pooling_takes_the_element_wise_maximum_ignoring_weights():
    vectors = [[1.0, -2.0, 0.5], [0.0, 3.0, 0.25]]

    assert pool_embeddings(vectors, [1.0, 99.0], "max") == [1.0, 3.0, 0.5]


def test_pooling_rejects_an_unknown_strategy():
    with pytest.raises(EmbeddingError) as exc_info:
        pool_embeddings([[1.0], [2.0]], [1.0, 1.0], "median")

    message = str(exc_info.value)
    assert "median" in message
    assert "mean" in message
    assert "max" in message
