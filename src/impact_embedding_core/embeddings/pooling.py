import math

from impact_embedding_core.embeddings.base import EmbeddingError

POOLING_STRATEGIES = ("max", "mean")

DEFAULT_POOLING_STRATEGY = "mean"


def pool_embeddings(
    vectors: list[list[float]],
    weights: list[float],
    strategy: str,
) -> list[float]:
    """Combine per-chunk vectors into a single vector for one text.

    - ``mean``: token-count-weighted average, then L2-normalized to stay
      comparable with single-chunk vectors in the same cosine index. The
      standard choice, and the weighting stops a short final chunk from
      counting as much as a full one.
    - ``max``: element-wise maximum, weights ignored. Kept for compatibility
      but it breaks the unit norm a cosine index expects; prefer ``mean``.
    - A single vector is returned unchanged for any strategy, so texts that fit
      in one chunk keep exactly the vector the model returned.
    """
    if not vectors:
        raise EmbeddingError("Cannot pool an empty list of vectors")
    if len(vectors) != len(weights):
        raise EmbeddingError(
            f"vectors ({len(vectors)}) and weights ({len(weights)}) "
            "must have the same length",
        )

    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise EmbeddingError(
            "Cannot pool vectors of differing dimensionality",
        )

    if len(vectors) == 1:
        return vectors[0]

    if strategy == "max":
        return _max_pool(vectors, dimension)
    if strategy == "mean":
        return _weighted_mean_pool(vectors, weights, dimension)
    raise EmbeddingError(
        f"Unknown pooling strategy '{strategy}'. "
        f"Supported strategies: {', '.join(POOLING_STRATEGIES)}",
    )


def _max_pool(vectors: list[list[float]], dimension: int) -> list[float]:
    return [
        max(vector[position] for vector in vectors) for position in range(dimension)
    ]


def _weighted_mean_pool(
    vectors: list[list[float]],
    weights: list[float],
    dimension: int,
) -> list[float]:
    total_weight = sum(weights)
    if total_weight <= 0:
        raise EmbeddingError(
            "Cannot compute weighted mean with non-positive total weight",
        )
    averaged = [
        sum(vector[position] * weight for vector, weight in zip(vectors, weights))
        / total_weight
        for position in range(dimension)
    ]
    return _l2_normalize(averaged)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        # Degenerate (e.g. chunk vectors that cancel out); nothing to scale by.
        return vector
    return [value / norm for value in vector]
