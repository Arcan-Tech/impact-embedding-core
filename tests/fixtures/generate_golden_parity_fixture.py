"""Regenerates golden_parity.json from a reference implementation's own code.

This fixture was originally frozen while extracting this library out of two
pre-existing implementations that had independently converged on the same
chunking/pooling behaviour. Run this script inside each source project's own
venv, against its own chunking/tokenization/embeddings modules, and diff the
outputs before freezing one as tests/fixtures/golden_parity.json — they must
be identical:

    uv run python generate_golden_parity_fixture.py <package_name>

where ``<package_name>`` is the importable package that exposes
``<package_name>.tokenization``, ``<package_name>.chunking`` and
``<package_name>.embeddings`` (or ``<package_name>.chunking.pool_embeddings``,
see below).

Prints one JSON object to stdout.
"""

import importlib
import json
import sys
from dataclasses import asdict


CORPUS = {
    "empty": "",
    "whitespace_only": "   \n\t  \n  ",
    "short": "The quick brown fox jumps over the lazy dog.",
    "multi_chunk": ("word " * 1200).strip(),
    "unicode": (
        "Café naïve façade — 東京タワー is 電波塔 (でんぱとう). "
        "emoji stress test: 😀🚀🔥 café touché résumé. "
    )
    * 40,
}


def main() -> None:
    prefix = sys.argv[1]

    tokenization = importlib.import_module(f"{prefix}.tokenization")
    chunking = importlib.import_module(f"{prefix}.chunking")

    TiktokenTokenizer = tokenization.TiktokenTokenizer
    DEFAULT_ENCODING_NAME = tokenization.DEFAULT_ENCODING_NAME
    RecursiveChunkingStrategy = chunking.RecursiveChunkingStrategy
    SlidingWindowChunkingStrategy = chunking.SlidingWindowChunkingStrategy

    tokenizer = TiktokenTokenizer(DEFAULT_ENCODING_NAME)

    max_tokens_per_chunk = 512
    overlap_ratio = 0.1

    strategies = {
        "recursive": RecursiveChunkingStrategy(
            tokenizer=tokenizer,
            max_tokens_per_chunk=max_tokens_per_chunk,
            overlap_ratio=overlap_ratio,
            encoding_name=DEFAULT_ENCODING_NAME,
        ),
        "sliding-window": SlidingWindowChunkingStrategy(
            tokenizer=tokenizer,
            max_tokens_per_chunk=max_tokens_per_chunk,
            overlap_ratio=overlap_ratio,
            encoding_name=DEFAULT_ENCODING_NAME,
        ),
    }

    chunking_results = {}
    for corpus_key, text in CORPUS.items():
        chunking_results[corpus_key] = {
            strategy_key: [asdict(chunk) for chunk in strategy.chunk(text)]
            for strategy_key, strategy in strategies.items()
        }

    try:
        pool_embeddings = importlib.import_module(f"{prefix}.embeddings").pool_embeddings
    except (ModuleNotFoundError, AttributeError):
        # Some source layouts place pooling under chunking/, not embeddings/.
        pool_embeddings = chunking.pool_embeddings

    pooling_cases = {
        "single_vector": {
            "vectors": [[0.6, 0.8, 0.0]],
            "weights": [5.0],
        },
        "equal_weights": {
            "vectors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "weights": [1.0, 1.0, 1.0],
        },
        "uneven_weights": {
            "vectors": [[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]],
            "weights": [10.0, 512.0, 51.0],
        },
        "negative_components": {
            "vectors": [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]],
            "weights": [2.0, 7.0],
        },
        "cancelling_to_zero": {
            "vectors": [[1.0, 1.0], [-1.0, -1.0]],
            "weights": [1.0, 1.0],
        },
    }

    pooling_results = {}
    for case_key, case in pooling_cases.items():
        pooling_results[case_key] = {}
        for strategy_name in ("mean", "max"):
            pooling_results[case_key][strategy_name] = pool_embeddings(
                case["vectors"], case["weights"], strategy_name
            )

    print(
        json.dumps(
            {
                "chunking": chunking_results,
                "pooling": pooling_results,
                "pooling_cases": pooling_cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
