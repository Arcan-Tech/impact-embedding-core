"""Golden parity suite.

The guardrail this library is worthless without: `tests/fixtures/golden_parity.json`
is frozen output from the pre-existing reference implementations this
package's chunking and pooling were extracted from (see
`tests/fixtures/generate_golden_parity_fixture.py` for how it was produced —
every source produced byte-identical output). If this suite ever fails, this
package's chunking or pooling has diverged from what already produced
previously-generated vectors, and similarity search against them will
silently degrade.
"""

import json
import math
from pathlib import Path

import pytest

from impact_embedding_core.chunking import RecursiveChunkingStrategy, SlidingWindowChunkingStrategy
from impact_embedding_core.embeddings import pool_embeddings
from impact_embedding_core.tokenization import DEFAULT_ENCODING_NAME, TiktokenTokenizer

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_parity.json"
GOLDEN = json.loads(FIXTURE_PATH.read_text())

MAX_TOKENS_PER_CHUNK = 512
OVERLAP_RATIO = 0.1

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


def _tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer(DEFAULT_ENCODING_NAME)


STRATEGIES = {
    "recursive": lambda: RecursiveChunkingStrategy(
        tokenizer=_tokenizer(),
        max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
        overlap_ratio=OVERLAP_RATIO,
        encoding_name=DEFAULT_ENCODING_NAME,
    ),
    "sliding-window": lambda: SlidingWindowChunkingStrategy(
        tokenizer=_tokenizer(),
        max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
        overlap_ratio=OVERLAP_RATIO,
        encoding_name=DEFAULT_ENCODING_NAME,
    ),
}


@pytest.mark.parametrize("corpus_key", sorted(CORPUS))
@pytest.mark.parametrize("strategy_key", sorted(STRATEGIES))
def test_chunk_boundaries_and_token_counts_match_pre_migration_code(
    corpus_key, strategy_key
):
    strategy = STRATEGIES[strategy_key]()
    chunks = strategy.chunk(CORPUS[corpus_key])

    expected = GOLDEN["chunking"][corpus_key][strategy_key]
    actual = [{"text": chunk.text, "token_count": chunk.token_count} for chunk in chunks]

    assert actual == expected


@pytest.mark.parametrize("case_key", sorted(GOLDEN["pooling_cases"]))
@pytest.mark.parametrize("pooling_strategy", ["mean", "max"])
def test_pooling_matches_pre_migration_code_to_1e_minus_12(case_key, pooling_strategy):
    case = GOLDEN["pooling_cases"][case_key]

    actual = pool_embeddings(case["vectors"], case["weights"], pooling_strategy)
    expected = GOLDEN["pooling"][case_key][pooling_strategy]

    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert math.isclose(actual_value, expected_value, abs_tol=1e-12)
