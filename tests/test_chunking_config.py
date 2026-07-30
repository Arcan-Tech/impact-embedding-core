import dataclasses
import logging

import pytest

from impact_embedding_core.chunking import ChunkingConfig
from impact_embedding_core.tokenization import DEFAULT_ENCODING_NAME


def test_config_defaults_match_the_embedding_generation_defaults():
    config = ChunkingConfig()

    assert config.strategy == "recursive"
    assert config.max_tokens_per_chunk == 512
    assert config.overlap_ratio == 0.1
    assert config.encoding_name == DEFAULT_ENCODING_NAME


def test_config_accepts_overrides_per_field():
    config = ChunkingConfig(strategy="sliding-window", max_tokens_per_chunk=256)

    assert config.strategy == "sliding-window"
    assert config.max_tokens_per_chunk == 256
    # Untouched fields keep their defaults.
    assert config.overlap_ratio == 0.1


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ChunkingConfig().strategy = "sliding-window"


def test_clamped_to_leaves_a_request_below_the_ceiling_untouched():
    config = ChunkingConfig(max_tokens_per_chunk=512)

    # 0.8 x 40960 = 32768, far above the default request.
    assert config.clamped_to(40960) is config


def test_clamped_to_caps_the_chunk_size_at_the_safety_margin(caplog):
    config = ChunkingConfig(max_tokens_per_chunk=200)

    with caplog.at_level(logging.WARNING):
        clamped = config.clamped_to(100)

    # The chunk size shrinks; the text is not truncated, it yields more chunks.
    assert clamped.max_tokens_per_chunk == 80
    assert clamped.strategy == config.strategy
    assert clamped.overlap_ratio == config.overlap_ratio
    assert "clamping" in caplog.text


def test_clamped_to_does_not_mutate_the_original_config():
    config = ChunkingConfig(max_tokens_per_chunk=200)

    config.clamped_to(100)

    assert config.max_tokens_per_chunk == 200
