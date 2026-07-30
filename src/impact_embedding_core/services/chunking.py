"""ChunkingService
----------------
Splits text the same way for every consumer, so indexed text and query text
share one vector space.

Always uses the library's standard pipeline (ChunkingConfig's defaults): no
constructor parameter lets a caller pick a different strategy, chunk size,
overlap or encoding. Letting one consumer diverge from another here is
exactly the silent drift this package exists to prevent.
"""

import logging

from impact_embedding_core.chunking import Chunk, ChunkingConfig, build_chunking_strategy
from impact_embedding_core.tokenization import TiktokenTokenizer

logger = logging.getLogger(__name__)


class ChunkingService:
    """Splits texts into embedding-sized chunks, using the standard pipeline."""

    def __init__(self) -> None:
        config = ChunkingConfig()
        self.strategy = build_chunking_strategy(
            config.strategy,
            tokenizer=TiktokenTokenizer(config.encoding_name),
            max_tokens_per_chunk=config.max_tokens_per_chunk,
            overlap_ratio=config.overlap_ratio,
            encoding_name=config.encoding_name,
        )

    def chunk(self, text: str) -> list[Chunk]:
        """Split one text. Always returns at least one chunk, even when empty."""
        return self.strategy.chunk(text)

    def chunk_many(self, texts: list[str]) -> list[list[Chunk]]:
        """Split several texts; ``result[i]`` holds the chunks of ``texts[i]``."""
        return [self.strategy.chunk(text) for text in texts]
