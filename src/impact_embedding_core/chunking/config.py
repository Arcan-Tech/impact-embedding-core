"""Chunking configuration.

The defaults here are the settings this library's consumers are expected to
share. Overriding them is a caller's decision, and it has to match every
other consumer: text chunked differently from what was indexed lands
elsewhere in the vector space and degrades retrieval with no error.
"""

import logging
from dataclasses import dataclass, replace

from impact_embedding_core.tokenization import DEFAULT_ENCODING_NAME

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "recursive"
DEFAULT_MAX_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_RATIO = 0.1

# Fraction of the provider's token limit a chunk may reach, absorbing the gap
# between our generic encoding and the provider's tokenizer. Internal by design.
_CHUNK_TOKEN_SAFETY_MARGIN = 0.8


@dataclass(frozen=True)
class ChunkingConfig:
    """How to split a text before embedding it.

    ``max_tokens_per_chunk`` is a target chunk size, decoupled from the model
    context window to keep chunks small and retrieval-friendly.
    """

    strategy: str = DEFAULT_STRATEGY
    max_tokens_per_chunk: int = DEFAULT_MAX_CHUNK_TOKENS
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO
    encoding_name: str = DEFAULT_ENCODING_NAME

    def clamped_to(self, max_input_tokens: int) -> "ChunkingConfig":
        """Cap the chunk size at a safe fraction of the provider's token limit.

        Lowers the chunk *size*, never truncates text: an oversized request
        simply yields more, smaller chunks. Returns ``self`` when already within
        the ceiling, which is the normal case.
        """
        ceiling = int(_CHUNK_TOKEN_SAFETY_MARGIN * max_input_tokens)
        if self.max_tokens_per_chunk <= ceiling:
            return self
        logger.warning(
            "Requested max_tokens_per_chunk=%s exceeds safety ceiling=%s "
            "(%s x max_input_tokens=%s); clamping to %s",
            self.max_tokens_per_chunk,
            ceiling,
            _CHUNK_TOKEN_SAFETY_MARGIN,
            max_input_tokens,
            ceiling,
        )
        return replace(self, max_tokens_per_chunk=ceiling)
