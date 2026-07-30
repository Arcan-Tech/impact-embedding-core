from langchain_text_splitters import TokenTextSplitter

from impact_embedding_core.chunking.base import Chunk
from impact_embedding_core.chunking.base import build_chunks
from impact_embedding_core.chunking.base import resolve_overlap_tokens
from impact_embedding_core.tokenization import DEFAULT_ENCODING_NAME
from impact_embedding_core.tokenization import Tokenizer


class SlidingWindowChunkingStrategy:
    """Fixed-size token window with overlap, blind to boundaries.

    The naive baseline: it can cut mid-sentence or mid-word. Callers only ever
    see our own ``Chunk`` type, so the splitter library stays swappable.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_tokens_per_chunk: int,
        overlap_ratio: float,
        encoding_name: str = DEFAULT_ENCODING_NAME,
    ) -> None:
        self._tokenizer = tokenizer
        overlap_tokens = resolve_overlap_tokens(max_tokens_per_chunk, overlap_ratio)
        self._splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=max_tokens_per_chunk,
            chunk_overlap=overlap_tokens,
        )

    def chunk(self, text: str) -> list[Chunk]:
        return build_chunks(self._splitter.split_text(text), text, self._tokenizer)
