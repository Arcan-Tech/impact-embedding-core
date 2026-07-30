import tiktoken

# Fixed encoding shared by the splitters (to size chunks) and by token counting
# (to weight them), so the two always agree. Generic, not the embedding
# provider's own tokenizer: the gap is absorbed by the safety margin in
# impact_embedding_core.chunking.config, not by exactness.
DEFAULT_ENCODING_NAME = "cl100k_base"


class TiktokenTokenizer:
    """Tokenizer backed by a standard, offline tiktoken encoding."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING_NAME) -> None:
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def encoding_name(self) -> str:
        return self._encoding_name

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))
