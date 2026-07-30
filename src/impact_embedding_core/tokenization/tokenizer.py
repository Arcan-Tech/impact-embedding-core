from typing import Protocol


class Tokenizer(Protocol):
    """Counts tokens in a text, to weight chunks when pooling their vectors."""

    def count_tokens(self, text: str) -> int: ...
