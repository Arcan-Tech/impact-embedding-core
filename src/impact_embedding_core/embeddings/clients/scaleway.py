"""Scaleway implementation of the EmbeddingClient role.

Scaleway's Generative API is OpenAI-compatible, so we drive it with the
``openai`` SDK pointed at ``{base_url}/{project_id}/v1``; the SDK's built-in
retry covers 429/5xx/connection failures.

The only module in this package that knows a concrete provider: another one
means another sibling here, and no change to the services that use it.
"""

import logging
from dataclasses import asdict, dataclass

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError

from impact_embedding_core.embeddings.base import (
    Embedding,
    EmbeddingError,
    RetryableEmbeddingError,
)
from impact_embedding_core.tokenization import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3-embedding-8b"
DEFAULT_BASE_URL = "https://api.scaleway.ai"


@dataclass(frozen=True)
class ScalewayEmbeddingConfig:
    """Everything the client needs, defaulted to the values the graph was built with.

    ``project_id`` and ``secret_key`` are the only required arguments:
    ``project_id`` identifies the deployment and is part of the endpoint URL, so
    there is no sensible default. ``secret_key`` is a credential, and this
    library never reads the environment on a caller's behalf — each consumer
    resolves it from wherever it keeps secrets and passes it in explicitly.
    """

    project_id: str
    secret_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    encoding_format: str = "float"
    dimensions: int | None = None
    max_inputs: int = 2048
    max_input_tokens: int = 40960
    provider: str = "scaleway"

    def __str__(self) -> str:
        description = asdict(self)
        description["secret_key"] = "***"
        return str(description)


class ScalewayEmbeddingClient:
    """Scaleway implementation of the embedding client role."""

    def __init__(
        self,
        config: ScalewayEmbeddingConfig,
        tokenizer: Tokenizer | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        config:      Endpoint, model and provider limits.
        tokenizer:   Optional guard that rejects inputs over the token limit.
        timeout:     Per-request timeout, in seconds.
        max_retries: SDK-level retries for transient failures.
        """
        self._config = config
        self._tokenizer = tokenizer
        self._client = OpenAI(
            base_url=self._embeddings_base_url(),
            api_key=config.secret_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def config(self) -> ScalewayEmbeddingConfig:
        return self._config

    def generate_embeddings(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        if len(texts) > self._config.max_inputs:
            raise EmbeddingError(
                "Scaleway embedding request exceeds max_inputs "
                f"({len(texts)} > {self._config.max_inputs})",
            )
        self._ensure_within_token_limit(texts)

        logger.debug(
            "Generating Scaleway embeddings inputs=%s model=%s dimensions=%s",
            len(texts),
            self._config.model,
            self._config.dimensions,
        )
        return self._normalize_response(self._request_embeddings(texts), texts)

    def _embeddings_base_url(self) -> str:
        base_url = self._config.base_url.rstrip("/")
        return f"{base_url}/{self._config.project_id}/v1"

    def _ensure_within_token_limit(self, texts: list[str]) -> None:
        """Guard against over-long inputs slipping through.

        An input past the limit means the chunker has a bug, so we fail loudly
        instead of truncating and losing text. Without an injected tokenizer the
        provider's own length check remains the backstop.
        """
        if self._tokenizer is None:
            return
        for index, text in enumerate(texts):
            token_count = self._tokenizer.count_tokens(text)
            if token_count > self._config.max_input_tokens:
                raise EmbeddingError(
                    f"Scaleway embedding input index={index} exceeds "
                    f"max_input_tokens ({token_count} > "
                    f"{self._config.max_input_tokens}); chunking should have "
                    "prevented this",
                )

    def _request_embeddings(self, texts: list[str]):
        """Call the embedding API, mapping SDK failures onto our error types.

        The SDK has already exhausted its own retries here, so
        ``RetryableEmbeddingError`` reports that the failure was transient in
        nature, not that another attempt is due.
        """
        kwargs = {
            "model": self._config.model,
            "input": texts,
            "encoding_format": self._config.encoding_format,
        }
        if self._config.dimensions is not None:
            kwargs["dimensions"] = self._config.dimensions

        try:
            return self._client.embeddings.create(**kwargs)
        except APIStatusError as exc:
            logger.exception(
                "Scaleway embedding API returned HTTP status=%s inputs=%s",
                exc.status_code,
                len(texts),
            )
            message = (
                f"Embedding API returned HTTP {exc.status_code} for "
                f"model {self._config.model}"
            )
            detail = self._extract_error_message(exc.body)
            if detail:
                message = f"{message}: {detail}"
            if self._is_retryable_status(exc.status_code):
                raise RetryableEmbeddingError(message) from exc
            raise EmbeddingError(message) from exc
        except APIConnectionError as exc:
            logger.exception(
                "Scaleway embedding API request failed inputs=%s", len(texts)
            )
            raise RetryableEmbeddingError(
                f"Embedding API request failed for model {self._config.model}",
            ) from exc
        except OpenAIError as exc:
            logger.exception("Scaleway embedding call failed inputs=%s", len(texts))
            raise EmbeddingError(
                f"Embedding API call failed for model {self._config.model}",
            ) from exc

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        """429 and 5xx are transient; other 4xx are not."""
        return status_code == 429 or status_code >= 500

    @staticmethod
    def _extract_error_message(body: object | None) -> str | None:
        """Pull a human-readable message out of Scaleway's error body.

        ``body`` is ``APIStatusError.body`` — the SDK's own parse of the
        response: a dict when it decoded as JSON, the raw response text
        otherwise, or ``None`` when the response could not be read at all (the
        SDK already handles the closed-stream edge case that arises from
        reading a response body twice, so this reuses its parse rather than
        re-reading ``exc.response`` here). Scaleway nests its message under
        ``error.message``; a top-level ``message`` is the fallback shape used
        by some error responses, and the body itself is the last resort so a
        message in an unrecognised shape is still surfaced rather than
        silently dropped.
        """
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
            message = body.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
            return str(body)
        if isinstance(body, str):
            text = body.strip()
            return text or None
        return None

    def _normalize_response(self, response, inputs: list[str]) -> list[Embedding]:
        """Reorder the response onto the input order, failing on any gap.

        Input order is not guaranteed, so results are keyed by ``index``. A
        missing, duplicated or out-of-range index would misalign vectors with
        their source text, so each case raises.
        """
        data = getattr(response, "data", None)
        if not isinstance(data, list) or not data:
            raise EmbeddingError("Scaleway embedding response is missing data[0]")

        embeddings_by_index: dict[int, list[float]] = {}
        for item in data:
            index = getattr(item, "index", None)
            if not isinstance(index, int):
                raise EmbeddingError(
                    "Scaleway embedding response item is missing index"
                )
            if index < 0 or index >= len(inputs):
                raise EmbeddingError(
                    f"Scaleway embedding response index {index} is out of range",
                )
            if index in embeddings_by_index:
                raise EmbeddingError(
                    f"Scaleway embedding response contains duplicate index {index}",
                )

            vector = getattr(item, "embedding", None)
            if not isinstance(vector, list):
                raise EmbeddingError(
                    "Scaleway embedding response item is missing embedding",
                )

            embeddings_by_index[index] = [float(value) for value in vector]

        missing_indexes = [
            index for index in range(len(inputs)) if index not in embeddings_by_index
        ]
        if missing_indexes:
            raise EmbeddingError(
                f"Scaleway embedding response is missing indexes {missing_indexes}",
            )

        return [
            Embedding(input=input_text, vector=embeddings_by_index[index])
            for index, input_text in enumerate(inputs)
        ]
