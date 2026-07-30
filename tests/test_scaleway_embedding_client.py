from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from impact_embedding_core.embeddings import EmbeddingError, RetryableEmbeddingError
from impact_embedding_core.embeddings.clients import (
    ScalewayEmbeddingClient,
    ScalewayEmbeddingConfig,
)


def _config(**kwargs):
    return ScalewayEmbeddingConfig(
        **{"project_id": "proj-1", "secret_key": "secret", **kwargs}
    )


def _response(*items):
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in items]
    )


def _client(monkeypatch, config=None, tokenizer=None, response=None, side_effect=None):
    """Build a client whose OpenAI SDK object is a MagicMock."""
    openai_class = MagicMock()
    monkeypatch.setattr(
        "impact_embedding_core.embeddings.clients.scaleway.OpenAI", openai_class
    )
    client = ScalewayEmbeddingClient(config or _config(), tokenizer=tokenizer)
    create = openai_class.return_value.embeddings.create
    if side_effect is not None:
        create.side_effect = side_effect
    else:
        create.return_value = (
            response if response is not None else _response((0, [1.0]))
        )
    return client, openai_class, create


def _status_error(status_code, body=None):
    request = httpx.Request("POST", "https://api.scaleway.ai")
    return APIStatusError(
        "boom",
        response=httpx.Response(status_code, request=request),
        body=body,
    )


def test_client_points_the_sdk_at_the_project_scoped_endpoint(monkeypatch):
    _, openai_class, _ = _client(monkeypatch)

    assert openai_class.call_args.kwargs["base_url"] == (
        "https://api.scaleway.ai/proj-1/v1"
    )
    assert openai_class.call_args.kwargs["api_key"] == "secret"


def test_client_returns_vectors_in_input_order(monkeypatch):
    # The API is not required to preserve input order.
    client, _, _ = _client(monkeypatch, response=_response((1, [2.0]), (0, [1.0])))

    embeddings = client.generate_embeddings(["first", "second"])

    assert [embedding.input for embedding in embeddings] == ["first", "second"]
    assert [embedding.vector for embedding in embeddings] == [[1.0], [2.0]]


def test_client_skips_the_api_call_for_no_inputs(monkeypatch):
    client, _, create = _client(monkeypatch)

    assert client.generate_embeddings([]) == []
    create.assert_not_called()


def test_client_rejects_more_inputs_than_the_provider_accepts(monkeypatch):
    client, _, create = _client(monkeypatch, config=_config(max_inputs=2))

    with pytest.raises(EmbeddingError, match="max_inputs"):
        client.generate_embeddings(["a", "b", "c"])
    create.assert_not_called()


def test_client_fails_loudly_on_an_input_over_the_token_limit(monkeypatch):
    # Never truncate: cutting here would silently drop text.
    tokenizer = SimpleNamespace(count_tokens=lambda text: len(text))
    client, _, create = _client(
        monkeypatch, config=_config(max_input_tokens=3), tokenizer=tokenizer
    )

    with pytest.raises(EmbeddingError, match="max_input_tokens"):
        client.generate_embeddings(["too long"])
    create.assert_not_called()


def test_client_accepts_an_input_within_the_token_limit(monkeypatch):
    tokenizer = SimpleNamespace(count_tokens=lambda text: len(text))
    client, _, _ = _client(
        monkeypatch, config=_config(max_input_tokens=10), tokenizer=tokenizer
    )

    assert len(client.generate_embeddings(["ok"])) == 1


def test_client_omits_dimensions_when_not_configured(monkeypatch):
    client, _, create = _client(monkeypatch)

    client.generate_embeddings(["text"])

    assert "dimensions" not in create.call_args.kwargs
    assert create.call_args.kwargs["model"] == "qwen3-embedding-8b"
    assert create.call_args.kwargs["encoding_format"] == "float"


def test_client_requests_the_configured_dimensions(monkeypatch):
    client, _, create = _client(monkeypatch, config=_config(dimensions=1024))

    client.generate_embeddings(["text"])

    assert create.call_args.kwargs["dimensions"] == 1024


def test_client_rejects_a_response_without_data(monkeypatch):
    client, _, _ = _client(monkeypatch, response=SimpleNamespace(data=[]))

    with pytest.raises(EmbeddingError, match="missing data"):
        client.generate_embeddings(["text"])


def test_client_rejects_a_response_with_a_missing_index(monkeypatch):
    client, _, _ = _client(monkeypatch, response=_response((0, [1.0])))

    with pytest.raises(EmbeddingError, match="missing indexes"):
        client.generate_embeddings(["first", "second"])


def test_client_rejects_a_response_with_a_duplicate_index(monkeypatch):
    client, _, _ = _client(monkeypatch, response=_response((0, [1.0]), (0, [2.0])))

    with pytest.raises(EmbeddingError, match="duplicate index"):
        client.generate_embeddings(["first", "second"])


def test_client_rejects_a_response_with_an_out_of_range_index(monkeypatch):
    client, _, _ = _client(monkeypatch, response=_response((5, [1.0])))

    with pytest.raises(EmbeddingError, match="out of range"):
        client.generate_embeddings(["only"])


def test_client_rejects_a_response_item_without_a_vector(monkeypatch):
    response = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=None)])
    client, _, _ = _client(monkeypatch, response=response)

    with pytest.raises(EmbeddingError, match="missing embedding"):
        client.generate_embeddings(["only"])


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_client_reports_transient_http_failures_as_retryable(monkeypatch, status_code):
    client, _, _ = _client(monkeypatch, side_effect=_status_error(status_code))

    with pytest.raises(RetryableEmbeddingError):
        client.generate_embeddings(["text"])


@pytest.mark.parametrize("status_code", [400, 401, 404])
def test_client_reports_client_side_http_failures_as_permanent(
    monkeypatch, status_code
):
    client, _, _ = _client(monkeypatch, side_effect=_status_error(status_code))

    with pytest.raises(EmbeddingError) as exc_info:
        client.generate_embeddings(["text"])
    assert not isinstance(exc_info.value, RetryableEmbeddingError)


def test_client_reports_connection_failures_as_retryable(monkeypatch):
    error = APIConnectionError(request=httpx.Request("POST", "https://api.scaleway.ai"))
    client, _, _ = _client(monkeypatch, side_effect=error)

    with pytest.raises(RetryableEmbeddingError):
        client.generate_embeddings(["text"])


def test_client_includes_the_scaleway_error_message_when_present(monkeypatch):
    # Covers how the openai SDK surfaces the Scaleway error body.
    body = {
        "error": {
            "message": (
                "This model's maximum context length is 40960 tokens. "
                "However, your request has 43813 input tokens."
            ),
            "type": "BadRequestError",
            "param": None,
            "code": 400,
        },
    }
    client, _, _ = _client(monkeypatch, side_effect=_status_error(400, body=body))

    with pytest.raises(EmbeddingError, match="maximum context length is 40960 tokens"):
        client.generate_embeddings(["text"])


def test_client_falls_back_to_a_top_level_message_field(monkeypatch):
    body = {"message": "top-level error, no nested error object"}
    client, _, _ = _client(monkeypatch, side_effect=_status_error(400, body=body))

    with pytest.raises(EmbeddingError, match="top-level error, no nested error object"):
        client.generate_embeddings(["text"])


def test_client_falls_back_to_the_raw_body_when_no_message_field_is_found(monkeypatch):
    body = {"unexpected": "shape"}
    client, _, _ = _client(monkeypatch, side_effect=_status_error(400, body=body))

    with pytest.raises(EmbeddingError, match="unexpected"):
        client.generate_embeddings(["text"])


def test_client_omits_the_detail_suffix_when_the_body_carries_no_message(monkeypatch):
    client, _, _ = _client(monkeypatch, side_effect=_status_error(400, body=None))

    with pytest.raises(EmbeddingError) as exc_info:
        client.generate_embeddings(["text"])

    assert str(exc_info.value) == (
        "Embedding API returned HTTP 400 for model qwen3-embedding-8b"
    )


def test_config_masks_the_secret_key_when_printed():
    rendered = str(
        ScalewayEmbeddingConfig(project_id="proj-1", secret_key="SCW-do-not-log-me")
    )

    assert "SCW-do-not-log-me" not in rendered
    assert "***" in rendered
    assert "proj-1" in rendered


def test_config_needs_project_id_and_secret_key_and_defaults_the_rest():
    config = ScalewayEmbeddingConfig(project_id="proj-1", secret_key="secret")

    assert config.model == "qwen3-embedding-8b"
    assert config.base_url == "https://api.scaleway.ai"
    assert config.encoding_format == "float"
    assert config.dimensions is None
    assert config.max_inputs == 2048
    assert config.max_input_tokens == 40960
    assert config.provider == "scaleway"


def test_config_accepts_overrides_per_field():
    config = ScalewayEmbeddingConfig(
        project_id="proj-1",
        secret_key="secret",
        model="other-model",
        base_url="https://proxy.internal",
        dimensions=512,
        max_inputs=64,
        max_input_tokens=8192,
    )

    assert config.model == "other-model"
    assert config.base_url == "https://proxy.internal"
    assert config.dimensions == 512
    assert config.max_inputs == 64
    assert config.max_input_tokens == 8192


def test_client_honours_an_overridden_base_url(monkeypatch):
    _, openai_class, _ = _client(
        monkeypatch, config=_config(base_url="https://proxy.internal/")
    )

    assert openai_class.call_args.kwargs["base_url"] == (
        "https://proxy.internal/proj-1/v1"
    )
