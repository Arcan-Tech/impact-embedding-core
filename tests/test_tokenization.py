from impact_embedding_core.tokenization import DEFAULT_ENCODING_NAME, TiktokenTokenizer


def test_tokenizer_uses_the_shared_default_encoding():
    assert TiktokenTokenizer().encoding_name == DEFAULT_ENCODING_NAME
    assert DEFAULT_ENCODING_NAME == "cl100k_base"


def test_tokenizer_counts_tokens_of_a_known_text():
    # Pins the encoding: a different one would desynchronise chunk sizing from
    # the embeddings already indexed.
    assert TiktokenTokenizer().count_tokens("hello world, this is a test") == 7


def test_tokenizer_counts_no_tokens_in_empty_text():
    assert TiktokenTokenizer().count_tokens("") == 0


def test_tokenizer_counts_more_tokens_for_longer_text():
    tokenizer = TiktokenTokenizer()

    short = tokenizer.count_tokens("hello")
    long = tokenizer.count_tokens("hello world this is a longer sentence")

    assert short >= 1
    assert long > short


def test_tokenizer_accepts_an_explicit_encoding():
    tokenizer = TiktokenTokenizer("p50k_base")

    assert tokenizer.encoding_name == "p50k_base"
    assert tokenizer.count_tokens("hello world") > 0
