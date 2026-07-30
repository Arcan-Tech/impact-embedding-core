from impact_embedding_core.services.chunking import ChunkingService


def test_chunking_service_returns_the_whole_text_as_one_chunk_when_it_fits():
    service = ChunkingService()

    chunks = service.chunk("The quick brown fox jumps over the lazy dog.")

    assert len(chunks) == 1
    assert chunks[0].text == "The quick brown fox jumps over the lazy dog."


def test_chunking_service_splits_long_text_within_the_standard_chunk_size():
    service = ChunkingService()

    chunks = service.chunk(("word " * 1200).strip())

    assert len(chunks) > 1
    assert all(chunk.token_count <= 512 for chunk in chunks)


def test_chunking_service_chunks_many_texts_preserving_order():
    service = ChunkingService()

    results = service.chunk_many(["a", "b c"])

    assert [len(chunks) for chunks in results] == [1, 1]
    assert results[0][0].text == "a"
    assert results[1][0].text == "b c"


def test_chunking_service_returns_an_empty_list_for_no_texts():
    assert ChunkingService().chunk_many([]) == []
