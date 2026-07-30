# impact-embedding-core

A small library for turning text into vectors: tokenization, chunking,
pooling and an embedding-generation orchestrator, packaged so that anything
that *produces* vectors and anything that *queries* them can run the exact
same pipeline and stay comparable in the same vector space.

## Why this exists

Two independent systems embedding the same kind of content — one indexing
documents, another embedding search queries, for example — need identical
chunking and pooling. The moment they drift (a different chunk size, a
different pooling formula, a different tokenizer encoding), similarity
search degrades silently, with nothing pointing at the cause. This package
is a single source of truth for that pipeline, instead of every consumer
maintaining its own near-identical copy.

The library holds pure logic only: no dependency on a database, a web
framework or any application state.

## The contract

Settings every consumer must agree on for their vectors to land in the same
space. Changing any of these invalidates every vector already generated with
the previous values.

| Setting | Value |
|---|---|
| Embedding model | `qwen3-embedding-8b` (Scaleway) — the default `ScalewayEmbeddingConfig` targets |
| Vector dimensions | 4096 |
| Chunking strategy | `recursive` (separator-aware, tiktoken-measured; `sliding-window` also available) |
| Chunk size | 512 tokens |
| Chunk overlap | 0.1 (10%, ~51 tokens) |
| Tokenizer encoding | `cl100k_base` (tiktoken) |
| Pooling formula | token-count-weighted mean of chunk vectors, then L2-normalized (`max` also available) |

These defaults live in code (`ChunkingConfig`, `ScalewayEmbeddingConfig.model`)
— this table exists so a change to any of them is a conscious decision made
in one place, rather than a one-line edit that silently desyncs whoever else
depends on this library. `tests/test_golden_parity.py` guards the chunking
and pooling half of this contract against a frozen reference fixture
(`tests/fixtures/golden_parity.json`).

## Public API

```python
from impact_embedding_core import (
    ChunkingService,
    EmbeddingService,
    ScalewayEmbeddingClient,
    ScalewayEmbeddingConfig,
)

client = ScalewayEmbeddingClient(
    ScalewayEmbeddingConfig(project_id="...", secret_key="...")
)
embedding_service = EmbeddingService(client, ChunkingService())

result = embedding_service.embed_text("some long document...")
result.pooled   # list[float], the single vector for this text
result.chunks   # list[ChunkEmbedding], one per chunk, for per-chunk storage

# Or embed several texts at once, reusing API requests across them: every
# text maps to exactly one RecordEmbedding, aligned by position.
results = embedding_service.embed_texts(["first document...", "second document..."])
```

`ChunkingService()` and `EmbeddingService(client, chunking_service)` take no
configuration beyond the collaborators they genuinely need: chunking
strategy, chunk size, overlap, tokenizer encoding and pooling strategy are
all fixed to the values in "The contract" above, not constructor arguments.
This is deliberate: the whole point of this library is that every consumer
gets the same vectors, and an exposed knob is a way for one consumer to
quietly stop matching the others. `ScalewayEmbeddingClient`/
`ScalewayEmbeddingConfig` are the one exception — their configuration
(credentials, project, endpoint) has no sensible library default and stays
fully caller-supplied.

Lower-level building blocks (individual chunking strategies, the strategy
registry, `ChunkingConfig`, the tokenizer, pooling internals) are not
re-exported at the package root; import them from their own submodules
(`impact_embedding_core.chunking`, `.embeddings`, `.tokenization`) when
needed — but overriding them yourself defeats the guarantee this package
provides.

## Installing with uv

Add it as a Git dependency pinned to a tag:

```sh
uv add "impact-embedding-core @ git+https://github.com/Arcan-Tech/impact-embedding-core@X.Y.Z"
```

or add it directly to `pyproject.toml`:

```toml
dependencies = [
    "impact-embedding-core @ git+https://github.com/Arcan-Tech/impact-embedding-core@X.Y.Z",
]
```

If your project builds with **hatchling**, also add:

```toml
[tool.hatch.metadata]
allow-direct-references = true
```

Hatchling refuses to build a package whose `project.dependencies` includes a
direct Git URL otherwise — `uv sync`/`uv build` fails with `Dependency ...
cannot be a direct reference unless field
'tool.hatch.metadata.allow-direct-references' is set to 'true'`. This has
nothing to do with authentication; it fails the same way even when the
dependency resolves and clones successfully. Other build backends have their
own equivalent setting for direct URL references in `project.dependencies`.

This repository is public, so the `https://` URL above resolves with no
authentication anywhere — locally, in CI, or in a Docker build. `uv`/`pip`
still shell out to the system `git` to fetch it, so `git` itself needs to be
installed wherever the dependency is resolved (e.g. a minimal Docker base
image needs an explicit `apt-get install git`).

Always pin to an exact tag, never a moving branch: this library has no
compatibility guarantees across versions, and an unpinned dependency lets a
change to the pipeline silently alter every consumer's vectors at once.

## Development

```sh
uv sync
uv run pytest
```
