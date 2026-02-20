# AGENTS.md

## Build & Test
- Package manager: `uv` (never use pip directly)
- Install: `uv sync`
- Run all tests: `uv run pytest`
- Run single test: `uv run pytest tests/test_ingest.py::TestTextIngestor::test_strips_whitespace`
- Lint: `uv run ruff check .`

## Architecture
- **Pipeline**: `analyze(text)` in `__init__.py` — ingest → embed → score (L2 norm) → store → `run_test(scores)`
- **ingest.py**: `TextIngestor` — validates/strips input text. `Ingestor` protocol for extensibility.
- **embed.py**: `FastEmbedEmbedder` — BAAI/bge-small-en-v1.5 via fastembed (ONNX). `Embedder` protocol.
- **test.py**: `run_test(list[float])` — MAD-based modified Z-score outlier detection on scalar scores.
- **store.py**: `InMemoryStore` — dict-backed. `Store` protocol for future DB swap.
- **types.py**: `EmbeddingRecord` (id, raw_input, embedding, score, created_at), `TestResult` (raw_output dict).
- **cli.py**: `python -m capchuh "text"` entry point.

## Code Style
- Python 3.10+. Use `from __future__ import annotations` in every module.
- Protocol-based interfaces for all layers. Implementations are concrete classes.
- Type hints on all function signatures. Dataclasses for data types.
- Minimal dependencies: fastembed, numpy. Dev: pytest, ruff.
