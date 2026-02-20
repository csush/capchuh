"""Capchuh — measure the generic-ness of text."""

from __future__ import annotations

from capchuh.embed import FastEmbedEmbedder
from capchuh.ingest import TextIngestor
from capchuh.store import InMemoryStore
from capchuh.test import run_test
from capchuh.types import EmbeddingRecord, TestResult

__all__ = [
    "analyze",
    "EmbeddingRecord",
    "TestResult",
]

_ingestor = TextIngestor()
_embedder = FastEmbedEmbedder()
_store = InMemoryStore()


def analyze(text: str) -> TestResult:
    """Full pipeline: ingest → embed → store → test."""
    clean_text = _ingestor.ingest(text)
    embedding = _embedder.embed(clean_text)
    record = EmbeddingRecord(id="", raw_input=clean_text, embedding=embedding)
    _store.save(record)

    all_embeddings = [r.embedding for r in _store.list_all()]
    return run_test(all_embeddings)
