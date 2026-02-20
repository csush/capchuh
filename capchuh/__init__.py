"""Capchuh — measure the generic-ness of text."""

from __future__ import annotations

import numpy as np

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
    """Full pipeline: ingest → embed → score → store → test."""
    clean_text = _ingestor.ingest(text)
    embedding = _embedder.embed(clean_text)
    score = float(np.linalg.norm(embedding))
    record = EmbeddingRecord(id="", raw_input=clean_text, embedding=embedding, score=score)
    _store.save(record)

    all_scores = [r.score for r in _store.list_all()]
    return run_test(all_scores)
