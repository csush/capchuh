from __future__ import annotations

import uuid
from typing import Protocol

from capchuh.types import EmbeddingRecord


class Store(Protocol):
    def save(self, record: EmbeddingRecord) -> str:
        """Persist an embedding record. Returns an ID."""
        ...

    def get(self, record_id: str) -> EmbeddingRecord | None: ...

    def list_all(self) -> list[EmbeddingRecord]: ...


class InMemoryStore:
    """Dict-backed store. Swap for SQLite/Postgres later."""

    def __init__(self) -> None:
        self._data: dict[str, EmbeddingRecord] = {}

    def save(self, record: EmbeddingRecord) -> str:
        if not record.id:
            record.id = uuid.uuid4().hex
        self._data[record.id] = record
        return record.id

    def get(self, record_id: str) -> EmbeddingRecord | None:
        return self._data.get(record_id)

    def list_all(self) -> list[EmbeddingRecord]:
        return list(self._data.values())
