from __future__ import annotations

from typing import Protocol


class Ingestor(Protocol):
    def ingest(self, raw_input: str) -> str:
        """Normalize and validate raw input. Returns clean text."""
        ...


class TextIngestor:
    """Handles plain text input."""

    def ingest(self, raw_input: str) -> str:
        text = raw_input.strip()
        if not text:
            raise ValueError("Input text must not be empty.")
        return text
