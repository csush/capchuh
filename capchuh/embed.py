from __future__ import annotations

from typing import Protocol

from fastembed import TextEmbedding


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector from text."""
        ...


class FastEmbedEmbedder:
    """Uses fastembed with BAAI/bge-small-en-v1.5 (ONNX, no GPU required)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()
