from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EmbeddingRecord:
    id: str
    raw_input: str
    embedding: list[float]
    score: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TestResult:
    raw_output: dict[str, Any] = field(default_factory=dict)
