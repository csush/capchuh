import pytest

from capchuh.ingest import TextIngestor


class TestTextIngestor:
    def setup_method(self) -> None:
        self.ingestor = TextIngestor()

    def test_strips_whitespace(self) -> None:
        assert self.ingestor.ingest("  hello world  ") == "hello world"

    def test_returns_clean_text(self) -> None:
        assert self.ingestor.ingest("some text") == "some text"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.ingestor.ingest("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.ingestor.ingest("   ")
