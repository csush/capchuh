from capchuh.store import InMemoryStore
from capchuh.types import EmbeddingRecord


class TestInMemoryStore:
    def setup_method(self) -> None:
        self.store = InMemoryStore()

    def _make_record(self, **kwargs) -> EmbeddingRecord:
        defaults = {"id": "", "raw_input": "hello", "embedding": [0.1, 0.2]}
        defaults.update(kwargs)
        return EmbeddingRecord(**defaults)

    def test_save_and_get(self) -> None:
        record = self._make_record()
        record_id = self.store.save(record)
        assert record_id
        retrieved = self.store.get(record_id)
        assert retrieved is record

    def test_get_nonexistent_returns_none(self) -> None:
        assert self.store.get("nonexistent") is None

    def test_list_all(self) -> None:
        self.store.save(self._make_record(raw_input="a"))
        self.store.save(self._make_record(raw_input="b"))
        assert len(self.store.list_all()) == 2

    def test_save_preserves_existing_id(self) -> None:
        record = self._make_record(id="my-id")
        returned_id = self.store.save(record)
        assert returned_id == "my-id"
        assert self.store.get("my-id") is record
