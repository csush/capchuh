from capchuh.embed import FastEmbedEmbedder


class TestFastEmbedEmbedder:
    def setup_method(self) -> None:
        self.embedder = FastEmbedEmbedder()

    def test_returns_list_of_floats(self) -> None:
        result = self.embedder.embed("hello world")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embedding_dimension(self) -> None:
        result = self.embedder.embed("hello world")
        assert len(result) == 384

    def test_different_texts_produce_different_embeddings(self) -> None:
        a = self.embedder.embed("the cat sat on the mat")
        b = self.embedder.embed("quantum computing is fascinating")
        assert a != b
