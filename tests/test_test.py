from capchuh.test import run_test
from capchuh.types import TestResult


class TestRunTest:
    def test_returns_test_result(self) -> None:
        result = run_test([[0.1, 0.2], [0.3, 0.4]])
        assert isinstance(result, TestResult)

    def test_placeholder_output(self) -> None:
        result = run_test([[0.1], [0.2], [0.3]])
        assert result.raw_output["status"] == "placeholder"
        assert result.raw_output["num_embeddings"] == 3

    def test_empty_embeddings(self) -> None:
        result = run_test([])
        assert result.raw_output["num_embeddings"] == 0
