import numpy as np

from capchuh.test import _is_outlier, run_test
from capchuh.types import TestResult


class TestIsOutlier:
    def test_outlier_detected(self) -> None:
        x = np.array([1.0, 2.0, 1.5, 1.3, 1.7, 1.2, 100.0])
        assert _is_outlier(x) is True

    def test_no_outlier(self) -> None:
        x = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
        assert _is_outlier(x) is False

    def test_zero_mad_returns_false(self) -> None:
        x = np.array([1.0, 1.0, 1.0, 1.0])
        assert _is_outlier(x) is False


class TestRunTest:
    def test_returns_test_result(self) -> None:
        result = run_test([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        assert isinstance(result, TestResult)

    def test_too_few_embeddings(self) -> None:
        result = run_test([[0.1, 0.2]])
        assert result.raw_output["is_outlier"] is None
        assert result.raw_output["num_embeddings"] == 1

    def test_empty_embeddings(self) -> None:
        result = run_test([])
        assert result.raw_output["num_embeddings"] == 0

    def test_similar_embeddings_no_outlier(self) -> None:
        vecs = [[1.0, 0.0, 0.0]] * 5
        result = run_test(vecs)
        assert result.raw_output["is_outlier"] is False

    def test_outlier_embedding_detected(self) -> None:
        rng = np.random.default_rng(42)
        cluster = (rng.standard_normal((20, 32)) * 0.01 + 1.0).tolist()
        outlier = (-1.0 * np.ones(32)).tolist()
        result = run_test(cluster + [outlier])
        assert result.raw_output["is_outlier"] is True
