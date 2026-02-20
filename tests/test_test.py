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
        result = run_test([0.5, 0.6, 0.7])
        assert isinstance(result, TestResult)

    def test_too_few_scores(self) -> None:
        result = run_test([0.5])
        assert result.raw_output["is_outlier"] is None
        assert result.raw_output["num_scores"] == 1

    def test_empty_scores(self) -> None:
        result = run_test([])
        assert result.raw_output["num_scores"] == 0

    def test_similar_scores_no_outlier(self) -> None:
        result = run_test([1.0, 1.01, 0.99, 1.0, 1.02])
        assert result.raw_output["is_outlier"] is False

    def test_outlier_score_detected(self) -> None:
        scores = [1.0, 2.0, 1.5, 1.3, 1.7, 1.2, 100.0]
        result = run_test(scores)
        assert result.raw_output["is_outlier"] is True

    def test_output_contains_score_fields(self) -> None:
        result = run_test([1.0, 1.1, 0.9, 1.0])
        assert "score" in result.raw_output
        assert "median_score" in result.raw_output
