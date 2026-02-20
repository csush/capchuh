from __future__ import annotations

import numpy as np

from capchuh.types import TestResult


def _is_outlier(x: np.ndarray, k: float = 3.5) -> bool:
    """Modified Z-score outlier detection using MAD. Checks the last element."""
    m = np.median(x)
    mad = np.median(np.abs(x - m))
    if mad == 0:
        return False
    return bool(abs(0.6745 * (x[-1] - m) / mad) > k)


def run_test(scores: list[float]) -> TestResult:
    """
    Run the deterministic Turing test on a list of scalar scores.

    Checks if the latest score (last element) is an outlier via MAD-based
    modified Z-score. Outliers are likely human-written (more unique);
    non-outliers are likely AI-generated (more generic).
    """
    n = len(scores)

    if n < 3:
        return TestResult(raw_output={
            "is_outlier": None,
            "num_scores": n,
            "message": "Need at least 3 scores to run the test.",
        })

    x = np.array(scores)
    outlier = _is_outlier(x)

    return TestResult(raw_output={
        "is_outlier": outlier,
        "score": float(x[-1]),
        "median_score": float(np.median(x)),
        "num_scores": n,
    })
