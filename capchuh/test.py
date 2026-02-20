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


def run_test(embeddings: list[list[float]]) -> TestResult:
    """
    Run the deterministic Turing test on a set of embeddings.

    Computes cosine similarity of each embedding to the centroid,
    then checks if the latest embedding is an outlier via MAD-based
    modified Z-score. Outliers are likely human-written (more unique);
    non-outliers are likely AI-generated (more generic).
    """
    n = len(embeddings)

    if n < 3:
        return TestResult(raw_output={
            "is_outlier": None,
            "num_embeddings": n,
            "message": "Need at least 3 embeddings to run the test.",
        })

    vecs = np.array(embeddings)
    centroid = vecs.mean(axis=0)

    norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(centroid)
    norms = np.where(norms == 0, 1.0, norms)
    similarities = vecs @ centroid / norms

    outlier = _is_outlier(similarities)

    return TestResult(raw_output={
        "is_outlier": outlier,
        "similarity": float(similarities[-1]),
        "median_similarity": float(np.median(similarities)),
        "num_embeddings": n,
    })
