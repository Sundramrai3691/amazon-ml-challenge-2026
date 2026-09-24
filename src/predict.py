"""Score candidate pairs and apply a configurable threshold.

Never forces top-1. Empty predictions are valid.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from src.metrics import macro_f05
from src.train import predict_proba_positive

CandidateMap = dict[str, list[str]]


def apply_threshold(
    pairs: Sequence[tuple[str, str]],
    scores: Sequence[float],
    *,
    threshold: float,
    s1_ids: Iterable[str] | None = None,
) -> CandidateMap:
    """Keep pairs with score >= threshold. No top-1 fallback."""
    matches: dict[str, list[str]] = {}
    if s1_ids is not None:
        for s1 in s1_ids:
            matches[str(s1)] = []
    for (s1, cand), score in zip(pairs, scores):
        matches.setdefault(s1, [])
        if float(score) >= float(threshold):
            if cand not in matches[s1]:
                matches[s1].append(cand)
    for s1 in matches:
        matches[s1] = sorted(dict.fromkeys(matches[s1]))
    return matches


def score_pairs(model, X) -> np.ndarray:
    return predict_proba_positive(model, np.asarray(X, dtype=np.float64))


def predict_matches(
    model,
    X,
    pairs: Sequence[tuple[str, str]],
    *,
    threshold: float,
    s1_ids: Iterable[str] | None = None,
) -> tuple[np.ndarray, CandidateMap]:
    scores = score_pairs(model, X)
    matches = apply_threshold(pairs, scores, threshold=threshold, s1_ids=s1_ids)
    return scores, matches


def sweep_thresholds(
    pairs: Sequence[tuple[str, str]],
    scores: Sequence[float],
    ground_truth: Mapping[str, Iterable[str]],
    thresholds: Sequence[float],
    *,
    s1_ids: Iterable[str] | None = None,
) -> list[dict[str, float]]:
    """Validation helper: measure macro F0.5 across thresholds. Does not assume 0.5."""
    results: list[dict[str, float]] = []
    ids = s1_ids if s1_ids is not None else ground_truth.keys()
    for threshold in thresholds:
        preds = apply_threshold(pairs, scores, threshold=threshold, s1_ids=ids)
        results.append(
            {
                "threshold": float(threshold),
                "macro_f0_5": macro_f05(ground_truth, preds),
            }
        )
    return results
