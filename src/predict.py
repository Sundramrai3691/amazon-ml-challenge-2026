"""Score candidate pairs and apply a configurable threshold.

Never forces top-1. Empty predictions are valid.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from src.metrics import detailed_entity_metrics
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


def evaluation_row(
    ground_truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    *,
    threshold: float,
) -> dict[str, float]:
    detail = detailed_entity_metrics(ground_truth, predictions)
    fp = 0
    fn = 0
    empty_pred = 0
    n_pred_matches = 0
    singleton_correct = 0
    n_singletons = 0
    for s1, true_ids in ground_truth.items():
        true_set = set(true_ids)
        pred_set = set(predictions.get(s1, ()))
        n_pred_matches += len(pred_set)
        if not pred_set:
            empty_pred += 1
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
        if not true_set:
            n_singletons += 1
            if not pred_set:
                singleton_correct += 1
    n = max(len(ground_truth), 1)
    return {
        "threshold": float(threshold),
        "macro_f0_5": float(detail["macro_f0_5"]),
        "mean_precision": float(detail["macro_precision"]),
        "mean_recall": float(detail["macro_recall"]),
        "singleton_accuracy": (singleton_correct / n_singletons) if n_singletons else float("nan"),
        "false_positive_count": float(fp),
        "false_negative_count": float(fn),
        "n_predicted_empty_s1": float(empty_pred),
        "average_predicted_matches_per_s1": n_pred_matches / n,
    }


def sweep_thresholds(
    pairs: Sequence[tuple[str, str]],
    scores: Sequence[float],
    ground_truth: Mapping[str, Iterable[str]],
    thresholds: Sequence[float],
    *,
    s1_ids: Iterable[str] | None = None,
) -> list[dict[str, float]]:
    """Measure entity-level macro F0.5 across thresholds. Does not assume 0.5."""
    results: list[dict[str, float]] = []
    ids = s1_ids if s1_ids is not None else ground_truth.keys()
    for threshold in thresholds:
        preds = apply_threshold(pairs, scores, threshold=threshold, s1_ids=ids)
        results.append(evaluation_row(ground_truth, preds, threshold=threshold))
    return results


def best_threshold_row(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No threshold rows")
    return dict(max(rows, key=lambda row: (row.get("macro_f0_5", float("-inf")), -row["threshold"])))
