"""Entity-level macro F0.5 (primary challenge metric)."""

from __future__ import annotations

from typing import Iterable, Mapping

BETA = 0.5
BETA2 = BETA * BETA  # 0.25
NUM_COEF = 1.0 + BETA2  # 1.25

IdSet = set[str]
PredictionMap = Mapping[str, Iterable[str]]


def _as_set(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    return {str(v) for v in values}


def precision_recall(true_ids: Iterable[str], pred_ids: Iterable[str]) -> tuple[float, float]:
    true_set = _as_set(true_ids)
    pred_set = _as_set(pred_ids)
    if not true_set and not pred_set:
        return 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0
    if not true_set:
        return 0.0, 0.0
    overlap = len(true_set & pred_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(true_set)
    return precision, recall


def entity_f05(true_ids: Iterable[str], pred_ids: Iterable[str]) -> float:
    """F0.5 for a single Source 1 entity.

    Empty truth and empty prediction => 1.0
    Empty prediction with non-empty truth => 0.0
    Non-empty prediction with empty truth => 0.0
    """
    precision, recall = precision_recall(true_ids, pred_ids)
    denom = (BETA2 * precision) + recall
    if denom == 0.0:
        return 0.0
    return (NUM_COEF * precision * recall) / denom


def macro_f05(
    ground_truth: PredictionMap,
    predictions: PredictionMap,
) -> float:
    """Macro-average entity F0.5 over all Source 1 IDs in ground_truth."""
    if not ground_truth:
        return 0.0
    scores = [
        entity_f05(true_ids, predictions.get(s1, ()))
        for s1, true_ids in ground_truth.items()
    ]
    return float(sum(scores) / len(scores))


def detailed_entity_metrics(
    ground_truth: PredictionMap,
    predictions: PredictionMap,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f05s: list[float] = []
    for s1, true_ids in ground_truth.items():
        pred = predictions.get(s1, ())
        precision, recall = precision_recall(true_ids, pred)
        score = entity_f05(true_ids, pred)
        precisions.append(precision)
        recalls.append(recall)
        f05s.append(score)
        rows.append(
            {
                "source1_entity_id": s1,
                "n_true": len(_as_set(true_ids)),
                "n_pred": len(_as_set(pred)),
                "precision": precision,
                "recall": recall,
                "f0_5": score,
                "is_singleton": len(_as_set(true_ids)) == 0,
            }
        )
    n = len(rows) or 1
    return {
        "per_entity": rows,
        "macro_precision": float(sum(precisions) / n) if rows else 0.0,
        "macro_recall": float(sum(recalls) / n) if rows else 0.0,
        "macro_f0_5": float(sum(f05s) / n) if rows else 0.0,
        "n_entities": len(rows),
        "n_singletons": sum(1 for row in rows if row["is_singleton"]),
    }
