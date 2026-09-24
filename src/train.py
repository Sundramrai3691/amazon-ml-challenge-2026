"""Pair-model training: labels from blocking candidates + ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.features import BASELINE_FEATURE_NAMES, feature_matrix, records_by_id
from src.load_data import ground_truth_to_sets

Pair = tuple[str, str]


@dataclass
class PairLabelStats:
    positives: int
    negatives: int
    s1_entities_represented: int
    singleton_entities_represented: int
    s1_with_positive_pairs: int


def build_pair_labels(
    candidates: Mapping[str, Iterable[str]],
    ground_truth: Mapping[str, Iterable[str]],
) -> tuple[list[Pair], np.ndarray, PairLabelStats]:
    """Label each candidate pair. No Cartesian-product negatives.

    label = 1 iff candidate ID is in the S1 ground-truth match set.
    """
    gt_sets = {s1: set(ids) for s1, ids in ground_truth.items()}
    pairs: list[Pair] = []
    labels: list[int] = []
    s1_seen: set[str] = set()
    s1_pos: set[str] = set()
    singletons = 0

    for s1, cand_ids in candidates.items():
        true_ids = gt_sets.get(s1, set())
        if s1 not in s1_seen:
            s1_seen.add(s1)
            if not true_ids:
                singletons += 1
        for cand in dict.fromkeys(list(cand_ids)):
            pairs.append((s1, str(cand)))
            label = 1 if str(cand) in true_ids else 0
            labels.append(label)
            if label == 1:
                s1_pos.add(s1)

    y = np.asarray(labels, dtype=np.int32)
    stats = PairLabelStats(
        positives=int((y == 1).sum()) if len(y) else 0,
        negatives=int((y == 0).sum()) if len(y) else 0,
        s1_entities_represented=len(s1_seen),
        singleton_entities_represented=singletons,
        s1_with_positive_pairs=len(s1_pos),
    )
    return pairs, y, stats


def make_model(model_type: str = "logistic_regression", **params: Any) -> ClassifierMixin:
    """Config-driven sklearn-compatible classifier. No pretrained downloads."""
    model_type = model_type.lower()
    if model_type in {"logistic_regression", "logreg", "lr"}:
        defaults = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}
        defaults.update(params)
        return LogisticRegression(**defaults)
    if model_type in {"decision_tree", "tree"}:
        defaults = {"class_weight": "balanced", "random_state": 42, "max_depth": 8}
        defaults.update(params)
        return DecisionTreeClassifier(**defaults)
    raise ValueError(
        f"Unknown model type {model_type!r}. "
        "Supported now: logistic_regression, decision_tree. "
        "Add LightGBM/XGBoost later only if experiments justify them."
    )


def fit_pair_model(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
) -> ClassifierMixin:
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if len(y) == 0:
        raise ValueError("Cannot fit on zero pairs")
    model.fit(X, y)
    return model


def predict_proba_positive(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba")
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        # Single-class training edge case.
        classes = getattr(model, "classes_", np.array([0, 1]))
        scores = np.zeros(len(X), dtype=np.float64)
        if 1 in set(classes):
            idx = list(classes).index(1)
            scores = proba[:, idx] if proba.ndim == 2 else proba.ravel()
        return scores
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X), dtype=np.float64)


def save_model(model: ClassifierMixin, path: str | Path, *, feature_names: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(feature_names or BASELINE_FEATURE_NAMES)}, path)


def load_model(path: str | Path) -> tuple[ClassifierMixin, list[str]]:
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], list(payload.get("feature_names") or BASELINE_FEATURE_NAMES)
    return payload, list(BASELINE_FEATURE_NAMES)


def build_training_matrix(
    candidates: Mapping[str, Iterable[str]],
    ground_truth_frame_or_map: Any,
    s1_frame: Any,
    s2_frame: Any,
    s3_frame: Any,
    pair_meta: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
):
    """Convenience: labels + features from candidates (hard negatives only)."""
    if hasattr(ground_truth_frame_or_map, "columns"):
        gt = ground_truth_to_sets(ground_truth_frame_or_map)
    else:
        gt = {k: set(v) for k, v in ground_truth_frame_or_map.items()}
    pairs, y, stats = build_pair_labels(candidates, gt)
    s1_records = records_by_id(s1_frame)
    cand_records = records_by_id(s2_frame)
    cand_records.update(records_by_id(s3_frame))
    X_df = feature_matrix(pairs, s1_records, cand_records, pair_meta)
    return X_df, y, pairs, stats


# Extension points for later experiments (documented, unused now):
def mine_hard_negatives(*_args: Any, **_kwargs: Any) -> None:
    """Placeholder: hard-negative mining (E6)."""
    raise NotImplementedError("Hard-negative mining is an E6 experiment, not part of init.")


def source_aware_sample(*_args: Any, **_kwargs: Any) -> None:
    """Placeholder: source-aware pair sampling."""
    raise NotImplementedError("Source-aware sampling is not part of initialization.")
