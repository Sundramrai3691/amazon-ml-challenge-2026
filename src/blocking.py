"""Candidate generation (blocking). Matching is a separate stage."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.normalize import normalize_address, normalize_name, strip_legal_suffixes

CandidateMap = dict[str, list[str]]
BlockIndex = dict[str, list[str]]

BASELINE_METHODS = (
    "exact_normalized_name",
    "exact_name_without_legal_suffix",
    "exact_normalized_address",
    "exact_name_and_address",
)

# Extension-point names (not implemented in initialization):
FUTURE_METHODS = (
    "char_tfidf",
    "word_tfidf",
    "numeric_address",
    "postcode_token",
    "fuzzy",
    "source_aware",
)


def _valid_candidate_id(entity_id: str) -> bool:
    return entity_id.startswith("S2-") or entity_id.startswith("S3-")


def _name_key(name: object) -> str:
    return normalize_name(name)


def _suffix_free_key(name: object) -> str:
    return strip_legal_suffixes(name)


def _address_key(address: object) -> str:
    return normalize_address(address)


def _combined_key(name: object, address: object) -> str:
    return f"{_name_key(name)}||{_address_key(address)}"


def build_block_index(
    frame: pd.DataFrame,
    key_fn: Callable[[pd.Series], str],
) -> BlockIndex:
    index: BlockIndex = defaultdict(list)
    for row in frame.itertuples(index=False):
        series = pd.Series(row._asdict()) if hasattr(row, "_asdict") else None
        # itertuples with name=None is faster; use column access via namedtuples.
        entity_id = str(getattr(row, "entity_id"))
        if not _valid_candidate_id(entity_id):
            continue
        if series is None:
            key = key_fn(pd.Series({"entity_id": entity_id,
                                    "business_name": getattr(row, "business_name", ""),
                                    "business_address": getattr(row, "business_address", "")}))
        else:
            key = key_fn(series)
        if not key:
            continue
        index[key].append(entity_id)
    for key, ids in index.items():
        index[key] = list(dict.fromkeys(ids))
    return dict(index)


def _key_from_name(row: pd.Series) -> str:
    return _name_key(row.get("business_name", ""))


def _key_from_name_suffix(row: pd.Series) -> str:
    return _suffix_free_key(row.get("business_name", ""))


def _key_from_address(row: pd.Series) -> str:
    return _address_key(row.get("business_address", ""))


def _key_from_combined(row: pd.Series) -> str:
    return _combined_key(row.get("business_name", ""), row.get("business_address", ""))


METHOD_KEY_FNS: dict[str, Callable[[pd.Series], str]] = {
    "exact_normalized_name": _key_from_name,
    "exact_name_without_legal_suffix": _key_from_name_suffix,
    "exact_normalized_address": _key_from_address,
    "exact_name_and_address": _key_from_combined,
}


def _row_series(row: object) -> pd.Series:
    if isinstance(row, pd.Series):
        return row
    mapping = row._asdict() if hasattr(row, "_asdict") else {}
    return pd.Series(mapping)


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
    max_candidates_per_s1: int | None = 200,
) -> tuple[CandidateMap, dict[tuple[str, str], dict[str, object]]]:
    """Return S1 -> candidate S2/S3 IDs plus pair metadata.

    Deterministic: candidate lists are de-duplicated and sorted.
    Does not form a Cartesian product.
    """
    methods = tuple(methods or BASELINE_METHODS)
    unknown = [m for m in methods if m not in METHOD_KEY_FNS]
    if unknown:
        raise ValueError(
            f"Unknown blocking methods {unknown}. "
            f"Implemented: {list(METHOD_KEY_FNS)}. Future: {list(FUTURE_METHODS)}"
        )

    pool = pd.concat([source2, source3], ignore_index=True)
    indexes = {method: build_block_index(pool, METHOD_KEY_FNS[method]) for method in methods}

    candidates: dict[str, set[str]] = {}
    meta: dict[tuple[str, str], dict[str, object]] = {}

    for row in source1.itertuples(index=False):
        s1 = str(row.entity_id)
        series = _row_series(row)
        found: dict[str, list[str]] = {}
        for method in methods:
            key = METHOD_KEY_FNS[method](series)
            if not key:
                continue
            for cand in indexes[method].get(key, []):
                if cand == s1 or cand.startswith("S1-"):
                    continue
                found.setdefault(cand, []).append(method)
        ordered = sorted(found)
        if max_candidates_per_s1 is not None:
            ordered = ordered[: max(0, int(max_candidates_per_s1))]
        candidates[s1] = set(ordered)
        for rank, cand in enumerate(ordered):
            meta[(s1, cand)] = {
                "blocking_methods": tuple(found[cand]),
                "blocking_method": found[cand][0],
                "candidate_rank": rank,
                "candidate_source": cand[:2],
            }

    canonical = {s1: sorted(ids) for s1, ids in candidates.items()}
    # Ensure every S1 appears even with zero candidates.
    for row in source1.itertuples(index=False):
        canonical.setdefault(str(row.entity_id), [])
    return canonical, meta


def evaluate_candidate_recall(
    candidates: Mapping[str, Iterable[str]],
    ground_truth: Mapping[str, Iterable[str]],
    *,
    n_s2: int | None = None,
    n_s3: int | None = None,
) -> dict[str, float]:
    """Evaluate blocking quality before matching."""
    recalls: list[float] = []
    sizes: list[int] = []
    n_s1 = len(ground_truth)
    for s1, true_ids in ground_truth.items():
        true_set = set(true_ids)
        cand_set = set(candidates.get(s1, []))
        sizes.append(len(cand_set))
        if not true_set:
            continue
        recalls.append(len(true_set & cand_set) / len(true_set))

    sizes_arr = np.array(sizes, dtype=float) if sizes else np.array([0.0])
    search_space = None
    if n_s2 is not None and n_s3 is not None and n_s1:
        search_space = float(n_s1) * float(n_s2 + n_s3)
    generated = float(sum(len(set(v)) for v in candidates.values()))
    reduction = (1.0 - generated / search_space) if search_space and search_space > 0 else float("nan")

    return {
        "candidate_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "average_candidates_per_s1": float(np.mean(sizes_arr)),
        "median_candidates_per_s1": float(np.median(sizes_arr)),
        "max_candidates_per_s1": float(np.max(sizes_arr)),
        "reduction_ratio": float(reduction),
        "n_s1_with_positives": float(len(recalls)),
        "n_s1": float(n_s1),
    }
