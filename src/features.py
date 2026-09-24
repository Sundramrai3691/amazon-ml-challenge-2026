"""Pair-level features. Feature set is config-driven and extensible."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.normalize import (
    address_tokens,
    numeric_address_tokens,
    normalize_address,
    normalize_name,
    strip_legal_suffixes,
)

BASELINE_FEATURE_NAMES = (
    "name_exact_normalized",
    "name_exact_no_suffix",
    "name_len_s1",
    "name_len_cand",
    "name_len_ratio",
    "name_token_overlap",
    "addr_exact_normalized",
    "addr_token_overlap",
    "addr_numeric_overlap",
    "addr_len_s1",
    "addr_len_cand",
    "addr_len_ratio",
    "country_exact_match",
    "country_both_present",
    "cand_is_s2",
    "cand_is_s3",
    "candidate_rank",
    "blocked_by_name",
    "blocked_by_address",
    "blocked_by_combined",
)

# Extension points (not computed in baseline_v1):
FUTURE_FEATURES = (
    "rapidfuzz_ratio",
    "levenshtein",
    "jaro",
    "jaccard",
    "char_tfidf_cosine",
    "word_tfidf_cosine",
)


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _len_ratio(a: int, b: int) -> float:
    if a == 0 and b == 0:
        return 1.0
    denom = max(a, b)
    return min(a, b) / denom if denom else 0.0


def _token_overlap(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pair_feature_dict(
    s1: Mapping[str, object],
    cand: Mapping[str, object],
    meta: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Compute baseline features for one (S1, candidate) pair."""
    meta = meta or {}
    s1_name = normalize_name(s1.get("business_name", ""))
    cand_name = normalize_name(cand.get("business_name", ""))
    s1_name_ns = strip_legal_suffixes(s1.get("business_name", ""))
    cand_name_ns = strip_legal_suffixes(cand.get("business_name", ""))
    s1_name_tokens = tuple(s1_name_ns.split()) if s1_name_ns else ()
    cand_name_tokens = tuple(cand_name_ns.split()) if cand_name_ns else ()

    s1_addr = normalize_address(s1.get("business_address", ""))
    cand_addr = normalize_address(cand.get("business_address", ""))
    s1_addr_tokens = address_tokens(s1.get("business_address", ""))
    cand_addr_tokens = address_tokens(cand.get("business_address", ""))
    s1_nums = numeric_address_tokens(s1.get("business_address", ""))
    cand_nums = numeric_address_tokens(cand.get("business_address", ""))

    s1_country = _text(s1.get("country", "")).strip()
    cand_country = _text(cand.get("country", "")).strip()
    both_countries = bool(s1_country) and bool(cand_country)

    cand_id = _text(cand.get("entity_id", ""))
    methods = meta.get("blocking_methods") or ()
    if isinstance(methods, str):
        methods = (methods,)

    name_len_s1 = len(s1_name)
    name_len_cand = len(cand_name)
    addr_len_s1 = len(s1_addr)
    addr_len_cand = len(cand_addr)

    rank = meta.get("candidate_rank")
    rank_value = float(rank) if rank is not None else -1.0

    return {
        "name_exact_normalized": float(s1_name == cand_name and bool(s1_name)),
        "name_exact_no_suffix": float(s1_name_ns == cand_name_ns and bool(s1_name_ns)),
        "name_len_s1": float(name_len_s1),
        "name_len_cand": float(name_len_cand),
        "name_len_ratio": _len_ratio(name_len_s1, name_len_cand),
        "name_token_overlap": _token_overlap(s1_name_tokens, cand_name_tokens),
        "addr_exact_normalized": float(s1_addr == cand_addr and bool(s1_addr)),
        "addr_token_overlap": _token_overlap(s1_addr_tokens, cand_addr_tokens),
        "addr_numeric_overlap": _token_overlap(s1_nums, cand_nums),
        "addr_len_s1": float(addr_len_s1),
        "addr_len_cand": float(addr_len_cand),
        "addr_len_ratio": _len_ratio(addr_len_s1, addr_len_cand),
        "country_exact_match": float(both_countries and s1_country.casefold() == cand_country.casefold()),
        "country_both_present": float(both_countries),
        "cand_is_s2": float(cand_id.startswith("S2-")),
        "cand_is_s3": float(cand_id.startswith("S3-")),
        "candidate_rank": rank_value,
        "blocked_by_name": float(
            "exact_normalized_name" in methods or "exact_name_without_legal_suffix" in methods
        ),
        "blocked_by_address": float("exact_normalized_address" in methods),
        "blocked_by_combined": float("exact_name_and_address" in methods),
    }


def records_by_id(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Compact id -> record mapping (one copy of source rows)."""
    records: dict[str, dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        records[str(row.entity_id)] = {
            "entity_id": str(row.entity_id),
            "business_name": str(getattr(row, "business_name", "")),
            "business_address": str(getattr(row, "business_address", "")),
            "country": str(getattr(row, "country", "")),
        }
    return records


def feature_matrix(
    pairs: Iterable[tuple[str, str]],
    s1_records: Mapping[str, Mapping[str, object]],
    cand_records: Mapping[str, Mapping[str, object]],
    pair_meta: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
    *,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    names = list(feature_names or BASELINE_FEATURE_NAMES)
    pair_meta = pair_meta or {}
    rows: list[dict[str, float]] = []
    index: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for s1_id, cand_id in pairs:
        s1 = s1_records.get(s1_id)
        cand = cand_records.get(cand_id)
        if s1 is None or cand is None:
            missing.append((s1_id, cand_id))
            continue
        feats = pair_feature_dict(s1, cand, pair_meta.get((s1_id, cand_id)))
        rows.append({name: float(feats.get(name, 0.0)) for name in names})
        index.append((s1_id, cand_id))
    if missing:
        raise KeyError(f"Missing records for {len(missing)} pairs, e.g. {missing[:3]}")
    frame = pd.DataFrame(rows, columns=names)
    if index:
        frame.index = pd.MultiIndex.from_tuples(index, names=["source1_entity_id", "candidate_entity_id"])
    return frame


def to_numpy(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=np.float64, copy=False)
