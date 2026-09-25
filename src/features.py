"""Pair-level features. Rarity statistics are fit on training records only."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import ratio as fuzz_ratio

from src.normalize import (
    address_tokens,
    compact_alnum,
    name_tokens,
    numeric_address_tokens,
    normalize_address,
    normalize_name,
    strip_legal_suffixes,
)

FEATURE_NAMES = (
    "name_exact_normalized",
    "name_exact_no_suffix",
    "name_exact_compact",
    "name_len_s1",
    "name_len_cand",
    "name_len_ratio",
    "name_token_count_diff",
    "name_token_jaccard",
    "name_token_containment",
    "name_char_similarity",
    "name_edit_similarity",
    "addr_exact_normalized",
    "addr_token_jaccard",
    "addr_token_containment",
    "addr_char_similarity",
    "addr_digit_overlap",
    "addr_shared_numeric_tokens",
    "addr_numeric_containment",
    "addr_len_s1",
    "addr_len_cand",
    "addr_len_ratio",
    "country_exact_match",
    "country_both_present",
    "cand_is_s2",
    "cand_is_s3",
    "candidate_rank",
    "tfidf_rank",
    "retrieved_by_exact_name",
    "retrieved_by_exact_address",
    "retrieved_by_char_name",
    "retrieved_by_char_address",
    "retrieved_by_char_combined",
    "n_retrieval_channels",
    "s1_name_missing",
    "cand_name_missing",
    "s1_addr_missing",
    "cand_addr_missing",
    "shared_rare_name_tokens",
    "name_rarity_weighted_overlap",
    "shared_rare_addr_tokens",
    "addr_rarity_weighted_overlap",
    "shared_rare_digits",
)

# Keep historical alias used by train/save_model.
BASELINE_FEATURE_NAMES = FEATURE_NAMES


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _is_missing(value: object) -> bool:
    return not _text(value).strip()


def _len_ratio(a: int, b: int) -> float:
    if a == 0 and b == 0:
        return 1.0
    denom = max(a, b)
    return min(a, b) / denom if denom else 0.0


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _containment(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa:
        return 1.0 if not sb else 0.0
    return len(sa & sb) / len(sa)


def _char_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return float(fuzz_ratio(a, b)) / 100.0


def _edit_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return float(JaroWinkler.similarity(a, b))


@dataclass
class RarityStats:
    """Document frequencies fitted on training-side records only."""

    n_docs: int = 1
    name_df: dict[str, int] = field(default_factory=dict)
    core_name_df: dict[str, int] = field(default_factory=dict)
    token_df: dict[str, int] = field(default_factory=dict)
    digit_df: dict[str, int] = field(default_factory=dict)

    def idf(self, table: Mapping[str, int], key: str) -> float:
        df = table.get(key, 0)
        return math.log((self.n_docs + 1.0) / (df + 1.0))


def fit_rarity_stats(frames: Sequence[pd.DataFrame]) -> RarityStats:
    """Fit name/token/digit frequencies on the provided frames (train only)."""
    name_df: Counter[str] = Counter()
    core_df: Counter[str] = Counter()
    token_df: Counter[str] = Counter()
    digit_df: Counter[str] = Counter()
    n_docs = 0
    for frame in frames:
        for row in frame.itertuples(index=False):
            n_docs += 1
            name = normalize_name(getattr(row, "business_name", ""))
            core = strip_legal_suffixes(getattr(row, "business_name", ""))
            if name:
                name_df[name] += 1
            if core:
                core_df[core] += 1
            tokens = set(name_tokens(getattr(row, "business_name", ""))) | set(
                address_tokens(getattr(row, "business_address", ""))
            )
            token_df.update(tokens)
            digit_df.update(set(numeric_address_tokens(getattr(row, "business_address", ""))))
    return RarityStats(
        n_docs=max(n_docs, 1),
        name_df=dict(name_df),
        core_name_df=dict(core_df),
        token_df=dict(token_df),
        digit_df=dict(digit_df),
    )


def pair_feature_dict(
    s1: Mapping[str, object],
    cand: Mapping[str, object],
    meta: Mapping[str, object] | None = None,
    rarity: RarityStats | None = None,
) -> dict[str, float]:
    meta = meta or {}
    s1_name_raw = _text(s1.get("business_name", ""))
    cand_name_raw = _text(cand.get("business_name", ""))
    s1_addr_raw = _text(s1.get("business_address", ""))
    cand_addr_raw = _text(cand.get("business_address", ""))

    s1_name = normalize_name(s1_name_raw)
    cand_name = normalize_name(cand_name_raw)
    s1_core = strip_legal_suffixes(s1_name_raw)
    cand_core = strip_legal_suffixes(cand_name_raw)
    s1_compact = compact_alnum(s1_name_raw)
    cand_compact = compact_alnum(cand_name_raw)
    s1_name_tokens = name_tokens(s1_name_raw)
    cand_name_tokens = name_tokens(cand_name_raw)

    s1_addr = normalize_address(s1_addr_raw)
    cand_addr = normalize_address(cand_addr_raw)
    s1_addr_tokens = address_tokens(s1_addr_raw)
    cand_addr_tokens = address_tokens(cand_addr_raw)
    s1_nums = numeric_address_tokens(s1_addr_raw)
    cand_nums = numeric_address_tokens(cand_addr_raw)

    s1_country = _text(s1.get("country", "")).strip()
    cand_country = _text(cand.get("country", "")).strip()
    both_countries = bool(s1_country) and bool(cand_country)
    cand_id = _text(cand.get("entity_id", ""))

    rank = meta.get("candidate_rank")
    tfidf_rank = meta.get("tfidf_rank")
    rarity = rarity or RarityStats()

    shared_name = set(s1_name_tokens) & set(cand_name_tokens)
    shared_addr = set(s1_addr_tokens) & set(cand_addr_tokens)
    shared_nums = set(s1_nums) & set(cand_nums)
    rare_name = sum(1 for tok in shared_name if rarity.idf(rarity.token_df, tok) >= 4.0)
    rare_addr = sum(1 for tok in shared_addr if rarity.idf(rarity.token_df, tok) >= 4.0)
    rare_digits = sum(1 for tok in shared_nums if rarity.idf(rarity.digit_df, tok) >= 4.0)
    name_weighted = sum(rarity.idf(rarity.token_df, tok) for tok in shared_name)
    addr_weighted = sum(rarity.idf(rarity.token_df, tok) for tok in shared_addr)

    def _flag(name: str) -> float:
        return float(int(meta.get(name, 0) or 0))

    return {
        "name_exact_normalized": float(s1_name == cand_name and bool(s1_name)),
        "name_exact_no_suffix": float(s1_core == cand_core and bool(s1_core)),
        "name_exact_compact": float(s1_compact == cand_compact and bool(s1_compact)),
        "name_len_s1": float(len(s1_name)),
        "name_len_cand": float(len(cand_name)),
        "name_len_ratio": _len_ratio(len(s1_name), len(cand_name)),
        "name_token_count_diff": float(abs(len(s1_name_tokens) - len(cand_name_tokens))),
        "name_token_jaccard": _jaccard(s1_name_tokens, cand_name_tokens),
        "name_token_containment": _containment(s1_name_tokens, cand_name_tokens),
        "name_char_similarity": _char_sim(s1_name, cand_name),
        "name_edit_similarity": _edit_sim(s1_name, cand_name),
        "addr_exact_normalized": float(s1_addr == cand_addr and bool(s1_addr)),
        "addr_token_jaccard": _jaccard(s1_addr_tokens, cand_addr_tokens),
        "addr_token_containment": _containment(s1_addr_tokens, cand_addr_tokens),
        "addr_char_similarity": _char_sim(s1_addr, cand_addr),
        "addr_digit_overlap": _jaccard(s1_nums, cand_nums),
        "addr_shared_numeric_tokens": float(len(shared_nums)),
        "addr_numeric_containment": _containment(s1_nums, cand_nums),
        "addr_len_s1": float(len(s1_addr)),
        "addr_len_cand": float(len(cand_addr)),
        "addr_len_ratio": _len_ratio(len(s1_addr), len(cand_addr)),
        "country_exact_match": float(both_countries and s1_country.casefold() == cand_country.casefold()),
        "country_both_present": float(both_countries),
        "cand_is_s2": float(cand_id.startswith("S2-")),
        "cand_is_s3": float(cand_id.startswith("S3-")),
        "candidate_rank": float(rank) if rank is not None else -1.0,
        "tfidf_rank": float(tfidf_rank) if tfidf_rank is not None else -1.0,
        "retrieved_by_exact_name": _flag("retrieved_by_exact_name"),
        "retrieved_by_exact_address": _flag("retrieved_by_exact_address"),
        "retrieved_by_char_name": _flag("retrieved_by_char_name"),
        "retrieved_by_char_address": _flag("retrieved_by_char_address"),
        "retrieved_by_char_combined": _flag("retrieved_by_char_combined"),
        "n_retrieval_channels": float(meta.get("n_retrieval_channels") or 0),
        "s1_name_missing": float(_is_missing(s1_name_raw)),
        "cand_name_missing": float(_is_missing(cand_name_raw)),
        "s1_addr_missing": float(_is_missing(s1_addr_raw)),
        "cand_addr_missing": float(_is_missing(cand_addr_raw)),
        "shared_rare_name_tokens": float(rare_name),
        "name_rarity_weighted_overlap": float(name_weighted),
        "shared_rare_addr_tokens": float(rare_addr),
        "addr_rarity_weighted_overlap": float(addr_weighted),
        "shared_rare_digits": float(rare_digits),
    }


def records_by_id(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
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
    rarity: RarityStats | None = None,
) -> pd.DataFrame:
    names = list(feature_names or FEATURE_NAMES)
    pair_meta = pair_meta or {}
    rows: list[list[float]] = []
    index: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for s1_id, cand_id in pairs:
        s1 = s1_records.get(s1_id)
        cand = cand_records.get(cand_id)
        if s1 is None or cand is None:
            missing.append((s1_id, cand_id))
            continue
        feats = pair_feature_dict(s1, cand, pair_meta.get((s1_id, cand_id)), rarity=rarity)
        rows.append([float(feats.get(name, 0.0)) for name in names])
        index.append((s1_id, cand_id))
    if missing:
        raise KeyError(f"Missing records for {len(missing)} pairs, e.g. {missing[:3]}")
    frame = pd.DataFrame(rows, columns=names)
    if index:
        frame.index = pd.MultiIndex.from_tuples(index, names=["source1_entity_id", "candidate_entity_id"])
    return frame


def to_numpy(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=np.float64, copy=False)
