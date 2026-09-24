"""Candidate generation (blocking). Matching is a separate stage."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.normalize import (
    compact_alnum,
    normalize_address,
    normalize_name,
    numeric_signature,
    strip_legal_suffixes,
)
from src.split import multiplicity_bucket

CandidateMap = dict[str, list[str]]
BlockIndex = dict[str, list[str]]
PairMeta = dict[tuple[str, str], dict[str, object]]

EXACT_METHODS = (
    "exact_normalized_name",
    "exact_name_without_legal_suffix",
    "exact_normalized_address",
    "exact_name_and_address",
    "exact_compact_name",
    "exact_numeric_signature",
)

BASELINE_METHODS = EXACT_METHODS

CHAR_TFIDF_METHODS = (
    "char_tfidf_name",
    "char_tfidf_address",
    "char_tfidf_combined",
)

METHOD_FLAG = {
    "exact_normalized_name": "retrieved_by_exact_name",
    "exact_name_without_legal_suffix": "retrieved_by_exact_name",
    "exact_compact_name": "retrieved_by_exact_name",
    "exact_normalized_address": "retrieved_by_exact_address",
    "exact_name_and_address": "retrieved_by_exact_address",
    "exact_numeric_signature": "retrieved_by_exact_address",
    "char_tfidf_name": "retrieved_by_char_name",
    "char_tfidf_address": "retrieved_by_char_address",
    "char_tfidf_combined": "retrieved_by_char_combined",
}

FUTURE_METHODS = ("word_tfidf", "postcode_token", "fuzzy", "source_aware")


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


def _compact_key(name: object) -> str:
    return compact_alnum(name)


def _numeric_key(address: object) -> str:
    return numeric_signature(address)


def _row_mapping(row: object) -> dict[str, object]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    return {}


def _key_from_name(row: Mapping[str, object]) -> str:
    return _name_key(row.get("business_name", ""))


def _key_from_name_suffix(row: Mapping[str, object]) -> str:
    return _suffix_free_key(row.get("business_name", ""))


def _key_from_address(row: Mapping[str, object]) -> str:
    return _address_key(row.get("business_address", ""))


def _key_from_combined(row: Mapping[str, object]) -> str:
    return _combined_key(row.get("business_name", ""), row.get("business_address", ""))


def _key_from_compact(row: Mapping[str, object]) -> str:
    return _compact_key(row.get("business_name", ""))


def _key_from_numeric(row: Mapping[str, object]) -> str:
    return _numeric_key(row.get("business_address", ""))


METHOD_KEY_FNS: dict[str, Callable[[Mapping[str, object]], str]] = {
    "exact_normalized_name": _key_from_name,
    "exact_name_without_legal_suffix": _key_from_name_suffix,
    "exact_normalized_address": _key_from_address,
    "exact_name_and_address": _key_from_combined,
    "exact_compact_name": _key_from_compact,
    "exact_numeric_signature": _key_from_numeric,
}


def build_block_index(
    frame: pd.DataFrame,
    key_fn: Callable[[Mapping[str, object]], str],
) -> BlockIndex:
    index: BlockIndex = defaultdict(list)
    for row in frame.itertuples(index=False):
        mapping = _row_mapping(row)
        entity_id = str(mapping.get("entity_id", getattr(row, "entity_id", "")))
        if not _valid_candidate_id(entity_id):
            continue
        key = key_fn(mapping)
        if not key:
            continue
        index[key].append(entity_id)
    return {key: list(dict.fromkeys(ids)) for key, ids in index.items()}


def _empty_flags() -> dict[str, object]:
    return {
        "retrieved_by_exact_name": 0,
        "retrieved_by_exact_address": 0,
        "retrieved_by_char_name": 0,
        "retrieved_by_char_address": 0,
        "retrieved_by_char_combined": 0,
        "blocking_methods": (),
        "blocking_method": "",
        "candidate_rank": -1,
        "tfidf_rank": -1,
        "n_retrieval_channels": 0,
        "candidate_source": "",
    }


def _merge_meta(meta: PairMeta, s1: str, cand: str, method: str, *, rank: int | None = None) -> None:
    key = (s1, cand)
    slot = meta.setdefault(key, _empty_flags())
    methods = list(slot.get("blocking_methods") or ())
    if method not in methods:
        methods.append(method)
    slot["blocking_methods"] = tuple(methods)
    slot["blocking_method"] = methods[0]
    flag = METHOD_FLAG.get(method)
    if flag:
        slot[flag] = 1
    channels = (
        int(slot["retrieved_by_exact_name"])
        + int(slot["retrieved_by_exact_address"])
        + int(slot["retrieved_by_char_name"])
        + int(slot["retrieved_by_char_address"])
        + int(slot["retrieved_by_char_combined"])
    )
    slot["n_retrieval_channels"] = channels
    if cand.startswith("S2-"):
        slot["candidate_source"] = "S2"
    elif cand.startswith("S3-"):
        slot["candidate_source"] = "S3"
    if rank is not None:
        if method.startswith("char_tfidf"):
            prev = int(slot.get("tfidf_rank", -1))
            slot["tfidf_rank"] = rank if prev < 0 else min(prev, rank)
        prev_rank = int(slot.get("candidate_rank", -1))
        slot["candidate_rank"] = rank if prev_rank < 0 else min(prev_rank, rank)


def generate_exact_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
    max_candidates_per_s1: int | None = None,
) -> tuple[CandidateMap, PairMeta]:
    """Exact-key blocking independently for S1→S2 and S1→S3, then union."""
    methods = tuple(methods or EXACT_METHODS)
    unknown = [m for m in methods if m not in METHOD_KEY_FNS]
    if unknown:
        raise ValueError(f"Unknown exact blocking methods {unknown}")

    indexes_s2 = {m: build_block_index(source2, METHOD_KEY_FNS[m]) for m in methods}
    indexes_s3 = {m: build_block_index(source3, METHOD_KEY_FNS[m]) for m in methods}

    found_by_s1: dict[str, dict[str, list[str]]] = {}
    meta: PairMeta = {}
    for row in source1.itertuples(index=False):
        mapping = _row_mapping(row)
        s1 = str(mapping.get("entity_id", ""))
        found: dict[str, list[str]] = {}
        for method in methods:
            key = METHOD_KEY_FNS[method](mapping)
            if not key:
                continue
            for cand in indexes_s2[method].get(key, []) + indexes_s3[method].get(key, []):
                if not _valid_candidate_id(cand) or cand == s1:
                    continue
                found.setdefault(cand, []).append(method)
        found_by_s1[s1] = found

    canonical: CandidateMap = {}
    for s1, found in found_by_s1.items():
        ordered = sorted(found)
        if max_candidates_per_s1 is not None:
            ordered = ordered[: max(0, int(max_candidates_per_s1))]
        canonical[s1] = ordered
        for rank, cand in enumerate(ordered):
            for method in found[cand]:
                _merge_meta(meta, s1, cand, method, rank=rank)
    for row in source1.itertuples(index=False):
        canonical.setdefault(str(row.entity_id), [])
    return canonical, meta


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
    max_candidates_per_s1: int | None = 200,
) -> tuple[CandidateMap, PairMeta]:
    """Backward-compatible exact blocking entry point."""
    return generate_exact_candidates(
        source1,
        source2,
        source3,
        methods=methods,
        max_candidates_per_s1=max_candidates_per_s1,
    )


def _field_texts(frame: pd.DataFrame, field: str) -> list[str]:
    if field == "name":
        return [normalize_name(v) for v in frame["business_name"].tolist()]
    if field == "address":
        return [normalize_address(v) for v in frame["business_address"].tolist()]
    names = [normalize_name(v) for v in frame["business_name"].tolist()]
    addrs = [normalize_address(v) for v in frame["business_address"].tolist()]
    return [f"{n} {a}".strip() for n, a in zip(names, addrs)]


def _sparse_topk(
    query_mat,
    index_mat,
    index_ids: Sequence[str],
    k: int,
    batch_size: int = 256,
) -> list[list[tuple[str, int, float]]]:
    """Return per-query list of (id, rank, score). Sparse matmul only."""
    out: list[list[tuple[str, int, float]]] = []
    index_t = index_mat.T.tocsr()
    n_q = query_mat.shape[0]
    for start in range(0, n_q, batch_size):
        block = query_mat[start : start + batch_size]
        sims = block @ index_t
        for row_i in range(sims.shape[0]):
            row = sims.getrow(row_i)
            if row.nnz == 0:
                out.append([])
                continue
            data = row.data
            indices = row.indices
            take_n = min(k, data.size)
            if data.size > k:
                chosen = np.argpartition(data, -k)[-k:]
            else:
                chosen = np.arange(data.size)
            order = chosen[np.argsort(data[chosen])[::-1]][:take_n]
            hits = []
            for rank, j in enumerate(order):
                entity_id = str(index_ids[int(indices[j])])
                if _valid_candidate_id(entity_id):
                    hits.append((entity_id, rank, float(data[j])))
            out.append(hits)
    return out


def char_tfidf_candidates(
    source1: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    field: str,
    k: int,
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_features: int = 50_000,
    method_name: str,
) -> tuple[CandidateMap, PairMeta, float]:
    started = time.perf_counter()
    index_ids = [str(x) for x in pool["entity_id"].tolist()]
    query_ids = [str(x) for x in source1["entity_id"].tolist()]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
    )
    index_mat = vectorizer.fit_transform(_field_texts(pool, field))
    query_mat = vectorizer.transform(_field_texts(source1, field))
    hits = _sparse_topk(query_mat, index_mat, index_ids, k=k)
    candidates: CandidateMap = {s1: [] for s1 in query_ids}
    meta: PairMeta = {}
    for s1, row_hits in zip(query_ids, hits):
        ordered = []
        seen: set[str] = set()
        for entity_id, rank, _score in row_hits:
            if entity_id == s1 or entity_id in seen:
                continue
            seen.add(entity_id)
            ordered.append(entity_id)
            _merge_meta(meta, s1, entity_id, method_name, rank=rank)
        candidates[s1] = ordered
    runtime = time.perf_counter() - started
    return candidates, meta, runtime


def union_candidates(
    maps: Sequence[CandidateMap],
    metas: Sequence[PairMeta],
    *,
    s1_ids: Sequence[str],
    max_candidates_per_s1: int | None = None,
) -> tuple[CandidateMap, PairMeta]:
    merged_meta: PairMeta = {}
    for meta in metas:
        for pair, payload in meta.items():
            slot = merged_meta.setdefault(pair, _empty_flags())
            methods = list(slot.get("blocking_methods") or ())
            for method in payload.get("blocking_methods") or ():
                if method not in methods:
                    methods.append(method)
                flag = METHOD_FLAG.get(str(method))
                if flag:
                    slot[flag] = 1
            slot["blocking_methods"] = tuple(methods)
            if methods:
                slot["blocking_method"] = methods[0]
            for key in (
                "retrieved_by_exact_name",
                "retrieved_by_exact_address",
                "retrieved_by_char_name",
                "retrieved_by_char_address",
                "retrieved_by_char_combined",
            ):
                if payload.get(key):
                    slot[key] = 1
            for rank_key in ("candidate_rank", "tfidf_rank"):
                incoming = int(payload.get(rank_key, -1))
                prev = int(slot.get(rank_key, -1))
                if incoming >= 0:
                    slot[rank_key] = incoming if prev < 0 else min(prev, incoming)
            if payload.get("candidate_source"):
                slot["candidate_source"] = payload["candidate_source"]
            slot["n_retrieval_channels"] = (
                int(slot["retrieved_by_exact_name"])
                + int(slot["retrieved_by_exact_address"])
                + int(slot["retrieved_by_char_name"])
                + int(slot["retrieved_by_char_address"])
                + int(slot["retrieved_by_char_combined"])
            )
    out: CandidateMap = {}
    for s1 in s1_ids:
        found: set[str] = set()
        for cmap in maps:
            found.update(cmap.get(s1, []))
        ordered = [cid for cid in found if _valid_candidate_id(cid)]
        if max_candidates_per_s1 is not None and len(ordered) > int(max_candidates_per_s1):
            exact = []
            rest = []
            for cand in ordered:
                slot = merged_meta.get((s1, cand), {})
                if slot.get("retrieved_by_exact_name") or slot.get("retrieved_by_exact_address"):
                    exact.append(cand)
                else:
                    rest.append(cand)
            rest.sort(key=lambda c: int(merged_meta.get((s1, c), {}).get("tfidf_rank", 10**9)))
            ordered = list(dict.fromkeys(exact + rest))[: int(max_candidates_per_s1)]
        ordered = sorted(ordered)
        out[s1] = ordered
        for rank, cand in enumerate(ordered):
            slot = merged_meta.setdefault((s1, cand), _empty_flags())
            prev = int(slot.get("candidate_rank", -1))
            slot["candidate_rank"] = rank if prev < 0 else min(prev, rank)
    return out, merged_meta


def generate_candidate_pool(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    exact_methods: Sequence[str] | None = None,
    tfidf_k: int = 50,
    use_char_tfidf: bool = True,
    max_candidates_per_s1: int | None = 250,
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_features: int = 50_000,
) -> tuple[CandidateMap, PairMeta, dict[str, float]]:
    """Union exact blocking with character TF-IDF retrieval (S1→S2 and S1→S3)."""
    s1_ids = [str(x) for x in source1["entity_id"].tolist()]
    exact, exact_meta = generate_exact_candidates(
        source1, source2, source3, methods=exact_methods, max_candidates_per_s1=None
    )
    maps = [exact]
    metas = [exact_meta]
    runtimes: dict[str, float] = {}
    if use_char_tfidf and tfidf_k > 0:
        field_methods = (
            ("name", "char_tfidf_name"),
            ("address", "char_tfidf_address"),
            ("combined", "char_tfidf_combined"),
        )
        for pool_name, pool in (("s2", source2), ("s3", source3)):
            for field, method_name in field_methods:
                cmap, meta, runtime = char_tfidf_candidates(
                    source1,
                    pool,
                    field=field,
                    k=tfidf_k,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    method_name=method_name,
                )
                maps.append(cmap)
                metas.append(meta)
                runtimes[f"{method_name}_{pool_name}"] = runtime
    merged, meta = union_candidates(
        maps, metas, s1_ids=s1_ids, max_candidates_per_s1=max_candidates_per_s1
    )
    return merged, meta, runtimes


def truncate_candidates(candidates: Mapping[str, Sequence[str]], k: int) -> CandidateMap:
    return {s1: list(ids)[:k] for s1, ids in candidates.items()}


def is_exact_hit(meta: Mapping[str, object] | None) -> bool:
    if not meta:
        return False
    return bool(meta.get("retrieved_by_exact_name") or meta.get("retrieved_by_exact_address"))


def union_exact_with_tfidf_k(
    exact: CandidateMap,
    union_map: Mapping[str, Sequence[str]],
    meta: PairMeta,
    k: int,
) -> CandidateMap:
    """Keep every exact hit; add TF-IDF hits with rank < k. Does not intersect."""
    out: CandidateMap = {}
    s1_ids = list(dict.fromkeys([*exact.keys(), *union_map.keys()]))
    for s1 in s1_ids:
        kept = set(exact.get(s1, []))
        if k > 0:
            for cand in union_map.get(s1, []):
                rank = int(meta.get((s1, cand), {}).get("tfidf_rank", -1))
                if 0 <= rank < k:
                    kept.add(cand)
        out[s1] = sorted(cid for cid in kept if _valid_candidate_id(cid))
    return out


def cap_preferring_exact(
    candidates: CandidateMap,
    meta: PairMeta,
    max_candidates_per_s1: int | None,
) -> CandidateMap:
    if max_candidates_per_s1 is None:
        return {s1: list(ids) for s1, ids in candidates.items()}
    capped: CandidateMap = {}
    limit = max(0, int(max_candidates_per_s1))
    for s1, ids in candidates.items():
        if len(ids) <= limit:
            capped[s1] = list(ids)
            continue
        exact_ids = [c for c in ids if is_exact_hit(meta.get((s1, c)))]
        rest = [c for c in ids if c not in set(exact_ids)]
        rest.sort(key=lambda c: (int(meta.get((s1, c), {}).get("tfidf_rank", 10**9)), c))
        capped[s1] = list(dict.fromkeys(exact_ids + rest))[:limit]
    return capped


def evaluate_candidate_recall(
    candidates: Mapping[str, Iterable[str]],
    ground_truth: Mapping[str, Iterable[str]],
    *,
    n_s2: int | None = None,
    n_s3: int | None = None,
    s1_records: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, float]:
    """Evaluate blocking quality before matching."""
    recalls: list[float] = []
    sizes: list[int] = []
    singleton_with_cands = 0
    singletons = 0
    n_s1 = len(ground_truth)
    by_country: dict[str, list[float]] = defaultdict(list)
    by_mult: dict[str, list[float]] = defaultdict(list)
    by_missing: dict[str, list[float]] = defaultdict(list)
    by_name_len: dict[str, list[float]] = defaultdict(list)

    for s1, true_ids in ground_truth.items():
        true_set = set(true_ids)
        cand_set = set(candidates.get(s1, []))
        sizes.append(len(cand_set))
        if not true_set:
            singletons += 1
            if cand_set:
                singleton_with_cands += 1
            continue
        rec = len(true_set & cand_set) / len(true_set)
        recalls.append(rec)
        rec_obj = s1_records.get(s1) if s1_records else None
        if rec_obj:
            country = str(rec_obj.get("country") or "MISSING")
            by_country[country].append(rec)
            name = str(rec_obj.get("business_name") or "")
            addr = str(rec_obj.get("business_address") or "")
            miss = "complete"
            if not name.strip() and not addr.strip():
                miss = "both_missing"
            elif not name.strip():
                miss = "name_missing"
            elif not addr.strip():
                miss = "address_missing"
            by_missing[miss].append(rec)
            by_name_len["short" if len(name) < 12 else "long"].append(rec)
        by_mult[multiplicity_bucket(len(true_set))].append(rec)

    sizes_arr = np.array(sizes, dtype=float) if sizes else np.array([0.0])
    search_space = None
    if n_s2 is not None and n_s3 is not None and n_s1:
        search_space = float(n_s1) * float(n_s2 + n_s3)
    generated = float(sum(len(set(v)) for v in candidates.values()))
    reduction = (1.0 - generated / search_space) if search_space and search_space > 0 else float("nan")

    def _mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    out: dict[str, float] = {
        "candidate_recall": _mean(recalls),
        "average_candidates_per_s1": float(np.mean(sizes_arr)),
        "median_candidates_per_s1": float(np.median(sizes_arr)),
        "p95_candidates_per_s1": float(np.quantile(sizes_arr, 0.95)),
        "p99_candidates_per_s1": float(np.quantile(sizes_arr, 0.99)),
        "max_candidates_per_s1": float(np.max(sizes_arr)),
        "reduction_ratio": float(reduction),
        "n_s1_with_positives": float(len(recalls)),
        "n_s1": float(n_s1),
        "singleton_candidate_rate": (singleton_with_cands / singletons) if singletons else float("nan"),
    }
    for country, vals in by_country.items():
        out[f"recall_country_{country}"] = _mean(vals)
    for bucket, vals in by_mult.items():
        out[f"recall_mult_{bucket}"] = _mean(vals)
    for key, vals in by_missing.items():
        out[f"recall_missing_{key}"] = _mean(vals)
    for key, vals in by_name_len.items():
        out[f"recall_name_{key}"] = _mean(vals)
    return out
