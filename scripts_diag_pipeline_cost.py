"""DIAGNOSTIC ONLY: trace E3 memory/time cost without doing full work.

Runs NO training, only the first few heavy steps and reports dimensions.
Output is printed to stdout as JSON for machine readability.
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

repo = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(repo))

from sklearn.feature_extraction.text import TfidfVectorizer

from src.blocking import (
    _field_texts,
    _sparse_topk,
    build_block_index,
    char_tfidf_candidates,
    generate_exact_candidates,
    METHOD_KEY_FNS,
    EXACT_METHODS,
)
from src.load_data import load_training_data, ground_truth_to_sets, load_source1, load_source2, load_source3
from src.split import stratified_source1_split, entity_stratum


def load_cfg():
    with open(repo / "configs" / "baseline.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fmt_bytes(n: int) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < step:
            return f"{n:.2f} {unit}"
        n /= step
    return f"{n:.2f} PB"


def main() -> None:
    cfg = load_cfg()
    train_dir = repo / cfg["paths"]["train_dir"]
    seed = int(cfg.get("seed", 42))

    report: dict[str, object] = {"notes": "DIAGNOSTIC ONLY. NO model training / threshold sweep."}
    t0 = time.perf_counter()

    # ---- 1. LOAD ----
    t = time.perf_counter()
    tracemalloc.start()
    s1 = load_source1(train_dir / "train_source1.tsv")
    s2 = load_source2(train_dir / "train_source2.tsv")
    s3 = load_source3(train_dir / "train_source3.tsv")
    gt_frame = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    _, load_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dt_load = time.perf_counter() - t

    report["load"] = {
        "seconds": round(dt_load, 2),
        "peak_tracemalloc_bytes": load_peak,
        "peak_tracemalloc_human": fmt_bytes(load_peak),
        "s1_rows": int(len(s1)),
        "s2_rows": int(len(s2)),
        "s3_rows": int(len(s3)),
        "gt_rows": int(len(gt_frame)),
    }

    # ---- 2. LIMIT S1 (simulate 20k smoke test) ----
    LIMIT_S1 = 20000
    s1_ids_full = [str(x) for x in s1["entity_id"].tolist()]
    s1_ids = s1_ids_full[:LIMIT_S1]
    s1_small = s1[s1["entity_id"].isin(s1_ids)].copy()
    gt = ground_truth_to_sets(gt_frame)
    gt_small = {k: gt.get(k, set()) for k in s1_ids}

    country_by_id = {str(r.entity_id): r.country for r in s1_small.itertuples(index=False)}
    strata = {sid: entity_stratum(country_by_id.get(sid, ""), gt_small.get(sid, set())) for sid in s1_ids}
    train_ids_small, val_ids_small, split_sum = stratified_source1_split(
        s1_ids, strata, seed=seed, validation_fraction=float(cfg.get("validation_fraction", 0.2)),
    )
    report["sampled_s1"] = {
        "limit": LIMIT_S1,
        "train_s1": len(train_ids_small),
        "val_s1": len(val_ids_small),
        "split_summary": split_sum,
    }
    val_s1 = s1_small[s1_small["entity_id"].isin(val_ids_small)].copy()
    train_s1 = s1_small[s1_small["entity_id"].isin(train_ids_small)].copy()

    blocking_cfg = cfg.get("blocking", {}) or {}
    exact_methods = list(blocking_cfg.get("methods") or EXACT_METHODS)
    tfidf_cfg = blocking_cfg.get("char_tfidf", {}) or {}
    ngram_range = tuple(tfidf_cfg.get("ngram_range", [3, 5]))
    min_df = int(tfidf_cfg.get("min_df", 2))
    max_features = int(tfidf_cfg.get("max_features", 30000))
    report["config_snapshot"] = {
        "exact_methods": exact_methods,
        "ngram_range": list(ngram_range),
        "min_df": min_df,
        "max_features": max_features,
    }

    # ---- 3. EXACT BLOCKING (VAL subset only) ----
    t = time.perf_counter()
    tracemalloc.start()
    exact_val, _ = generate_exact_candidates(
        val_s1, s2, s3, methods=exact_methods, max_candidates_per_s1=None,
    )
    _, exact_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report["exact_val_blocking"] = {
        "seconds": round(time.perf_counter() - t, 2),
        "peak_tracemalloc_human": fmt_bytes(exact_peak),
        "n_s1_with_candidates": int(sum(1 for v in exact_val.values() if v)),
        "total_candidate_links": int(sum(len(v) for v in exact_val.values())),
    }

    # ---- 4. DIAGNOSE char_tfidf: S2 only, name only, fit S2 only ----
    # The full pipeline calls generate_candidate_pool which does:
    #   for pool in (S2, S3):
    #     for field in (name, address, combined):
    #         fit vectorizer on pool[field]
    #         transform queries (val S1)
    #         sparse matmul top-k
    # That's 2 pools * 3 fields = 6 vectorizer fits + 6 sparse matmul queries.
    # S1 query count is small (val subset ~4k in 20k limit case) but POOL size is FULL.
    report["char_tfidf"] = {}
    per_pool_field = []
    total_seconds = 0.0
    for pool_name, pool in (("s2", s2), ("s3", s3)):
        for field in ("name", "address", "combined"):
            entry = {"pool": pool_name, "field": field, "pool_rows": int(len(pool))}
            # --- FIT only (no transform) ---
            t = time.perf_counter()
            tracemalloc.start()
            try:
                vec = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    lowercase=False,
                )
                pool_texts = _field_texts(pool, field)
                mat_fit = vec.fit_transform(pool_texts)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                dt_fit = time.perf_counter() - t
                n_vocab = int(len(vec.vocabulary_))
                entry["fit_seconds"] = round(dt_fit, 2)
                entry["vocab_size"] = n_vocab
                entry["fit_matrix_shape"] = list(mat_fit.shape)
                entry["fit_matrix_nnz"] = int(mat_fit.nnz)
                entry["fit_matrix_bytes_estimate"] = int(
                    mat_fit.data.nbytes + mat_fit.indices.nbytes + mat_fit.indptr.nbytes
                )
                entry["fit_peak_tracemalloc_human"] = fmt_bytes(peak)

                # --- TRANSFORM queries (val S1) + top-k matmul (K=10) ---
                t = time.perf_counter()
                tracemalloc.start()
                q_texts = _field_texts(val_s1, field)
                q_mat = vec.transform(q_texts)
                index_ids = [str(x) for x in pool["entity_id"].tolist()]
                hits = _sparse_topk(q_mat, mat_fit, index_ids, k=10, batch_size=256)
                _, peak2 = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                dt_q = time.perf_counter() - t
                entry["query_seconds"] = round(dt_q, 2)  # includes transform + sparse topk
                entry["query_shape"] = list(q_mat.shape)
                entry["query_nnz"] = int(q_mat.nnz)
                entry["query_peak_tracemalloc_human"] = fmt_bytes(peak2)
                entry["total_candidate_links_returned"] = int(sum(len(h) for h in hits))
                total_seconds += dt_fit + dt_q
                entry["status"] = "ok"
                del vec, mat_fit, q_mat, hits
            except MemoryError as e:
                entry["status"] = "oom"
                entry["error"] = str(e)
            per_pool_field.append(entry)
    report["char_tfidf"]["per_pool_field"] = per_pool_field
    report["char_tfidf"]["estimated_total_seconds_val_only_K10"] = round(total_seconds, 2)

    # ---- 5. SUMMARIZE ROOT CAUSE ----
    report["bottleneck_summary"] = {
        "why_limit_s1_does_not_help_fit": (
            "TF-IDF is fit on the FULL S2 (~5.03M) and S3 (~5.29M) corpora regardless of S1 limit. "
            "Val S1 limit only reduces query rows; fit/matmul cost is dominated by pool size."
        ),
        "channels_per_s1_scope": 6,  # (name,addr,combined) x (s2,s3)
        "note_k_grid_repeated_work": (
            "Baseline pipeline fits TF-IDF ONCE per scope at max(K_grid) then uses "
            "union_exact_with_tfidf_k to slice ranks for smaller K; no repeated fit. "
            "However every scope still builds two sparse matrices of size pool x max_features, "
            "whose memory/CPU is invariant to S1 limit size."
        ),
        "total_runtime_seconds_so_far": round(time.perf_counter() - t0, 2),
    }

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
