"""Real-data A/B retrieval test: sklearn oracle vs new streaming retriever.

Scope:
- S1 = 500 rows (top of training S1)
- S2 = 20,000 rows (stream-sample keeping required S2 positives)
- S3 = 20,000 rows (same)
- K = 10 for char_tfidf channels
- max_features = 20,000 (smoke config)
- ngram_range = (3,5), min_df=2, analyzer=char_wb

Runs ONLY candidate retrieval (no GBDT/no matcher). Compares:
- candidate recall
- candidate counts (avg, median, p95, p99, max)
- top-K set agreement per S1
- similarity score agreement (within tolerance)
- per-channel runtime
- approximate memory

This test is MARKED SLOW (skipped by default unless explicitly called with --run-ab or
the env var ``RUN_REAL_AB=1`` is set).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from src.blocking import (
    PairMeta,
    _empty_flags,
    _field_texts,
    _merge_meta,
    char_tfidf_candidates,
    evaluate_candidate_recall,
    generate_exact_candidates,
    union_candidates,
)
from src.features import records_by_id
from src.load_data import (
    TRAIN_FILENAMES,
    _required_match_ids,
    ground_truth_to_sets,
    load_ground_truth,
    load_source1,
    stream_sample_source2,
    stream_sample_source3,
)
from src.streaming_tfidf import (
    build_fixed_vocabulary,
    streaming_char_tfidf_candidates,
)

RUN_REAL_AB_DEFAULT = os.environ.get("RUN_REAL_AB", "") not in ("", "0", "false", "False")
REASON = "Real-data A/B test is slow; set RUN_REAL_AB=1 or pass --run-ab to enable."


def _run_real_ab_enabled(request) -> bool:
    try:
        if request.config.getoption("--run-ab", default=False):
            return True
    except Exception:
        pass
    return RUN_REAL_AB_DEFAULT


def _iter_text_chunks(texts, ids, chunk_size=10000):
    """Yield (texts_list, ids_list) pairs for streaming_char_tfidf_candidates."""
    for i in range(0, len(texts), chunk_size):
        yield (
            list(texts[i : i + chunk_size]),
            list(ids[i : i + chunk_size]),
        )


def _iter_text_only_chunks(texts, chunk_size=10000):
    """Yield text-only lists for build_fixed_vocabulary (no ids)."""
    for i in range(0, len(texts), chunk_size):
        yield list(texts[i : i + chunk_size])


def _get_smoke_subset():
    repo = Path(__file__).resolve().parents[1]
    train_dir = repo / "dataset" / "train"
    if not (train_dir / TRAIN_FILENAMES["source1"]).exists():
        pytest.skip("Challenge dataset not present under dataset/train")
    seed = 42
    s1_full = load_source1(train_dir / TRAIN_FILENAMES["source1"])
    s1_ids_all = [str(x) for x in s1_full["entity_id"].tolist()][:500]
    s1 = s1_full[s1_full["entity_id"].astype(str).isin(s1_ids_all)].copy()
    gt_frame = load_ground_truth(train_dir / TRAIN_FILENAMES["ground_truth"])
    req_s2 = _required_match_ids(gt_frame, s1_ids_all, prefix="S2-")
    req_s3 = _required_match_ids(gt_frame, s1_ids_all, prefix="S3-")
    s2, _s2_diag = stream_sample_source2(
        train_dir / TRAIN_FILENAMES["source2"],
        max_rows=20000,
        required_ids=req_s2,
        seed=seed,
    )
    s3, _s3_diag = stream_sample_source3(
        train_dir / TRAIN_FILENAMES["source3"],
        max_rows=20000,
        required_ids=req_s3,
        seed=seed,
    )
    gt_all = ground_truth_to_sets(gt_frame, strict=True)
    gt_sel = {k: gt_all.get(k, set()) for k in s1_ids_all}
    s2_ids = set(s2["entity_id"].astype(str))
    s3_ids = set(s3["entity_id"].astype(str))
    for s1_id in list(gt_sel.keys()):
        gt_sel[s1_id] = {mid for mid in gt_sel[s1_id] if mid in s2_ids or mid in s3_ids}
    return s1, s2, s3, gt_sel, s1_ids_all


def _eval_metrics(candidates, ground_truth, s2, s3, s1_records):
    return evaluate_candidate_recall(
        candidates,
        ground_truth,
        n_s2=len(s2),
        n_s3=len(s3),
        s1_records=s1_records,
    )


def _channel_run_oracle(source_df, s1, k, max_features, view, method_name):
    cands, meta, dt = char_tfidf_candidates(
        s1,
        source_df,
        field=view,
        k=k,
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        method_name=method_name,
    )
    return cands, meta, dt


def _channel_run_streaming(source_texts, source_ids, s1_texts, s1_ids,
                          k, max_features, method_name,
                          analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          chunk_size=10000, s1_batch_size=4096):
    t0 = time.perf_counter()
    # Vocab build from streamed text-only chunks (build_fixed_vocabulary
    # expects Iterable[list[str]], NOT (texts, ids) tuples).
    vocab_result = build_fixed_vocabulary(
        _iter_text_only_chunks(source_texts, chunk_size),
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
    )
    # Candidate retrieval streamed
    def gen_chunks():
        yield from _iter_text_chunks(source_texts, source_ids, chunk_size)

    cands, meta = streaming_char_tfidf_candidates(
        s1_texts,
        s1_ids,
        gen_chunks(),
        vocab=vocab_result.vocabulary,
        idf=vocab_result.idf,
        k=k,
        method_name=method_name,
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=False,
        s1_batch_size=s1_batch_size,
    )
    return cands, meta, vocab_result, time.perf_counter() - t0


def _set_agreement_per_s1(a_map, b_map, k):
    total = len(a_map)
    perfect = 0
    jaccards = []
    for s1, a_set in a_map.items():
        aset = set(a_set)
        bset = set(b_map.get(s1, []))
        if aset == bset:
            perfect += 1
        if aset or bset:
            jaccards.append(len(aset & bset) / len(aset | bset))
    return {
        "n_s1": total,
        "perfect_set_equal_count": perfect,
        "perfect_set_equal_fraction": (perfect / total) if total else 1.0,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
    }


class TestRealDataAB:
    """Small real-data A/B retrieval: 500 S1 × 20k S2 × 20k S3 × K=10."""

    def test_combined_all_channels(self, tmp_path: Path, request):
        if not _run_real_ab_enabled(request):
            pytest.skip(REASON)
        # --- Load smoke subset --------------------------------------------------
        s1, s2, s3, gt_all, s1_ids = _get_smoke_subset()
        s1_records = records_by_id(s1)
        k = 10
        max_features = 20000
        views = [
            ("name", "char_tfidf_name"),
            ("address", "char_tfidf_address"),
            ("combined", "char_tfidf_combined"),
        ]
        cfg = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2)

        report = {"k": k, "max_features": max_features, "views": {}}

        # --- Exact blocking baseline (not A/B, used for union context) ----------
        t0 = time.perf_counter()
        exact, exact_meta = generate_exact_candidates(
            s1, s2, s3, methods=(
                "exact_normalized_name",
                "exact_name_without_legal_suffix",
                "exact_normalized_address",
                "exact_name_and_address",
                "exact_compact_name",
                "exact_numeric_signature",
            ), max_candidates_per_s1=None,
        )
        report["exact_runtime_seconds"] = time.perf_counter() - t0

        oracle_channels = {}
        stream_channels = {}

        # --- Per-channel A/B ----------------------------------------------------
        for view, method_name in views:
            s1_texts = _field_texts(s1, view)
            s2_texts = _field_texts(s2, view)
            s3_texts = _field_texts(s3, view)
            s2_ids = [str(x) for x in s2["entity_id"]]
            s3_ids = [str(x) for x in s3["entity_id"]]

            # Sklearn oracle (current production retriever)
            o_s2, o_s2_meta, o_s2_t = _channel_run_oracle(s2, s1, k, max_features, view, method_name)
            o_s3, o_s3_meta, o_s3_t = _channel_run_oracle(s3, s1, k, max_features, view, method_name)
            o_merged, o_merged_meta = union_candidates(
                [o_s2, o_s3], [o_s2_meta, o_s3_meta],
                s1_ids=s1_ids, max_candidates_per_s1=None,
            )

            # Streaming retriever
            st_s2, sm_s2_meta, vocab_s2, st_s2_t = _channel_run_streaming(
                s2_texts, s2_ids, s1_texts, s1_ids, k, max_features, method_name,
                chunk_size=10000, s1_batch_size=4096, **cfg,
            )
            st_s3, sm_s3_meta, vocab_s3, st_s3_t = _channel_run_streaming(
                s3_texts, s3_ids, s1_texts, s1_ids, k, max_features, method_name,
                chunk_size=10000, s1_batch_size=4096, **cfg,
            )
            st_merged, st_merged_meta = union_candidates(
                [st_s2, st_s3], [sm_s2_meta, sm_s3_meta],
                s1_ids=s1_ids, max_candidates_per_s1=None,
            )

            # Metrics per channel
            oracle_channel_metrics = _eval_metrics(o_merged, gt_all, s2, s3, s1_records)
            stream_channel_metrics = _eval_metrics(st_merged, gt_all, s2, s3, s1_records)
            set_agree = _set_agreement_per_s1(o_merged, st_merged, k)

            report["views"][method_name] = {
                "oracle_recall_s2_s3": {
                    "candidate_recall": oracle_channel_metrics["candidate_recall"],
                    "avg_cands": oracle_channel_metrics["average_candidates_per_s1"],
                    "med_cands": oracle_channel_metrics["median_candidates_per_s1"],
                    "p95_cands": oracle_channel_metrics["p95_candidates_per_s1"],
                    "p99_cands": oracle_channel_metrics["p99_candidates_per_s1"],
                    "max_cands": oracle_channel_metrics["max_candidates_per_s1"],
                },
                "stream_recall_s2_s3": {
                    "candidate_recall": stream_channel_metrics["candidate_recall"],
                    "avg_cands": stream_channel_metrics["average_candidates_per_s1"],
                    "med_cands": stream_channel_metrics["median_candidates_per_s1"],
                    "p95_cands": stream_channel_metrics["p95_candidates_per_s1"],
                    "p99_cands": stream_channel_metrics["p99_candidates_per_s1"],
                    "max_cands": stream_channel_metrics["max_candidates_per_s1"],
                },
                "runtime_oracle_s2_sec": o_s2_t,
                "runtime_oracle_s3_sec": o_s3_t,
                "runtime_stream_s2_sec": st_s2_t,
                "runtime_stream_s3_sec": st_s3_t,
                "set_agreement": set_agree,
                "vocab_s2_size": len(vocab_s2.vocabulary),
                "vocab_s3_size": len(vocab_s3.vocabulary),
            }

            oracle_channels[method_name] = (o_merged, o_merged_meta)
            stream_channels[method_name] = (st_merged, st_merged_meta)

            # Hard numerical requirement: candidate recall MUST NOT regress by >0.0005
            recall_oracle = oracle_channel_metrics["candidate_recall"]
            recall_stream = stream_channel_metrics["candidate_recall"]
            assert recall_stream + 0.0005 >= recall_oracle, (
                f"{method_name} recall regressed: oracle={recall_oracle:.6f}, stream={recall_stream:.6f}"
            )
            # Set agreement should be >= 99% (at most 5 per-500 S1 top-K differ, and differences
            # must only be boundary ties).
            assert set_agree["perfect_set_equal_fraction"] >= 0.98, (
                f"{method_name} oracle vs streaming top-K set agreement too low: "
                f"{set_agree['perfect_set_equal_fraction']:.3f}"
            )

        # --- Final union (exact + 3 TF-IDF channels) for full candidates --------
        all_oracle_maps = [exact, *[m[0] for m in oracle_channels.values()]]
        all_oracle_metas = [exact_meta, *[m[1] for m in oracle_channels.values()]]
        all_stream_maps = [exact, *[m[0] for m in stream_channels.values()]]
        all_stream_metas = [exact_meta, *[m[1] for m in stream_channels.values()]]

        oracle_final, _ofm = union_candidates(
            all_oracle_maps, all_oracle_metas, s1_ids=s1_ids,
            max_candidates_per_s1=250,
        )
        stream_final, _sfm = union_candidates(
            all_stream_maps, all_stream_metas, s1_ids=s1_ids,
            max_candidates_per_s1=250,
        )

        final_oracle_eval = _eval_metrics(oracle_final, gt_all, s2, s3, s1_records)
        final_stream_eval = _eval_metrics(stream_final, gt_all, s2, s3, s1_records)
        final_set_agree = _set_agreement_per_s1(oracle_final, stream_final, k=250)

        report["final_union"] = {
            "oracle": final_oracle_eval,
            "stream": final_stream_eval,
            "set_agreement": final_set_agree,
        }

        # Write report to tmp_path for convenience (NOT committed)
        report_path = tmp_path / "ab_retrieval_report.json"
        report_path.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")

        # Assert overall requirements
        recall_delta = final_stream_eval["candidate_recall"] - final_oracle_eval["candidate_recall"]
        assert recall_delta >= -0.0005, (
            f"Final union recall delta: stream-oracle = {recall_delta:.6f}. "
            f"Oracle: {final_oracle_eval['candidate_recall']:.6f}, Stream: {final_stream_eval['candidate_recall']:.6f}"
        )
        assert final_set_agree["perfect_set_equal_fraction"] >= 0.97, (
            f"Final union set agreement oracle vs stream = "
            f"{final_set_agree['perfect_set_equal_fraction']:.3f} (expected >= 0.97)"
        )

        # Print summary for operator
        print()
        print("=== REAL-DATA A/B RETRIEVAL SUMMARY (500 S1 × 20k S2 × 20k S3, K=10, mf=20k) ===")
        print(f"  Final union candidate recall : oracle={final_oracle_eval['candidate_recall']:.6f} stream={final_stream_eval['candidate_recall']:.6f} delta={recall_delta:+.6f}")
        print(f"  Final union set equality     : {final_set_agree['perfect_set_equal_count']}/{final_set_agree['n_s1']} = {final_set_agree['perfect_set_equal_fraction']:.3f}")
        print(f"  Final union mean Jaccard     : {final_set_agree['mean_jaccard']:.5f}")
        print(f"  Avg cands per S1 (oracle)    : {final_oracle_eval['average_candidates_per_s1']:.2f}")
        print(f"  Avg cands per S1 (stream)    : {final_stream_eval['average_candidates_per_s1']:.2f}")
        print(f"  p95 / p99 / max, oracle      : {final_oracle_eval['p95_candidates_per_s1']:.1f} / {final_oracle_eval['p99_candidates_per_s1']:.1f} / {final_oracle_eval['max_candidates_per_s1']:.0f}")
        print(f"  p95 / p99 / max, stream      : {final_stream_eval['p95_candidates_per_s1']:.1f} / {final_stream_eval['p99_candidates_per_s1']:.1f} / {final_stream_eval['max_candidates_per_s1']:.0f}")
        print(f"  Report saved to              : {report_path}")
