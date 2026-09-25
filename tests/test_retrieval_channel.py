"""Unit tests for retrieval_channel_contribution diagnostic."""

from __future__ import annotations

from src.error_analysis import retrieval_channel_contribution


def test_retrieval_channel_contribution_counts_flags_correctly() -> None:
    gt = {"S1-1": ["S2-10", "S2-20", "S3-30", "S3-40"]}
    candidates = {"S1-1": ["S2-10", "S2-20", "S3-30"]}  # S3-40 missed by both
    pair_meta = {
        ("S1-1", "S2-10"): {"retrieved_by_exact_name": 1, "retrieved_by_char_name": 1},  # both exact & tfidf
        ("S1-1", "S2-20"): {"retrieved_by_exact_address": 1},                          # exact only
        ("S1-1", "S3-30"): {"retrieved_by_char_combined": 1},                          # tfidf only (combined)
    }
    out = retrieval_channel_contribution(
        ground_truth=gt, candidates=candidates, pair_meta=pair_meta, label="test"
    )
    assert out["label"] == "test"
    assert out["totals"] == {
        "total_gt_pairs": 4,
        "recovered_either": 3,
        "recovered_by_exact_union": 2,
        "recovered_by_tfidf_union": 2,
        "recovered_by_both_exact_and_tfidf": 1,
        "missed_by_both": 1,
    }
    assert out["tp_recovery_breakdown"] == {
        "exact_only": 1,
        "tfidf_only": 1,
        "both_exact_and_tfidf": 1,
        "unknown_retrieval_path": 0,
    }
    assert out["tfidf_channel_breakdown"] == {
        "char_name_tp": 1,
        "char_address_tp": 0,
        "char_combined_tp": 1,
    }
    assert len(out["missed_by_both_examples"]) == 1
    assert out["missed_by_both_examples"][0]["id"] == "S3-40"


def test_retrieval_channel_contribution_no_pair_meta_falls_back_to_unknown() -> None:
    gt = {"S1-a": ["S2-x"]}
    cands = {"S1-a": ["S2-x"]}
    out = retrieval_channel_contribution(ground_truth=gt, candidates=cands, pair_meta=None)
    # recovered_either = 1 (in candidates), but no meta -> unknown_retrieval_path=1
    assert out["totals"]["total_gt_pairs"] == 1
    assert out["totals"]["recovered_either"] == 1
    assert out["tp_recovery_breakdown"]["unknown_retrieval_path"] == 1
