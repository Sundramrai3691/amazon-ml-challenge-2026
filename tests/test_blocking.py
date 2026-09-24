import pandas as pd

from src.blocking import evaluate_candidate_recall, generate_candidates


def _frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def test_candidate_generation_no_duplicates_and_prefixes() -> None:
    s1 = _frame(
        [
            ("S1-1", "Acme Inc", "1 Main Street", "US"),
            ("S1-2", "No Match Co", "9 Other Rd", "India"),
        ]
    )
    s2 = _frame(
        [
            ("S2-1", "Acme", "1 Main St", "US"),
            ("S2-2", "Acme Inc", "99 Nowhere", "US"),
        ]
    )
    s3 = _frame(
        [
            ("S3-1", "Acme LLC", "1 Main Street", "France"),
            ("S3-2", "Unrelated", "9 Other Road", "India"),
        ]
    )
    candidates, meta = generate_candidates(s1, s2, s3, max_candidates_per_s1=50)
    assert set(candidates) == {"S1-1", "S1-2"}
    for s1_id, ids in candidates.items():
        assert ids == sorted(set(ids))
        for cand in ids:
            assert cand.startswith("S2-") or cand.startswith("S3-")
            assert not cand.startswith("S1-")
            assert (s1_id, cand) in meta
    assert "S2-1" in candidates["S1-1"]
    assert "S3-1" in candidates["S1-1"]


def test_candidate_recall_synthetic() -> None:
    s1 = _frame([("S1-1", "Acme Inc", "1 Main Street", "US")])
    s2 = _frame([("S2-1", "Acme", "1 Main St", "US"), ("S2-9", "Zeta", "8 Oak", "US")])
    s3 = _frame([("S3-1", "Other", "2 Pine", "India")])
    candidates, _ = generate_candidates(s1, s2, s3)
    gt = {"S1-1": {"S2-1"}}
    metrics = evaluate_candidate_recall(candidates, gt, n_s2=2, n_s3=1)
    assert metrics["candidate_recall"] == 1.0
    assert metrics["average_candidates_per_s1"] >= 1.0
    assert metrics["max_candidates_per_s1"] >= 1.0
    assert 0.0 <= metrics["reduction_ratio"] <= 1.0
