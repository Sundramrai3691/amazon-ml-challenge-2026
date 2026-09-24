import pandas as pd

from src.blocking import (
    char_tfidf_candidates,
    evaluate_candidate_recall,
    generate_candidate_pool,
    generate_candidates,
)
from src.features import fit_rarity_stats, pair_feature_dict
from src.predict import apply_threshold


def _frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def test_tfidf_blocking_tiny() -> None:
    s1 = _frame([("S1-1", "Acme Incorporated", "100 Main Street", "US")])
    s2 = _frame(
        [
            ("S2-1", "Acme Inc", "100 Main St", "US"),
            ("S2-9", "Zeta Foods", "8 Oak Avenue", "US"),
        ]
    )
    s3 = _frame([("S3-1", "Unrelated Co", "2 Pine Road", "India")])
    cmap, meta, _runtime = char_tfidf_candidates(
        s1, s2, field="name", k=2, min_df=1, max_features=200, method_name="char_tfidf_name"
    )
    assert cmap["S1-1"]
    assert all(x.startswith("S2-") for x in cmap["S1-1"])
    assert ( "S1-1", cmap["S1-1"][0] ) in meta
    pool, pool_meta, _ = generate_candidate_pool(
        s1, s2, s3, tfidf_k=2, use_char_tfidf=True, min_df=1, max_features=200, max_candidates_per_s1=20
    )
    assert "S2-1" in pool["S1-1"]
    flags = pool_meta[("S1-1", "S2-1")]
    assert flags["n_retrieval_channels"] >= 1
    metrics = evaluate_candidate_recall(pool, {"S1-1": {"S2-1"}}, n_s2=2, n_s3=1)
    assert metrics["candidate_recall"] == 1.0
    assert "p95_candidates_per_s1" in metrics


def test_pair_features_and_rarity() -> None:
    s1 = {"entity_id": "S1-1", "business_name": "Acme Inc", "business_address": "1 Main St", "country": "US"}
    cand = {"entity_id": "S2-1", "business_name": "Acme", "business_address": "1 Main Street", "country": "US"}
    frames = [
        _frame([("S1-9", "Acme Inc", "1 Main St", "US")]),
        _frame([("S2-1", "Acme", "1 Main Street", "US")]),
        _frame([("S3-1", "Other", "9 Oak", "India")]),
    ]
    rarity = fit_rarity_stats(frames)
    feats = pair_feature_dict(
        s1,
        cand,
        {
            "retrieved_by_exact_name": 1,
            "retrieved_by_char_name": 1,
            "n_retrieval_channels": 2,
            "candidate_rank": 0,
            "tfidf_rank": 1,
        },
        rarity=rarity,
    )
    assert feats["name_exact_no_suffix"] == 1.0
    assert feats["country_exact_match"] == 1.0
    assert feats["cand_is_s2"] == 1.0
    assert feats["retrieved_by_char_name"] == 1.0
    assert 0.0 <= feats["name_char_similarity"] <= 1.0
    assert feats["name_rarity_weighted_overlap"] >= 0.0


def test_threshold_allows_empty_and_many() -> None:
    pairs = [("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-9")]
    scores = [0.9, 0.8, 0.1]
    preds = apply_threshold(pairs, scores, threshold=0.7, s1_ids=["S1-1", "S1-2", "S1-3"])
    assert preds["S1-1"] == ["S2-1", "S3-1"]
    assert preds["S1-2"] == []
    assert preds["S1-3"] == []
    none = apply_threshold(pairs, scores, threshold=0.99, s1_ids=["S1-1"])
    assert none["S1-1"] == []
