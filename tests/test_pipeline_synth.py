from pathlib import Path

import pandas as pd

from src.blocking import generate_candidate_pool
from src.features import feature_matrix, fit_rarity_stats, records_by_id
from src.predict import apply_threshold, score_pairs
from src.submission import write_submission
from src.train import build_pair_labels, fit_pair_model, make_model


def test_synthetic_e2e_pipeline(tmp_path: Path) -> None:
    s1 = pd.DataFrame(
        [
            ("S1-1", "Acme Inc", "10 Main Street", "US"),
            ("S1-2", "Solo Shop", "1 Oak", "India"),
            ("S1-3", "Beta LLC", "5 Rue", "France"),
        ],
        columns=["entity_id", "business_name", "business_address", "country"],
    )
    s2 = pd.DataFrame(
        [
            ("S2-1", "Acme", "10 Main St", "US"),
            ("S2-9", "Noise Co", "99 Nowhere", "US"),
        ],
        columns=["entity_id", "business_name", "business_address", "country"],
    )
    s3 = pd.DataFrame(
        [
            ("S3-1", "Acme Incorporated", "10 Main Street", "US"),
            ("S3-8", "Other", "2 Pine", "India"),
        ],
        columns=["entity_id", "business_name", "business_address", "country"],
    )
    gt = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set(), "S1-3": set()}
    cands, meta, _ = generate_candidate_pool(
        s1, s2, s3, tfidf_k=3, min_df=1, max_features=300, max_candidates_per_s1=20
    )
    rarity = fit_rarity_stats([s1.iloc[:2], s2, s3])
    pairs, y, stats = build_pair_labels(cands, gt)
    assert stats.positives >= 1
    assert stats.singleton_entities_represented >= 1
    s1_rec = records_by_id(s1)
    cand_rec = {**records_by_id(s2), **records_by_id(s3)}
    X = feature_matrix(pairs, s1_rec, cand_rec, meta, rarity=rarity)
    model = make_model("gbdt", n_estimators=20, random_state=42, verbose=-1)
    fit_pair_model(model, X.to_numpy(dtype=float), y)
    scores = score_pairs(model, X.to_numpy(dtype=float))
    preds = apply_threshold(pairs, scores, threshold=0.3, s1_ids=list(s1["entity_id"]))
    matching, candidate = write_submission(list(s1["entity_id"]), cands, preds, tmp_path)
    text = matching.read_text(encoding="utf-8")
    assert text.startswith("source1_entity_id\tmatched_entity_ids")
    assert "S1-2" in candidate.read_text(encoding="utf-8")
    for s1_id, pred_ids in preds.items():
        assert set(pred_ids) <= set(cands[s1_id])
