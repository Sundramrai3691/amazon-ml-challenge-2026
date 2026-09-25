"""Ground-truth parsing tests. Synthetic data only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.load_data import GroundTruthError, ground_truth_to_sets, load_ground_truth, parse_matched_ids


def test_empty_matches_become_empty_set(tmp_path: Path) -> None:
    path = tmp_path / "gt.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1, S3-2\nS1-3\t\n",
        encoding="utf-8",
    )
    mapping = ground_truth_to_sets(load_ground_truth(path))
    assert mapping["S1-1"] == {"S2-1", "S3-2"}
    assert mapping["S1-3"] == set()
    assert set(mapping) == {"S1-1", "S1-3"}


def test_rejects_malformed_ids() -> None:
    with pytest.raises(GroundTruthError):
        parse_matched_ids("not-an-id")
    with pytest.raises(GroundTruthError):
        parse_matched_ids("S1-99")
    frame = pd.DataFrame(
        {"source1_entity_id": ["S1-1", "S1-1"], "matched_entity_ids": ["S2-1", "S3-1"]}
    )
    with pytest.raises(GroundTruthError):
        ground_truth_to_sets(frame)
