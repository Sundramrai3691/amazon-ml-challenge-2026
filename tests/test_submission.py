from pathlib import Path

import pandas as pd
import pytest

from src.submission import SubmissionError, write_submission
from utils.validate_submission import main as validate_main


def test_submission_headers_and_rules(tmp_path: Path) -> None:
    s1_ids = ["S1-1", "S1-2", "S1-3"]
    candidates = {
        "S1-1": ["S2-1", "S3-1", "S2-1"],
        "S1-2": [],
        "S1-3": ["S2-9"],
    }
    matches = {
        "S1-1": ["S3-1"],
        "S1-2": [],
        "S1-3": ["S2-9"],
    }
    matching_path, candidate_path = write_submission(s1_ids, candidates, matches, tmp_path)

    matching = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    candidate = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    assert list(matching.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(candidate.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert matching_path.read_text(encoding="utf-8").splitlines()[0] == (
        "source1_entity_id\tmatched_entity_ids"
    )
    assert len(matching) == 3
    assert set(matching["source1_entity_id"]) == set(s1_ids)
    empty = matching.loc[matching["source1_entity_id"] == "S1-2", "matched_entity_ids"].iloc[0]
    assert empty == ""
    s1_1_cands = candidate.loc[candidate["source1_entity_id"] == "S1-1", "candidate_entity_ids"].iloc[0]
    assert s1_1_cands == "S2-1,S3-1"
    for cell in list(matching["matched_entity_ids"]) + list(candidate["candidate_entity_ids"]):
        ids = [p for p in cell.split(",") if p]
        assert len(ids) == len(set(ids))
        for entity_id in ids:
            assert entity_id.startswith("S2-") or entity_id.startswith("S3-")


def test_matches_must_be_subset(tmp_path: Path) -> None:
    with pytest.raises(SubmissionError):
        write_submission(
            ["S1-1"],
            {"S1-1": ["S2-1"]},
            {"S1-1": ["S2-99"]},
            tmp_path,
        )


def test_rejects_s1_in_lists(tmp_path: Path) -> None:
    with pytest.raises(SubmissionError):
        write_submission(["S1-1"], {"S1-1": ["S1-1"]}, {"S1-1": []}, tmp_path)


def test_official_validator_fast_mode(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tA\t1 St\tUS\n"
        "S1-2\tB\t2 St\tFrance\n",
        encoding="utf-8",
    )
    (test_dir / "test_source2.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\tA\t1 St\tUS\n",
        encoding="utf-8",
    )
    (test_dir / "test_source3.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-1\tA\t1 St\tUS\n",
        encoding="utf-8",
    )
    matching_path, candidate_path = write_submission(
        ["S1-1", "S1-2"],
        {"S1-1": ["S2-1", "S3-1"], "S1-2": []},
        {"S1-1": ["S2-1"], "S1-2": []},
        tmp_path / "output",
    )
    code = validate_main(
        [
            "--matching",
            str(matching_path),
            "--candidate",
            str(candidate_path),
            "--test-dir",
            str(test_dir),
        ]
    )
    assert code == 0
    code_ids = validate_main(
        [
            "--matching",
            str(matching_path),
            "--candidate",
            str(candidate_path),
            "--test-dir",
            str(test_dir),
            "--check-ids",
        ]
    )
    assert code_ids == 0
