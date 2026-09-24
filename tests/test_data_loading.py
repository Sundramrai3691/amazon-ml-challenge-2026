"""Synthetic TSV loading tests. Do not use the competition dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.load_data import (
    DataSchemaError,
    TSV_SEP,
    load_ground_truth,
    load_source1,
    parse_matched_ids,
    split_source1_ids,
)


def _write_source(path: Path) -> None:
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tAcme Inc\t1 Main St\tUS\n"
        "S1-2\tBeta LLC\t2 Rue\tFrance\n",
        encoding="utf-8",
    )


def test_tsv_separator(tmp_path: Path) -> None:
    path = tmp_path / "train_source1.tsv"
    _write_source(path)
    frame = load_source1(path)
    assert list(frame.columns) == ["entity_id", "business_name", "business_address", "country"]
    assert frame.iloc[0]["entity_id"] == "S1-1"
    assert frame.iloc[1]["country"] == "France"
    raw = path.read_text(encoding="utf-8")
    assert TSV_SEP in raw
    # Open-set country: France is preserved, not filtered.
    assert set(frame["country"]) == {"US", "France"}


def test_schema_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("entity_id\tbusiness_name\nS1-1\tAcme\n", encoding="utf-8")
    with pytest.raises(DataSchemaError):
        load_source1(path)


def test_malformed_one_column_tsv(tmp_path: Path) -> None:
    path = tmp_path / "comma.csv"
    path.write_text(
        "entity_id,business_name,business_address,country\nS1-1,Acme,1 Main,US\n",
        encoding="utf-8",
    )
    with pytest.raises(DataSchemaError):
        load_source1(path)


def test_ground_truth_empty_matches(tmp_path: Path) -> None:
    path = tmp_path / "gt.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n",
        encoding="utf-8",
    )
    frame = load_ground_truth(path)
    assert parse_matched_ids("S2-1, S2-1,S3-9") == ["S2-1", "S3-9"]
    assert frame.iloc[0]["matched_entity_id_list"] == ["S2-1", "S3-1"]
    assert frame.iloc[1]["matched_entity_id_list"] == []


def test_split_is_reproducible() -> None:
    ids = [f"S1-{i}" for i in range(20)]
    a_train, a_val = split_source1_ids(ids, seed=42, validation_fraction=0.2)
    b_train, b_val = split_source1_ids(ids, seed=42, validation_fraction=0.2)
    assert a_train == b_train and a_val == b_val
    assert set(a_train).isdisjoint(a_val)
    assert len(a_train) + len(a_val) == 20
