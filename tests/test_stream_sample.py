"""Unit tests for the memory-safe S2/S3 stream-sample loader.

All tests build synthetic TSV files. They never touch the real 5M-row datasets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.load_data import (
    SOURCE_COLUMNS,
    TSV_SEP,
    _required_match_ids,
    load_ground_truth,
    load_source2,
    load_source3,
    stream_sample_source2,
    stream_sample_source3,
)


def _write_source_tsv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    header = TSV_SEP.join(SOURCE_COLUMNS) + "\n"
    body = "".join(
        f"{eid}\t{name}\t{addr}\t{country}\n" for eid, name, addr, country in rows)
    path.write_text(header + body, encoding="utf-8")


def _write_gt_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        + "".join(f"{s1}\t{mids}\n" for s1, mids in rows),
        encoding="utf-8",
    )


def test_stream_sample_basic_deterministic(tmp_path: Path) -> None:
    rows = [(f"S2-{i}", f"Company{i}", f"{i} Main St", "US") for i in range(100)]
    p = tmp_path / "s2.tsv"
    _write_source_tsv(p, rows)
    a, diag_a = stream_sample_source2(p, max_rows=20, required_ids=None, seed=42, chunksize=7)
    b, diag_b = stream_sample_source2(p, max_rows=20, required_ids=None, seed=42, chunksize=7)
    assert list(a["entity_id"]) == list(b["entity_id"])
    assert diag_a.rows_requested == 20
    assert len(a) == 20
    assert len(b) == 20
    assert set(a["entity_id"]).isdisjoint(set()) or len(set(a["entity_id"])) == len(a["entity_id"])
    assert diag_a.via_sample == 20
    assert diag_a.via_positive_keep == 0


def test_stream_sample_keeps_required_ids(tmp_path: Path) -> None:
    rows = [(f"S2-{i}", f"Name{i}", f"{i} St", "US") for i in range(100)]
    p = tmp_path / "s2.tsv"
    _write_source_tsv(p, rows)
    required = {"S2-0", "S2-3", "S2-99"}
    frame, diag = stream_sample_source2(
        p, max_rows=15, required_ids=required, seed=42, chunksize=11
    )
    ids = set(frame["entity_id"])
    assert required.issubset(ids)
    assert diag.required_ids == required
    assert diag.required_ids_found == required
    assert diag.required_ids_missing == set()
    assert diag.via_positive_keep == 3
    assert diag.via_sample == 12
    # rows_loaded=3+12<=15 (cap)
    assert 14 <= diag.rows_loaded <= 15


def test_stream_sample_positives_exceed_cap(tmp_path: Path) -> None:
    rows = [(f"S2-{i}", f"Co{i}", f"A{i}", "IN") for i in range(20)]
    p = tmp_path / "s2.tsv"
    _write_source_tsv(p, rows)
    required = {f"S2-{i}" for i in range(10)}
    # Request max_rows=5 but 10 required IDs -> returns all 10 required (overflow)
    frame, diag = stream_sample_source3(
        p, max_rows=5, required_ids=required, seed=1, chunksize=6
    )
    assert set(frame["entity_id"]) >= required
    assert diag.rows_loaded >= 10
    assert diag.rows_requested == 5
    # No sample slots: reservoir_cap = 5-10 negative, so no sample fills
    assert diag.via_sample == 0
    assert diag.via_positive_keep == 10
    assert diag.required_ids_missing == set()


def test_stream_sample_reports_missing_required(tmp_path: Path) -> None:
    rows = [(f"S2-{i}", f"X{i}", f"Y{i}", "US") for i in range(10)]
    p = tmp_path / "s2.tsv"
    _write_source_tsv(p, rows)
    required = {"S2-1", "S2-999", "S2-5"}
    frame, diag = stream_sample_source2(
        p, max_rows=30, required_ids=required, seed=1, chunksize=4
    )
    assert diag.required_ids_missing == {"S2-999"}
    assert {"S2-1", "S2-5"}.issubset(set(frame["entity_id"]))
    assert diag.required_ids_found == {"S2-1", "S2-5"}


def test_stream_sample_no_duplicates(tmp_path: Path) -> None:
    rows = [
        ("S2-1", "Dup", "1", "US"),
        ("S2-2", "A", "2", "US"),
        ("S2-1", "Dup2", "1b", "US"),  # duplicate entity again -> skipped
        ("S2-3", "B", "3", "US"),
    ]
    p = tmp_path / "s2.tsv"
    _write_source_tsv(p, rows)
    frame, _diag = stream_sample_source2(
        p, max_rows=10, required_ids={"S2-1"}, seed=42, chunksize=2
    )
    assert list(frame["entity_id"]) == ["S2-1", "S2-2", "S2-3"]  # order: S2-1 appears once


def test_stream_sample_chunked_loader(tmp_path: Path) -> None:
    rows = [(f"S3-{i:05d}", f"Firm{i}", f"{i} Blvd", "FR") for i in range(200)]
    p = tmp_path / "s3.tsv"
    _write_source_tsv(p, rows)
    required = {"S3-00001", "S3-00199"}
    frame, diag = stream_sample_source3(
        p, max_rows=50, required_ids=required, seed=7, chunksize=13
    )
    assert len(frame) == 50
    assert required.issubset(set(frame["entity_id"]))
    assert diag.stream_chunks > 1  # chunksize=13, 200 rows -> ceil(200/13)~16 chunks)
    assert len(set(frame["entity_id"])) == len(frame["entity_id"])  # no duplicate IDs


def test_required_match_ids_filtering(tmp_path: Path) -> None:
    gt = tmp_path / "gt.tsv"
    _write_gt_tsv(
        gt,
        [
            ("S1-1", "S2-1,S2-77,S3-9"),
            ("S1-2", "S3-5"),
            ("S1-3", "S2-3,S3-1"),
            ("S1-4", ""),  # S1-4 not selected;
        ],
    )
    gt_frame = load_ground_truth(gt)
    s2_ids = _required_match_ids(gt_frame, ["S1-1", "S1-3"], prefix="S2-")
    s3_ids = _required_match_ids(gt_frame, ["S1-1", "S1-3"], prefix="S3-")
    assert s2_ids == {"S2-1", "S2-77", "S2-3"}
    assert s3_ids == {"S3-9", "S3-1"}


def test_stream_sample_full_loader_unchanged(tmp_path: Path) -> None:
    rows = [(f"S2-{i}", f"Co{i}", f"Addr{i}", "IN") for i in range(50)]
    p = tmp_path / "s2.tsv"
    _write_source_tsv(p, rows)
    full = load_source2(p)
    assert len(full) == 50
    assert list(full.columns) == list(SOURCE_COLUMNS)
    # Streaming sampler's sampled subset columns identical dtype=str
    sample, _diag = stream_sample_source2(p, max_rows=50, seed=42, chunksize=10)
    assert len(sample) == 50
    assert set(sample["entity_id"]) == set(full["entity_id"])
    assert list(sample.columns) == list(SOURCE_COLUMNS)


def test_stream_sample_s3(tmp_path: Path) -> None:
    rows = [(f"S3-{i}", f"X{i}", f"Y{i}", "US") for i in range(30)]
    p = tmp_path / "s3.tsv"
    _write_source_tsv(p, rows)
    required = {"S3-0", "S3-29"}
    frame, diag = stream_sample_source3(
        p, max_rows=10, required_ids=required, seed=0, chunksize=5
    )
    assert diag.required_ids_found == {"S3-0", "S3-29"}
    assert diag.required_ids_missing == set()
    assert diag.rows_loaded == 10
    assert required.issubset(set(frame["entity_id"]))
