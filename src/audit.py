"""Streaming / chunked data audit. Does not modify raw files."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.load_data import (
    GROUND_TRUTH_COLUMNS,
    MATCH_ID_RE,
    SOURCE_COLUMNS,
    TSV_SEP,
    TRAIN_FILENAMES,
    TEST_FILENAMES,
    ground_truth_to_sets,
    load_ground_truth,
)
from src.normalize import normalize_address, normalize_name
from src.split import multiplicity_bucket, s2_s3_composition


def count_data_rows(path: str | Path) -> int:
    """Count TSV body rows without loading the file into a DataFrame."""
    path = Path(path)
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            if line.strip():
                n += 1
    return n


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def audit_source_file(path: str | Path, *, chunksize: int = 200_000) -> dict[str, Any]:
    path = Path(path)
    missing = Counter()
    countries = Counter()
    ids: set[str] = set()
    dup_ids = 0
    name_keys: Counter[str] = Counter()
    addr_keys: Counter[str] = Counter()
    n_rows = 0
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    reader = pd.read_csv(
        path,
        sep=TSV_SEP,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
    )
    for chunk in reader:
        if not columns:
            columns = list(chunk.columns)
            dtypes = {c: "str" for c in columns}
        n_rows += len(chunk)
        for col in ("business_name", "business_address", "country"):
            if col in chunk.columns:
                missing[col] += int(chunk[col].map(_is_missing).sum())
        if "country" in chunk.columns:
            countries.update(chunk["country"].fillna("").replace("", "MISSING").tolist())
        if "entity_id" in chunk.columns:
            for entity_id in chunk["entity_id"].astype(str):
                if entity_id in ids:
                    dup_ids += 1
                else:
                    ids.add(entity_id)
        if "business_name" in chunk.columns:
            name_keys.update(normalize_name(v) for v in chunk["business_name"] if not _is_missing(v))
        if "business_address" in chunk.columns:
            addr_keys.update(normalize_address(v) for v in chunk["business_address"] if not _is_missing(v))
    dup_names = sum(1 for key, n in name_keys.items() if key and n > 1)
    dup_addrs = sum(1 for key, n in addr_keys.items() if key and n > 1)
    return {
        "path": str(path),
        "n_rows": n_rows,
        "columns": columns,
        "dtypes": dtypes,
        "missing": dict(missing),
        "country_distribution": dict(countries.most_common()),
        "duplicate_raw_ids": dup_ids,
        "n_unique_ids": len(ids),
        "duplicate_normalized_names": dup_names,
        "duplicate_normalized_addresses": dup_addrs,
        "schema_ok": all(col in columns for col in SOURCE_COLUMNS),
    }


def audit_ground_truth(path: str | Path) -> dict[str, Any]:
    frame = load_ground_truth(path)
    gt = ground_truth_to_sets(frame)
    match_counts = Counter(len(v) for v in gt.values())
    buckets = Counter(multiplicity_bucket(n) for n in (len(v) for v in gt.values()))
    n = len(gt) or 1
    n_singleton = buckets.get("0", 0)
    s2_owners: dict[str, set[str]] = defaultdict(set)
    s3_owners: dict[str, set[str]] = defaultdict(set)
    both = 0
    for s1, ids in gt.items():
        if s2_s3_composition(ids) == "both":
            both += 1
        for mid in ids:
            if mid.startswith("S2-"):
                s2_owners[mid].add(s1)
            elif mid.startswith("S3-"):
                s3_owners[mid].add(s1)
            elif not MATCH_ID_RE.match(mid):
                pass
    s2_owner_n = Counter(len(v) for v in s2_owners.values())
    s3_owner_n = Counter(len(v) for v in s3_owners.values())
    n_s2_multi = sum(c for n_own, c in s2_owner_n.items() if n_own > 1)
    n_s3_multi = sum(c for n_own, c in s3_owner_n.items() if n_own > 1)
    return {
        "path": str(path),
        "n_rows": len(frame),
        "n_s1": len(gt),
        "columns": list(GROUND_TRUTH_COLUMNS),
        "match_count_raw": dict(sorted(match_counts.items())),
        "match_count_distribution": {
            "0": int(buckets.get("0", 0)),
            "1": int(buckets.get("1", 0)),
            "2": int(buckets.get("2", 0)),
            "3+": int(buckets.get("3+", 0)),
        },
        "singleton_count": n_singleton,
        "singleton_percentage": 100.0 * n_singleton / n,
        "s1_with_both_s2_and_s3": both,
        "s2_ownership": {
            "n_distinct_s2_in_gt": len(s2_owners),
            "max_s1_owners": max(s2_owner_n) if s2_owner_n else 0,
            "owner_count_distribution": dict(sorted(s2_owner_n.items())),
            "n_s2_with_multiple_s1": n_s2_multi,
        },
        "s3_ownership": {
            "n_distinct_s3_in_gt": len(s3_owners),
            "max_s1_owners": max(s3_owner_n) if s3_owner_n else 0,
            "owner_count_distribution": dict(sorted(s3_owner_n.items())),
            "n_s3_with_multiple_s1": n_s3_multi,
        },
        "note": "Ownership is measured only. No uniqueness constraint is imposed.",
    }


def run_full_audit(
    train_dir: str | Path,
    test_dir: str | Path | None = None,
    *,
    include_test_s2_s3: bool = False,
    chunksize: int = 200_000,
) -> dict[str, Any]:
    """Audit train fully. Test S2/S3 full-column diagnostics are optional (large)."""
    train_dir = Path(train_dir)
    report: dict[str, Any] = {"train": {}, "test": {}}
    report["train"]["source1"] = audit_source_file(train_dir / TRAIN_FILENAMES["source1"], chunksize=chunksize)
    report["train"]["source2"] = audit_source_file(train_dir / TRAIN_FILENAMES["source2"], chunksize=chunksize)
    report["train"]["source3"] = audit_source_file(train_dir / TRAIN_FILENAMES["source3"], chunksize=chunksize)
    report["train"]["ground_truth"] = audit_ground_truth(train_dir / TRAIN_FILENAMES["ground_truth"])
    if test_dir is not None:
        test_dir = Path(test_dir)
        report["test"]["source1"] = audit_source_file(test_dir / TEST_FILENAMES["source1"], chunksize=chunksize)
        report["test"]["source2_rowcount"] = count_data_rows(test_dir / TEST_FILENAMES["source2"])
        report["test"]["source3_rowcount"] = count_data_rows(test_dir / TEST_FILENAMES["source3"])
        if include_test_s2_s3:
            report["test"]["source2"] = audit_source_file(test_dir / TEST_FILENAMES["source2"], chunksize=chunksize)
            report["test"]["source3"] = audit_source_file(test_dir / TEST_FILENAMES["source3"], chunksize=chunksize)
        else:
            report["test"]["source2_s3_note"] = (
                "Full missingness/country diagnostics for test S2/S3 skipped by default "
                "(~1GB). Row counts only. Pass include_test_s2_s3=True for a full scan."
            )
    return report


def render_audit_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Data audit",
        "",
        "Measured locally from challenge TSV files. Values are not invented.",
        "Country is treated as an open-set string. No ownership constraint is applied.",
        "",
    ]

    def _src_block(title: str, stats: Mapping[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if "n_rows" in stats:
            lines.append(f"- rows: **{stats['n_rows']}**")
        if "columns" in stats:
            lines.append(f"- columns: `{stats['columns']}`")
        if "dtypes" in stats:
            lines.append(f"- dtypes: `{stats['dtypes']}`")
        if "missing" in stats:
            lines.append(f"- missingness: `{stats['missing']}`")
        if "country_distribution" in stats:
            lines.append(f"- country distribution: `{stats['country_distribution']}`")
        if "duplicate_raw_ids" in stats:
            lines.append(f"- duplicate raw IDs: **{stats['duplicate_raw_ids']}** (unique={stats.get('n_unique_ids')})")
        if "duplicate_normalized_names" in stats:
            lines.append(f"- duplicate normalized names: **{stats['duplicate_normalized_names']}**")
        if "duplicate_normalized_addresses" in stats:
            lines.append(f"- duplicate normalized addresses: **{stats['duplicate_normalized_addresses']}**")
        lines.append("")

    train = report.get("train", {})
    for key in ("source1", "source2", "source3"):
        if key in train:
            _src_block(f"Train {key}", train[key])
    if "ground_truth" in train:
        gt = train["ground_truth"]
        lines.extend(
            [
                "## Train ground truth",
                "",
                f"- rows / S1 entities: **{gt.get('n_s1')}**",
                f"- match-count distribution: `{gt.get('match_count_distribution')}`",
                f"- singleton percentage: **{gt.get('singleton_percentage'):.4f}%**" if gt.get("singleton_percentage") is not None else "- singleton percentage: not measured",
                f"- S1 with both S2 and S3 matches: **{gt.get('s1_with_both_s2_and_s3')}**",
                f"- S2 ownership: `{gt.get('s2_ownership')}`",
                f"- S3 ownership: `{gt.get('s3_ownership')}`",
                f"- note: {gt.get('note')}",
                "",
            ]
        )
    test = report.get("test", {})
    if "source1" in test:
        _src_block("Test source1", test["source1"])
    if "source2_rowcount" in test:
        lines.append("## Test source2 / source3 (row counts)")
        lines.append("")
        lines.append(f"- test_source2 rows: **{test['source2_rowcount']}**")
        lines.append(f"- test_source3 rows: **{test['source3_rowcount']}**")
        if "source2_s3_note" in test:
            lines.append(f"- {test['source2_s3_note']}")
        lines.append("")
    if "source2" in test:
        _src_block("Test source2 (full)", test["source2"])
    if "source3" in test:
        _src_block("Test source3 (full)", test["source3"])
    return "\n".join(lines) + "\n"
