#!/usr/bin/env python3
"""Stdlib submission validator for Business Entity Resolution.

This file was added at repository initialization because the official
organizer validator was not present. If Amazon provides
``utils/validate_submission.py``, replace this file with that official
copy and do not rewrite its logic.

Default (fast) mode does not load full S2/S3 ID sets.
Pass ``--check-ids`` only in a memory-aware environment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
SOURCE_HEADER = ("entity_id", "business_name", "business_address", "country")


class ValidationError(Exception):
    """Raised when a submission file fails a hard check."""


def parse_id_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = raw.strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _read_tsv_with_header(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ValidationError(f"File not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def load_source1_ids(test_dir: Path) -> list[str]:
    path = test_dir / "test_source1.tsv"
    fieldnames, rows = _read_tsv_with_header(path)
    missing = [col for col in SOURCE_HEADER if col not in fieldnames]
    if missing:
        raise ValidationError(f"{path} missing columns: {missing}")
    ids = [row["entity_id"].strip() for row in rows if row.get("entity_id")]
    if not ids:
        raise ValidationError(f"No Source 1 entity IDs in {path}")
    return ids


def load_s2_s3_ids(test_dir: Path) -> set[str]:
    """Memory-heavy on the full test set. Only used with --check-ids."""
    ids: set[str] = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = test_dir / name
        fieldnames, rows = _read_tsv_with_header(path)
        if "entity_id" not in fieldnames:
            raise ValidationError(f"{path} missing entity_id")
        for row in rows:
            entity_id = (row.get("entity_id") or "").strip()
            if entity_id:
                ids.add(entity_id)
    return ids


def check_header(fieldnames: list[str], expected: tuple[str, str], path: Path) -> None:
    cleaned = [name.strip() for name in fieldnames]
    if tuple(cleaned[:2]) != expected:
        raise ValidationError(
            f"{path} header must be exactly {' and '.join(expected)}; got {fieldnames}"
        )


def check_unique_s1_rows(
    rows: list[dict[str, str]],
    id_col: str,
    required_s1: list[str],
    path: Path,
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        s1 = (row.get(id_col) or "").strip()
        if not s1:
            raise ValidationError(f"{path} contains an empty {id_col}")
        if s1 in seen:
            duplicates.append(s1)
        seen.add(s1)
    if duplicates:
        raise ValidationError(f"{path} duplicate Source 1 rows: {sorted(set(duplicates))[:20]}")
    missing = [s1 for s1 in required_s1 if s1 not in seen]
    extra = sorted(seen - set(required_s1))
    if missing:
        raise ValidationError(
            f"{path} missing {len(missing)} required Source 1 row(s); first: {missing[:5]}"
        )
    if extra:
        raise ValidationError(
            f"{path} has {len(extra)} unexpected Source 1 id(s); first: {extra[:5]}"
        )
    if len(rows) != len(required_s1):
        raise ValidationError(
            f"{path} must have exactly one row per test S1 "
            f"({len(required_s1)} required, {len(rows)} found)"
        )


def check_id_list(s1_id: str, ids: list[str], path: Path) -> None:
    if len(ids) != len(set(ids)):
        raise ValidationError(f"{path} duplicate IDs in list for {s1_id}")
    for entity_id in ids:
        if entity_id == s1_id or entity_id.startswith("S1-"):
            raise ValidationError(f"{path} S1 self-match / S1 ID in list for {s1_id}: {entity_id}")
        if not (entity_id.startswith("S2-") or entity_id.startswith("S3-")):
            raise ValidationError(f"{path} invalid prefix for {s1_id}: {entity_id}")


def validate_matching(path: Path, required_s1: list[str]) -> dict[str, list[str]]:
    fieldnames, rows = _read_tsv_with_header(path)
    check_header(fieldnames, MATCHING_HEADER, path)
    check_unique_s1_rows(rows, MATCHING_HEADER[0], required_s1, path)
    parsed: dict[str, list[str]] = {}
    for row in rows:
        s1 = row[MATCHING_HEADER[0]].strip()
        ids = parse_id_list(row.get(MATCHING_HEADER[1], ""))
        check_id_list(s1, ids, path)
        parsed[s1] = ids
    return parsed


def validate_candidates(path: Path, required_s1: list[str]) -> dict[str, list[str]]:
    fieldnames, rows = _read_tsv_with_header(path)
    check_header(fieldnames, CANDIDATE_HEADER, path)
    check_unique_s1_rows(rows, CANDIDATE_HEADER[0], required_s1, path)
    parsed: dict[str, list[str]] = {}
    for row in rows:
        s1 = row[CANDIDATE_HEADER[0]].strip()
        ids = parse_id_list(row.get(CANDIDATE_HEADER[1], ""))
        check_id_list(s1, ids, path)
        parsed[s1] = ids
    return parsed


def warn_match_candidate_consistency(
    matches: dict[str, list[str]],
    candidates: dict[str, list[str]],
) -> list[str]:
    warnings: list[str] = []
    for s1, matched in matches.items():
        cand_set = set(candidates.get(s1, []))
        missing = [mid for mid in matched if mid not in cand_set]
        if missing:
            warnings.append(
                f"WARNING: {s1} final matches not in candidate list: {missing[:10]}"
            )
    return warnings


def maybe_check_ids_exist(
    matches: dict[str, list[str]],
    candidates: dict[str, list[str]] | None,
    known_ids: set[str],
) -> None:
    unknown: list[str] = []
    for mapping in (matches, candidates or {}):
        for ids in mapping.values():
            for entity_id in ids:
                if entity_id not in known_ids:
                    unknown.append(entity_id)
    if unknown:
        unique = sorted(set(unknown))
        raise ValidationError(f"Unknown S2/S3 IDs (showing up to 20): {unique[:20]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BER challenge submission TSV files.")
    parser.add_argument("--matching", required=True, type=Path, help="matching_results.tsv")
    parser.add_argument("--candidate", type=Path, default=None, help="candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, type=Path, help="Directory with test_source*.tsv")
    parser.add_argument(
        "--check-ids",
        action="store_true",
        help="Load all S2/S3 IDs and verify existence (memory-heavy on full test).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings: list[str] = []
    try:
        required_s1 = load_source1_ids(args.test_dir)
        matches = validate_matching(args.matching, required_s1)
        candidates = None
        if args.candidate is not None:
            candidates = validate_candidates(args.candidate, required_s1)
            warnings.extend(warn_match_candidate_consistency(matches, candidates))
        if args.check_ids:
            known = load_s2_s3_ids(args.test_dir)
            maybe_check_ids_exist(matches, candidates, known)
    except ValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    for line in warnings:
        print(line, file=sys.stderr)
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
