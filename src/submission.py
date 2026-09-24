"""Write matching_results.tsv and candidate_pairs.tsv."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

MATCHING_COLUMNS = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")
TSV_SEP = "\t"


class SubmissionError(ValueError):
    """Invalid submission content before write."""


def _as_id_list(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        entity_id = str(raw).strip()
        if not entity_id or entity_id in seen:
            continue
        if entity_id.startswith("S1-"):
            raise SubmissionError(f"S1 IDs are not allowed in match/candidate lists: {entity_id}")
        if not (entity_id.startswith("S2-") or entity_id.startswith("S3-")):
            raise SubmissionError(f"Invalid ID prefix (expected S2-/S3-): {entity_id}")
        seen.add(entity_id)
        ordered.append(entity_id)
    return ordered


def format_id_list(ids: Sequence[str]) -> str:
    return ",".join(ids)


def build_submission_frames(
    s1_ids: Sequence[str],
    candidates: Mapping[str, Iterable[str] | None],
    matches: Mapping[str, Iterable[str] | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not s1_ids:
        raise SubmissionError("s1_ids must contain every test Source 1 entity")

    unique_s1 = list(dict.fromkeys(str(s) for s in s1_ids))
    match_rows: list[dict[str, str]] = []
    cand_rows: list[dict[str, str]] = []

    for s1 in unique_s1:
        cand_ids = _as_id_list(candidates.get(s1, []))
        match_ids = _as_id_list(matches.get(s1, []))
        cand_set = set(cand_ids)
        extra = [mid for mid in match_ids if mid not in cand_set]
        if extra:
            raise SubmissionError(
                f"Final matches for {s1} are not a subset of candidates: {extra}"
            )
        match_rows.append(
            {
                MATCHING_COLUMNS[0]: s1,
                MATCHING_COLUMNS[1]: format_id_list(match_ids),
            }
        )
        cand_rows.append(
            {
                CANDIDATE_COLUMNS[0]: s1,
                CANDIDATE_COLUMNS[1]: format_id_list(cand_ids),
            }
        )

    matching = pd.DataFrame(match_rows, columns=list(MATCHING_COLUMNS))
    candidate = pd.DataFrame(cand_rows, columns=list(CANDIDATE_COLUMNS))
    return matching, candidate


def write_submission(
    s1_ids: Sequence[str],
    candidates: Mapping[str, Iterable[str] | None],
    matches: Mapping[str, Iterable[str] | None],
    output_dir: str | Path,
    *,
    matching_name: str = "matching_results.tsv",
    candidate_name: str = "candidate_pairs.tsv",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matching, candidate = build_submission_frames(s1_ids, candidates, matches)
    matching_path = output_dir / matching_name
    candidate_path = output_dir / candidate_name
    matching.to_csv(matching_path, sep=TSV_SEP, index=False)
    candidate.to_csv(candidate_path, sep=TSV_SEP, index=False)
    return matching_path, candidate_path
