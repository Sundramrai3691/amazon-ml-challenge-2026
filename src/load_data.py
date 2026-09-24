"""TSV loaders for the Business Entity Resolution challenge.

Every read uses ``sep="\\t"``. Raw files are never modified.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

TSV_SEP = "\t"

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")

TRAIN_FILENAMES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
    "ground_truth": "train_ground_truth.tsv",
}
TEST_FILENAMES = {
    "source1": "test_source1.tsv",
    "source2": "test_source2.tsv",
    "source3": "test_source3.tsv",
}


class DataSchemaError(ValueError):
    """Raised when a TSV does not match the expected challenge schema."""


class GroundTruthError(ValueError):
    """Raised when ground-truth IDs are malformed."""


S1_ID_RE = re.compile(r"^S1-[A-Za-z0-9]+$")
MATCH_ID_RE = re.compile(r"^S[23]-[A-Za-z0-9]+$")


def _as_path(path: str | Path) -> Path:
    return Path(path)


def _read_tsv(
    path: str | Path,
    *,
    usecols: Sequence[str] | None = None,
    dtype: dict[str, Any] | None = None,
) -> pd.DataFrame:
    path = _as_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TSV not found: {path}")

    try:
        frame = pd.read_csv(
            path,
            sep=TSV_SEP,
            dtype=dtype or str,
            keep_default_na=False,
            na_filter=False,
            usecols=list(usecols) if usecols is not None else None,
        )
    except Exception as exc:  # pandas may raise ParserError subclasses
        raise DataSchemaError(f"Failed to parse TSV {path}: {exc}") from exc

    if frame.shape[1] == 1 and usecols is None:
        sample = path.read_text(encoding="utf-8").splitlines()[:3]
        joined = " | ".join(sample)
        if "\t" not in path.read_text(encoding="utf-8")[:4096]:
            raise DataSchemaError(
                f"{path} appears not to be tab-separated (single column). "
                f"First lines: {joined}"
            )
    return frame


def _require_columns(frame: pd.DataFrame, required: Sequence[str], path: str | Path) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise DataSchemaError(f"{path} missing required columns {missing}; got {list(frame.columns)}")


def load_source_table(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a source TSV and validate the entity schema."""
    usecols = list(columns) if columns is not None else None
    if usecols is not None:
        unknown = [col for col in usecols if col not in SOURCE_COLUMNS]
        if unknown:
            raise DataSchemaError(f"Unknown source columns requested: {unknown}")
        for col in SOURCE_COLUMNS:
            if col not in usecols and col == "entity_id":
                usecols = ["entity_id", *usecols]
    frame = _read_tsv(path, usecols=usecols)
    required = SOURCE_COLUMNS if columns is None else tuple(usecols or SOURCE_COLUMNS)
    _require_columns(frame, required, path)
    return frame


def load_source1(path: str | Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return load_source_table(path, columns=columns)


def load_source2(path: str | Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return load_source_table(path, columns=columns)


def load_source3(path: str | Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return load_source_table(path, columns=columns)


def parse_matched_ids(raw: str | None, *, strict: bool = True) -> list[str]:
    """Parse a comma-separated matched-ID cell. Empty cells become [].

    When ``strict`` is True (default), reject IDs that are not S2-/S3-.
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for part in text.split(","):
        entity_id = part.strip()
        if not entity_id:
            continue
        if strict and not MATCH_ID_RE.match(entity_id):
            raise GroundTruthError(f"Malformed matched entity id: {entity_id!r}")
        if entity_id not in seen:
            seen.add(entity_id)
            ordered.append(entity_id)
    return ordered


def ground_truth_to_sets(frame: pd.DataFrame, *, strict: bool = True) -> dict[str, set[str]]:
    """Map each S1 ID to a set of true S2/S3 IDs. Empty lists become set()."""
    mapping: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        s1 = str(row.source1_entity_id).strip()
        if strict and not S1_ID_RE.match(s1):
            raise GroundTruthError(f"Malformed Source 1 id: {s1!r}")
        if s1 in mapping:
            raise GroundTruthError(f"Duplicate Source 1 id in ground truth: {s1}")
        mapping[s1] = set(parse_matched_ids(row.matched_entity_ids, strict=strict))
    return mapping


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    frame = _read_tsv(path)
    _require_columns(frame, GROUND_TRUTH_COLUMNS, path)
    frame = frame.copy()
    frame["matched_entity_id_list"] = frame["matched_entity_ids"].map(parse_matched_ids)
    return frame


def iter_source_chunks(
    path: str | Path,
    *,
    chunksize: int,
    usecols: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Chunked TSV reader for large sources. Still uses sep='\\t'."""
    path = _as_path(path)
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    reader = pd.read_csv(
        path,
        sep=TSV_SEP,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        usecols=list(usecols) if usecols is not None else None,
        chunksize=chunksize,
    )
    for chunk in reader:
        if usecols is None:
            _require_columns(chunk, SOURCE_COLUMNS, path)
        yield chunk


def load_training_data(train_dir: str | Path) -> dict[str, pd.DataFrame]:
    train_dir = _as_path(train_dir)
    return {
        "source1": load_source1(train_dir / TRAIN_FILENAMES["source1"]),
        "source2": load_source2(train_dir / TRAIN_FILENAMES["source2"]),
        "source3": load_source3(train_dir / TRAIN_FILENAMES["source3"]),
        "ground_truth": load_ground_truth(train_dir / TRAIN_FILENAMES["ground_truth"]),
    }


def load_test_data(test_dir: str | Path) -> dict[str, pd.DataFrame]:
    test_dir = _as_path(test_dir)
    return {
        "source1": load_source1(test_dir / TEST_FILENAMES["source1"]),
        "source2": load_source2(test_dir / TEST_FILENAMES["source2"]),
        "source3": load_source3(test_dir / TEST_FILENAMES["source3"]),
    }


def split_source1_ids(
    s1_ids: Sequence[str],
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[list[str], list[str]]:
    """Reproducible Source-1 entity split. Primary validation protocol."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    unique = list(dict.fromkeys(str(s) for s in s1_ids))
    rng_frame = pd.Series(unique)
    sampled = rng_frame.sample(frac=1.0, random_state=seed).tolist()
    n_val = max(1, int(round(len(sampled) * validation_fraction))) if sampled else 0
    n_val = min(n_val, max(0, len(sampled) - 1)) if len(sampled) > 1 else n_val
    val_ids = sampled[:n_val]
    train_ids = sampled[n_val:]
    return train_ids, val_ids
