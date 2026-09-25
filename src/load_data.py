"""TSV loaders for the Business Entity Resolution challenge.

Every read uses ``sep="\\t"``. Raw files are never modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
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


@dataclass
class SampleDiagnostics:
    """Diagnostic info returned by the smoke-sample loader.

    ``rows_requested`` is the nominal cap (0/None = full load).
    ``rows_loaded`` is the actual number of unique rows returned (may exceed
    ``rows_requested`` if required positive IDs alone overflow the cap).
    ``required_ids`` are the S2/S3 match IDs the caller asked to retain.
    ``required_ids_found`` are the subset actually observed while streaming.
    ``required_ids_missing`` are IDs listed as required but never encountered.
    """

    rows_requested: int
    rows_loaded: int
    required_ids: set[str] = field(default_factory=set)
    required_ids_found: set[str] = field(default_factory=set)
    required_ids_missing: set[str] = field(default_factory=set)
    via_positive_keep: int = 0
    via_sample: int = 0
    stream_chunks: int = 0


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


def _stream_sample_source(
    path: str | Path,
    *,
    max_rows: int,
    required_ids: Iterable[str] | None = None,
    seed: int = 42,
    chunksize: int = 50_000,
) -> tuple[pd.DataFrame, SampleDiagnostics]:
    """Memory-safe streaming S2/S3 sampler for diagnostic smoke tests.

    The full TSV is NEVER materialized as a single DataFrame.
    Rows are read in chunks; required IDs are always retained; the remaining
    capacity is filled with a deterministic reservoir sample over the full
    stream (skipping duplicates). Returns the resulting DataFrame plus
    diagnostics about which required IDs were observed/missing.

    If the set of required positive IDs alone exceeds ``max_rows``, every
    required ID is still retained and the returned frame may be larger than
    ``max_rows`` (overflow is tracked in diagnostics and reported to the
    caller so it is never silent).
    """
    path = _as_path(path)
    if max_rows is None or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    rng = np.random.default_rng(int(seed))

    required = {str(eid) for eid in (required_ids or ()) if str(eid)}
    seen_ids: set[str] = set()
    kept_rows: list[dict[str, str]] = []
    reservoir: list[dict[str, str]] = []
    reservoir_cap = max(0, int(max_rows) - len(required))
    # If required IDs alone already exceed max_rows, reservoir stays empty
    # and we still collect every required row (returned size may exceed cap).
    non_required_count = 0
    required_found: set[str] = set()
    chunks_seen = 0

    for chunk in iter_source_chunks(path, chunksize=chunksize):
        chunks_seen += 1
        for row in chunk.itertuples(index=False):
            mapping = {col: str(getattr(row, col, "")) for col in SOURCE_COLUMNS}
            eid = mapping.get("entity_id", "")
            if not eid or eid in seen_ids:
                continue
            if eid in required:
                seen_ids.add(eid)
                required_found.add(eid)
                kept_rows.append(mapping)
                continue
            non_required_count += 1
            if reservoir_cap <= 0:
                continue
            if len(reservoir) < reservoir_cap:
                seen_ids.add(eid)
                reservoir.append(mapping)
            else:
                # Algorithm R: reservoir sampling, 1-based index.
                j = int(rng.integers(0, non_required_count))
                if j < reservoir_cap:
                    removed = reservoir[j]
                    seen_ids.discard(removed.get("entity_id", ""))
                    reservoir[j] = mapping
                    seen_ids.add(eid)

    merged = kept_rows + reservoir
    frame = pd.DataFrame(merged, columns=list(SOURCE_COLUMNS))
    # Preserve column order and str dtype.
    for col in SOURCE_COLUMNS:
        frame[col] = frame[col].astype(str)
    _require_columns(frame, SOURCE_COLUMNS, path)

    missing = required - required_found
    diagnostics = SampleDiagnostics(
        rows_requested=int(max_rows),
        rows_loaded=int(frame.shape[0]),
        required_ids=set(required),
        required_ids_found=required_found,
        required_ids_missing=missing,
        via_positive_keep=len(kept_rows),
        via_sample=len(reservoir),
        stream_chunks=chunks_seen,
    )
    return frame, diagnostics


def stream_sample_source2(
    path: str | Path,
    *,
    max_rows: int,
    required_ids: Iterable[str] | None = None,
    seed: int = 42,
    chunksize: int = 50_000,
) -> tuple[pd.DataFrame, SampleDiagnostics]:
    """Stream-sample S2. Thin wrapper around ``_stream_sample_source``."""
    return _stream_sample_source(
        path,
        max_rows=max_rows,
        required_ids=required_ids,
        seed=seed,
        chunksize=chunksize,
    )


def stream_sample_source3(
    path: str | Path,
    *,
    max_rows: int,
    required_ids: Iterable[str] | None = None,
    seed: int = 42,
    chunksize: int = 50_000,
) -> tuple[pd.DataFrame, SampleDiagnostics]:
    """Stream-sample S3. Thin wrapper around ``_stream_sample_source``."""
    return _stream_sample_source(
        path,
        max_rows=max_rows,
        required_ids=required_ids,
        seed=seed,
        chunksize=chunksize,
    )


def _required_match_ids(
    gt_frame: pd.DataFrame,
    s1_ids: Iterable[str],
    *,
    prefix: str,
) -> set[str]:
    """Return S2 or S3 true-match IDs referenced by the given S1 subset.

    Scans only ground-truth rows whose source1_entity_id is in ``s1_ids``.
    The full ground-truth table is loaded once up-front (its footprint is
    ~127 MB for the training set), which is acceptable.
    """
    if prefix not in ("S2-", "S3-"):
        raise ValueError(f"prefix must be 'S2-' or 'S3-', got {prefix!r}")
    s1_set = {str(s) for s in s1_ids}
    out: set[str] = set()
    for row in gt_frame.itertuples(index=False):
        s1 = str(getattr(row, "source1_entity_id", "")).strip()
        if s1 not in s1_set:
            continue
        raw = getattr(row, "matched_entity_ids", None)
        for mid in parse_matched_ids(raw, strict=True):
            if str(mid).startswith(prefix):
                out.add(str(mid))
    return out


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
