"""Append-only experiment logging. Never invent metric values."""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

LOG_FIELDS = (
    "experiment_id",
    "timestamp",
    "git_commit",
    "description",
    "hypothesis",
    "dataset_version",
    "validation_split_seed",
    "blocking_version",
    "feature_version",
    "model",
    "model_config",
    "candidate_recall",
    "avg_candidates",
    "median_candidates",
    "p95_candidates",
    "p99_candidates",
    "max_candidates",
    "reduction_ratio",
    "precision",
    "recall",
    "f0_5",
    "macro_f05",
    "threshold",
    "runtime_seconds",
    "notes",
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def append_experiment(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {field: row.get(field, "") for field in LOG_FIELDS}
    payload["timestamp"] = payload["timestamp"] or datetime.now(timezone.utc).isoformat()
    payload["git_commit"] = payload["git_commit"] or git_commit()
    if payload.get("macro_f05") in ("", None) and payload.get("f0_5") not in ("", None):
        payload["macro_f05"] = payload["f0_5"]
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(payload)
