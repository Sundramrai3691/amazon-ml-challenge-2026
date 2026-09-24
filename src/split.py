"""Reproducible Source-1 entity splits. Never split pair rows."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.load_data import split_source1_ids


def multiplicity_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def s2_s3_composition(true_ids: Iterable[str]) -> str:
    ids = list(true_ids)
    has_s2 = any(str(i).startswith("S2-") for i in ids)
    has_s3 = any(str(i).startswith("S3-") for i in ids)
    if has_s2 and has_s3:
        return "both"
    if has_s2:
        return "s2_only"
    if has_s3:
        return "s3_only"
    return "none"


def entity_stratum(
    country: object,
    true_ids: Iterable[str],
) -> str:
    true_set = {str(x) for x in true_ids}
    country_key = str(country).strip() if country is not None and str(country).strip() else "MISSING"
    return "|".join(
        (
            "singleton" if not true_set else "matched",
            country_key,
            multiplicity_bucket(len(true_set)),
            s2_s3_composition(true_set),
        )
    )


def stratified_source1_split(
    s1_ids: Sequence[str],
    strata: Mapping[str, str],
    *,
    seed: int,
    validation_fraction: float = 0.2,
    min_stratum: int = 2,
) -> tuple[list[str], list[str], dict[str, object]]:
    """80/20 S1 split, stratified when strata are large enough."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    unique = list(dict.fromkeys(str(s) for s in s1_ids))
    counts = Counter(strata.get(s1, "other") for s1 in unique)
    mapped = {}
    collapsed = 0
    for s1 in unique:
        label = strata.get(s1, "other")
        if counts[label] < min_stratum:
            mapped[s1] = "other"
            collapsed += 1
        else:
            mapped[s1] = label
    labels = [mapped[s1] for s1 in unique]
    n_unique_labels = len(set(labels))
    if n_unique_labels < 2 or min(Counter(labels).values()) < 2:
        train_ids, val_ids = split_source1_ids(
            unique, seed=seed, validation_fraction=validation_fraction
        )
        method = "random_entity"
    else:
        frame = pd.DataFrame({"s1": unique, "stratum": labels})
        val_ids: list[str] = []
        rng_seed = seed
        for _stratum, group in frame.groupby("stratum", sort=True):
            n_take = int(round(len(group) * validation_fraction))
            if len(group) <= 1:
                n_take = 0
            else:
                n_take = min(max(n_take, 1), len(group) - 1)
            if n_take > 0:
                sampled = group.sample(n=n_take, random_state=rng_seed)
                val_ids.extend(sampled["s1"].tolist())
                rng_seed += 1
        val_ids = list(dict.fromkeys(val_ids))
        val_set = set(val_ids)
        train_ids = [s1 for s1 in unique if s1 not in val_set]
        if not train_ids and unique:
            train_ids = [unique[0]]
            val_ids = [s for s in unique if s != unique[0]]
        method = "stratified_entity"
    summary = {
        "method": method,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "collapsed_rare_strata": collapsed,
        "n_strata_used": len(set(mapped.values())),
    }
    return train_ids, val_ids, summary


def write_id_list(path: str | Path, ids: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def read_id_list(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]
