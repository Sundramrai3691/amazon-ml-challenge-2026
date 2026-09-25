"""Structured validation error analysis. Stores IDs and short feature snapshots only."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from src.features import pair_feature_dict


def _set(values: Iterable[str] | None) -> set[str]:
    return {str(v) for v in values} if values else set()


def classify_false_positive(feats: Mapping[str, float], *, is_singleton: bool) -> str:
    if is_singleton:
        return "singleton_false_positive"
    name = float(feats.get("name_char_similarity", 0))
    addr = float(feats.get("addr_char_similarity", 0))
    if name >= 0.85 and addr < 0.4:
        return "high_name_weak_address"
    if addr >= 0.85 and name < 0.4:
        return "high_address_weak_name"
    if name >= 0.8 and addr >= 0.8:
        return "both_high_wrong_entity"
    if float(feats.get("name_token_jaccard", 0)) >= 0.3 and float(feats.get("shared_rare_name_tokens", 0)) == 0:
        return "common_token_false_positive"
    return "other_false_positive"


def classify_false_negative(
    *,
    in_candidates: bool,
    score: float | None,
    threshold: float,
    feats: Mapping[str, float] | None,
) -> str:
    if not in_candidates:
        return "missed_by_blocking"
    if score is None:
        return "candidate_retrieved_low_score"
    if threshold - 0.10 <= score < threshold:
        return "just_below_threshold"
    feats = feats or {}
    if any(
        float(feats.get(key, 0))
        for key in ("s1_name_missing", "cand_name_missing", "s1_addr_missing", "cand_addr_missing")
    ):
        return "missing_field_case"
    if float(feats.get("name_char_similarity", 1)) < 0.5:
        return "noisy_name_case"
    if float(feats.get("addr_char_similarity", 1)) < 0.5:
        return "noisy_address_case"
    if score < threshold:
        return "candidate_retrieved_low_score"
    return "other_false_negative"


def analyze_errors(
    ground_truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    pairs: Sequence[tuple[str, str]],
    scores: Sequence[float],
    *,
    threshold: float,
    s1_records: Mapping[str, Mapping[str, object]],
    cand_records: Mapping[str, Mapping[str, object]],
    pair_meta: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
    max_examples: int = 8,
) -> dict[str, dict[str, object]]:
    score_by_pair = {(s1, cand): float(score) for (s1, cand), score in zip(pairs, scores)}
    pair_meta = pair_meta or {}
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)

    def _add(cat: str, payload: dict[str, object]) -> None:
        counts[cat] += 1
        if len(examples[cat]) < max_examples:
            examples[cat].append(payload)

    for s1, true_ids in ground_truth.items():
        true_set = _set(true_ids)
        pred_set = _set(predictions.get(s1, ()))
        cand_set = _set(candidates.get(s1, ()))
        is_singleton = not true_set
        s1_rec = s1_records.get(s1, {"entity_id": s1})

        for fp_id in sorted(pred_set - true_set):
            cand_rec = cand_records.get(fp_id, {"entity_id": fp_id})
            feats = pair_feature_dict(s1_rec, cand_rec, pair_meta.get((s1, fp_id)))
            cat = classify_false_positive(feats, is_singleton=is_singleton)
            _add(
                cat,
                {
                    "s1": s1,
                    "id": fp_id,
                    "score": score_by_pair.get((s1, fp_id)),
                    "name_sim": round(float(feats.get("name_char_similarity", 0)), 3),
                    "addr_sim": round(float(feats.get("addr_char_similarity", 0)), 3),
                },
            )

        for fn_id in sorted(true_set - pred_set):
            in_cands = fn_id in cand_set
            score = score_by_pair.get((s1, fn_id))
            cand_rec = cand_records.get(fn_id, {"entity_id": fn_id})
            feats = pair_feature_dict(s1_rec, cand_rec, pair_meta.get((s1, fn_id))) if in_cands else None
            cat = classify_false_negative(
                in_candidates=in_cands, score=score, threshold=threshold, feats=feats
            )
            payload: dict[str, object] = {"s1": s1, "id": fn_id, "score": score, "in_candidates": in_cands}
            if feats:
                payload["name_sim"] = round(float(feats.get("name_char_similarity", 0)), 3)
                payload["addr_sim"] = round(float(feats.get("addr_char_similarity", 0)), 3)
            _add(cat, payload)

    total = sum(counts.values()) or 1
    report = {}
    for cat, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        report[cat] = {
            "count": count,
            "fraction": count / total,
            "examples": examples.get(cat, []),
        }
    report["_totals"] = {"n_labeled_errors": sum(counts.values()), "n_categories": len(counts)}
    return report


def render_error_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Baseline error analysis",
        "",
        "Validation-only. Counts are measured. Examples are IDs plus compact feature snapshots.",
        "Raw business strings are not dumped.",
        "",
    ]
    totals = report.get("_totals", {})
    lines.append(f"Labeled errors: **{totals.get('n_labeled_errors', 0)}**")
    lines.append("")
    for cat, payload in report.items():
        if cat.startswith("_"):
            continue
        if not isinstance(payload, dict):
            continue
        lines.append(f"## {cat}")
        lines.append("")
        lines.append(f"- count: **{payload.get('count')}**")
        lines.append(f"- fraction: **{payload.get('fraction'):.4f}**")
        examples = payload.get("examples") or []
        if examples:
            lines.append("- examples:")
            for ex in examples:
                lines.append(f"  - `{ex}`")
        lines.append("")
    return "\n".join(lines) + "\n"
