"""Structured validation error analysis. Stores IDs and short feature snapshots only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

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


EXACT_FLAGS = ("retrieved_by_exact_name", "retrieved_by_exact_address")
TFIDF_FLAGS = ("retrieved_by_char_name", "retrieved_by_char_address", "retrieved_by_char_combined")
ALL_CHANNEL_FLAGS = EXACT_FLAGS + TFIDF_FLAGS


def _flag(meta_row: Mapping[str, Any] | None, key: str) -> bool:
    if meta_row is None:
        return False
    val = meta_row.get(key)
    try:
        return bool(int(val))
    except (TypeError, ValueError):
        return False


def retrieval_channel_contribution(
    *,
    ground_truth: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    pair_meta: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
    label: str = "",
) -> dict[str, Any]:
    """True-positive retrieval-channel breakdown for diagnostic artifacts.

    Aggregates over ground truth pairs only (not negatives).
    Results are meant to be persisted as ``retrieval_channel_contribution.json``
    so future runs don't need to recompute from in-memory pair_meta.

    Any pair not in ``candidates`` counts as ``missed_by_both``.  If pair_meta
    is missing for a retrieved pair (should not happen), flags conservatively
    become ``unknown``.
    """
    pair_meta = pair_meta or {}

    total_gt_pairs = 0
    recovered_exact_union = 0
    recovered_tfidf_union = 0
    recovered_both = 0
    recovered_either = 0
    missed_by_both = 0

    exact_only_tp = 0
    tfidf_only_tp = 0
    both_exact_and_tfidf_tp = 0
    unknown_retrieval_tp = 0

    char_name_tp = 0
    char_address_tp = 0
    char_combined_tp = 0

    missed_by_both_examples: list[dict[str, Any]] = []

    for s1, true_ids in ground_truth.items():
        cand_set = _set(candidates.get(s1, ()))
        for mid in _set(true_ids):
            total_gt_pairs += 1
            if mid not in cand_set:
                missed_by_both += 1
                if len(missed_by_both_examples) < 50:
                    missed_by_both_examples.append(
                        {
                            "s1": s1,
                            "id": mid,
                            "in_candidates": False,
                            "retrieval_meta_available": False,
                            "flags": {},
                        }
                    )
                continue
            recovered_either += 1
            meta = pair_meta.get((s1, mid))
            exact_hit = any(_flag(meta, f) for f in EXACT_FLAGS)
            tfidf_hit = any(_flag(meta, f) for f in TFIDF_FLAGS)
            if exact_hit:
                recovered_exact_union += 1
            if tfidf_hit:
                recovered_tfidf_union += 1
            if exact_hit and tfidf_hit:
                recovered_both += 1
                both_exact_and_tfidf_tp += 1
            elif exact_hit:
                exact_only_tp += 1
            elif tfidf_hit:
                tfidf_only_tp += 1
            else:
                unknown_retrieval_tp += 1

            if _flag(meta, "retrieved_by_char_name"):
                char_name_tp += 1
            if _flag(meta, "retrieved_by_char_address"):
                char_address_tp += 1
            if _flag(meta, "retrieved_by_char_combined"):
                char_combined_tp += 1

    return {
        "label": label,
        "totals": {
            "total_gt_pairs": total_gt_pairs,
            "recovered_either": recovered_either,
            "recovered_by_exact_union": recovered_exact_union,
            "recovered_by_tfidf_union": recovered_tfidf_union,
            "recovered_by_both_exact_and_tfidf": recovered_both,
            "missed_by_both": missed_by_both,
        },
        "tp_recovery_breakdown": {
            "exact_only": exact_only_tp,
            "tfidf_only": tfidf_only_tp,
            "both_exact_and_tfidf": both_exact_and_tfidf_tp,
            "unknown_retrieval_path": unknown_retrieval_tp,
        },
        "tfidf_channel_breakdown": {
            "char_name_tp": char_name_tp,
            "char_address_tp": char_address_tp,
            "char_combined_tp": char_combined_tp,
        },
        "missed_by_both_examples": missed_by_both_examples,
    }
