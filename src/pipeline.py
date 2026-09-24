"""E0–E5 baseline pipeline. Blocking stays separate from matching."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.audit import render_audit_markdown, run_full_audit
from src.blocking import (
    cap_preferring_exact,
    evaluate_candidate_recall,
    generate_candidate_pool,
    generate_exact_candidates,
    union_exact_with_tfidf_k,
)
from src.error_analysis import analyze_errors, render_error_markdown
from src.experiment_log import append_experiment
from src.features import FEATURE_NAMES, fit_rarity_stats, feature_matrix, records_by_id
from src.load_data import ground_truth_to_sets, load_training_data
from src.predict import apply_threshold, best_threshold_row, score_pairs, sweep_thresholds
from src.split import entity_stratum, stratified_source1_split, write_id_list
from src.submission import write_submission
from src.train import build_pair_labels, downsample_negatives, fit_pair_model, make_model, save_model


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _filter(frame: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return frame[frame["entity_id"].astype(str).isin(ids)].copy()


def run_audit(config: dict[str, Any], repo: Path) -> Path:
    train_dir = repo / config["paths"]["train_dir"]
    test_dir = repo / config["paths"]["test_dir"]
    report = run_full_audit(train_dir, test_dir, include_test_s2_s3=False)
    markdown = render_audit_markdown(report)
    out = repo / "docs" / "data_audit.md"
    out.write_text(markdown, encoding="utf-8")
    (repo / "artifacts" / "data_audit.json").parent.mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def run_baseline(
    config: dict[str, Any],
    repo: Path,
    *,
    limit_s1: int | None = None,
    skip_tfidf: bool = False,
    write_val_submission: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    train_dir = repo / config["paths"]["train_dir"]
    data = load_training_data(train_dir)
    s1 = data["source1"]
    s2 = data["source2"]
    s3 = data["source3"]
    gt_all = ground_truth_to_sets(data["ground_truth"])
    s1_ids = [str(x) for x in s1["entity_id"].tolist()]
    if limit_s1 is not None:
        s1_ids = s1_ids[: int(limit_s1)]
        s1 = _filter(s1, set(s1_ids))
        gt_all = {k: gt_all.get(k, set()) for k in s1_ids}

    country_by_id = {str(r.entity_id): r.country for r in s1.itertuples(index=False)}
    strata = {sid: entity_stratum(country_by_id.get(sid, ""), gt_all.get(sid, set())) for sid in s1_ids}
    train_ids, val_ids, split_summary = stratified_source1_split(
        s1_ids,
        strata,
        seed=seed,
        validation_fraction=float(config.get("validation_fraction", 0.2)),
    )
    split_dir = repo / "artifacts" / "splits"
    write_id_list(split_dir / "train_s1_ids.txt", train_ids)
    write_id_list(split_dir / "validation_s1_ids.txt", val_ids)
    (split_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    train_s1 = _filter(s1, set(train_ids))
    val_s1 = _filter(s1, set(val_ids))
    gt_train = {k: gt_all.get(k, set()) for k in train_ids}
    gt_val = {k: gt_all.get(k, set()) for k in val_ids}

    blocking_cfg = config.get("blocking", {})
    exact_methods = blocking_cfg.get("methods")
    k_grid = list(blocking_cfg.get("char_tfidf", {}).get("k_grid", [10, 20, 50, 100]))
    k_final = int(blocking_cfg.get("char_tfidf", {}).get("k_final", 50))
    max_cands = blocking_cfg.get("max_candidates_per_s1", 250)
    tfidf_cfg = blocking_cfg.get("char_tfidf", {})
    use_tfidf = bool(tfidf_cfg.get("enabled", True)) and not skip_tfidf

    exact_val, _exact_val_meta = generate_exact_candidates(
        val_s1, s2, s3, methods=exact_methods, max_candidates_per_s1=None
    )
    val_pool, val_meta, tfidf_runtime = generate_candidate_pool(
        val_s1,
        s2,
        s3,
        exact_methods=exact_methods,
        tfidf_k=max(k_grid) if use_tfidf else 0,
        use_char_tfidf=use_tfidf,
        max_candidates_per_s1=None,
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [3, 5])),
        min_df=int(tfidf_cfg.get("min_df", 1)),
        max_features=int(tfidf_cfg.get("max_features", 50000)),
    )
    s1_records = records_by_id(s1)
    recall_by_k = {}
    for k in k_grid:
        mixed = union_exact_with_tfidf_k(exact_val, val_pool, val_meta, k if use_tfidf else 0)
        capped = cap_preferring_exact(mixed, val_meta, max_cands)
        recall_by_k[str(k)] = evaluate_candidate_recall(
            capped, gt_val, n_s2=len(s2), n_s3=len(s3), s1_records=s1_records
        )

    val_candidates = cap_preferring_exact(
        union_exact_with_tfidf_k(exact_val, val_pool, val_meta, k_final if use_tfidf else 0),
        val_meta,
        max_cands,
    )
    val_recall = evaluate_candidate_recall(
        val_candidates, gt_val, n_s2=len(s2), n_s3=len(s3), s1_records=s1_records
    )

    train_pool, train_meta, _ = generate_candidate_pool(
        train_s1,
        s2,
        s3,
        exact_methods=exact_methods,
        tfidf_k=k_final if use_tfidf else 0,
        use_char_tfidf=use_tfidf,
        max_candidates_per_s1=None,
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [3, 5])),
        min_df=int(tfidf_cfg.get("min_df", 1)),
        max_features=int(tfidf_cfg.get("max_features", 50000)),
    )
    train_pool = cap_preferring_exact(train_pool, train_meta, max_cands)

    rarity = fit_rarity_stats([train_s1, s2, s3])
    pairs, y, pair_stats = build_pair_labels(train_pool, gt_train)
    sample_cfg = config.get("sampling", {})
    pairs, y = downsample_negatives(
        pairs,
        y,
        max_neg_per_pos=sample_cfg.get("max_neg_per_pos"),
        seed=seed,
    )
    cand_records = records_by_id(s2)
    cand_records.update(records_by_id(s3))
    X_train = feature_matrix(pairs, s1_records, cand_records, train_meta, rarity=rarity)
    model_cfg = config.get("model", {})
    model = make_model(model_cfg.get("type", "gbdt"), **(model_cfg.get("params") or {}))
    if len(y) == 0:
        raise ValueError("No training pairs produced by blocking; cannot fit a matcher.")
    fit_pair_model(model, X_train.to_numpy(dtype=float), y)
    models_dir = repo / config["paths"].get("models_dir", "models")
    save_model(
        model,
        models_dir / "baseline_gbdt.joblib",
        feature_names=list(X_train.columns or FEATURE_NAMES),
    )

    val_pairs = [(s1, cand) for s1, cands in val_candidates.items() for cand in cands]
    X_val = feature_matrix(val_pairs, s1_records, cand_records, val_meta, rarity=rarity)
    scores = score_pairs(model, X_val.to_numpy(dtype=float)) if len(val_pairs) else []
    grid = list(config.get("threshold", {}).get("grid", [i / 100 for i in range(5, 100, 5)]))
    sweep = sweep_thresholds(val_pairs, scores, gt_val, grid, s1_ids=val_ids)
    best = best_threshold_row(sweep) if sweep else {"threshold": 0.7, "macro_f0_5": 0.0}
    preds = apply_threshold(val_pairs, scores, threshold=float(best["threshold"]), s1_ids=val_ids)

    errors = analyze_errors(
        gt_val,
        preds,
        val_candidates,
        val_pairs,
        scores,
        threshold=float(best["threshold"]),
        s1_records=s1_records,
        cand_records=cand_records,
        pair_meta=val_meta,
    )
    err_path = repo / "docs" / "baseline_error_analysis.md"
    err_path.write_text(render_error_markdown(errors), encoding="utf-8")

    if write_val_submission:
        out_dir = repo / config["paths"]["output_dir"]
        write_submission(val_ids, val_candidates, preds, out_dir)

    runtime = time.perf_counter() - started
    result = {
        "split": split_summary,
        "pair_stats": pair_stats.__dict__,
        "val_recall": val_recall,
        "recall_by_k": recall_by_k,
        "tfidf_runtime": tfidf_runtime,
        "best_threshold": best,
        "runtime_seconds": runtime,
        "n_train_s1": len(train_ids),
        "n_val_s1": len(val_ids),
        "n_train_pairs": int(len(y)),
        "feature_names": list(X_train.columns),
    }
    (repo / "artifacts" / "baseline_last.json").write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
    append_experiment(
        repo / "experiments" / "experiment_log.csv",
        {
            "experiment_id": "E0-E5-baseline",
            "description": "Entity-level GBDT baseline",
            "hypothesis": "Exact blocking union char TF-IDF plus GBDT and threshold sweep beats a naive 0.5 cutoff",
            "dataset_version": "challenge_train",
            "validation_split_seed": seed,
            "blocking_version": f"exact+char_tfidf_k{k_final}" if use_tfidf else "exact",
            "feature_version": "e3_v1",
            "model": model_cfg.get("type", "gbdt"),
            "model_config": json.dumps(model_cfg.get("params") or {}),
            "candidate_recall": val_recall.get("candidate_recall"),
            "avg_candidates": val_recall.get("average_candidates_per_s1"),
            "median_candidates": val_recall.get("median_candidates_per_s1"),
            "p95_candidates": val_recall.get("p95_candidates_per_s1"),
            "p99_candidates": val_recall.get("p99_candidates_per_s1"),
            "max_candidates": val_recall.get("max_candidates_per_s1"),
            "reduction_ratio": val_recall.get("reduction_ratio"),
            "precision": best.get("mean_precision"),
            "recall": best.get("mean_recall"),
            "f0_5": best.get("macro_f0_5"),
            "macro_f05": best.get("macro_f0_5"),
            "threshold": best.get("threshold"),
            "runtime_seconds": runtime,
            "notes": f"limit_s1={limit_s1}; skip_tfidf={skip_tfidf}",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E0-E5 baseline pipeline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--limit-s1", type=int, default=None, help="Optional cap on S1 rows for a measured slice")
    parser.add_argument("--skip-tfidf", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = load_config(repo / args.config)
    if args.audit_only:
        path = run_audit(config, repo)
        print(f"Wrote {path}")
        return
    result = run_baseline(config, repo, limit_s1=args.limit_s1, skip_tfidf=args.skip_tfidf)
    print(json.dumps({k: result[k] for k in ("split", "best_threshold", "val_recall", "runtime_seconds") if k in result}, default=str, indent=2))


if __name__ == "__main__":
    main()
