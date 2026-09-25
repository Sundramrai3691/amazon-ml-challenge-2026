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
    evaluate_candidate_recall_per_source,
    generate_candidate_pool,
    generate_exact_candidates,
    union_exact_with_tfidf_k,
)
from src.error_analysis import analyze_errors, render_error_markdown
from src.experiment_log import append_experiment
from src.features import FEATURE_NAMES, fit_rarity_stats, feature_matrix, records_by_id
from src.load_data import (
    TRAIN_FILENAMES,
    SampleDiagnostics,
    _required_match_ids,
    ground_truth_to_sets,
    load_ground_truth,
    load_source1,
    load_training_data,
    stream_sample_source2,
    stream_sample_source3,
)
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
    limit_s2: int | None = None,
    limit_s3: int | None = None,
    skip_tfidf: bool = False,
    write_val_submission: bool = True,
    smoke_kgrid: int | None = None,
    smoke_max_features: int | None = None,
    loader_precheck_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    train_dir = repo / config["paths"]["train_dir"]
    train_p = Path(train_dir)

    smoke_mode = (limit_s2 is not None and int(limit_s2) > 0) or (
        limit_s3 is not None and int(limit_s3) > 0
    )
    loader_diagnostics: dict[str, Any] = {}

    if smoke_mode:
        s1_full = load_source1(train_p / TRAIN_FILENAMES["source1"])
        s1_ids_all = [str(x) for x in s1_full["entity_id"].tolist()]
        if limit_s1 is not None and int(limit_s1) > 0:
            s1_ids_selected = s1_ids_all[: int(limit_s1)]
        else:
            s1_ids_selected = s1_ids_all
        s1 = _filter(s1_full, set(s1_ids_selected))
        del s1_full

        gt_frame = load_ground_truth(train_p / TRAIN_FILENAMES["ground_truth"])
        req_s2 = _required_match_ids(gt_frame, s1_ids_selected, prefix="S2-")
        req_s3 = _required_match_ids(gt_frame, s1_ids_selected, prefix="S3-")

        load_s2_t0 = time.perf_counter()
        if limit_s2 is not None and int(limit_s2) > 0:
            s2, s2_diag = stream_sample_source2(
                train_p / TRAIN_FILENAMES["source2"],
                max_rows=int(limit_s2),
                required_ids=req_s2,
                seed=seed,
            )
        else:
            from src.load_data import load_source2
            s2 = load_source2(train_p / TRAIN_FILENAMES["source2"])
            s2_diag = None
        load_s2_elapsed = time.perf_counter() - load_s2_t0

        load_s3_t0 = time.perf_counter()
        if limit_s3 is not None and int(limit_s3) > 0:
            s3, s3_diag = stream_sample_source3(
                train_p / TRAIN_FILENAMES["source3"],
                max_rows=int(limit_s3),
                required_ids=req_s3,
                seed=seed,
            )
        else:
            from src.load_data import load_source3
            s3 = load_source3(train_p / TRAIN_FILENAMES["source3"])
            s3_diag = None
        load_s3_elapsed = time.perf_counter() - load_s3_t0

        gt_all = ground_truth_to_sets(gt_frame, strict=True)
        gt_all = {k: gt_all.get(k, set()) for k in s1_ids_selected}
        del gt_frame
        s2_ids = set(s2["entity_id"].astype(str).tolist())
        s3_ids = set(s3["entity_id"].astype(str).tolist())
        for s1_id in list(gt_all.keys()):
            gt_all[s1_id] = {mid for mid in gt_all[s1_id] if mid in s2_ids or mid in s3_ids}

        def _diag(d: SampleDiagnostics | None, elapsed: float) -> dict[str, Any] | None:
            if d is None:
                return None
            return {
                "rows_requested": d.rows_requested,
                "rows_loaded": d.rows_loaded,
                "required_ids_count": len(d.required_ids),
                "required_ids_found_count": len(d.required_ids_found),
                "required_ids_missing_count": len(d.required_ids_missing),
                "required_ids_missing": sorted(d.required_ids_missing)[:200],
                "via_positive_keep": d.via_positive_keep,
                "via_sample": d.via_sample,
                "stream_chunks": d.stream_chunks,
                "load_seconds": float(elapsed),
            }

        loader_diagnostics = {
            "smoke_mode": True,
            "s1_selected_count": len(s1_ids_selected),
            "s2": _diag(s2_diag, load_s2_elapsed),
            "s3": _diag(s3_diag, load_s3_elapsed),
        }
    else:
        data = load_training_data(train_dir)
        s1 = data["source1"]
        s2 = data["source2"]
        s3 = data["source3"]
        gt_all = ground_truth_to_sets(data["ground_truth"])
        s1_ids_selected = [str(x) for x in s1["entity_id"].tolist()]
        if limit_s1 is not None and int(limit_s1) > 0:
            s1_ids_selected = s1_ids_selected[: int(limit_s1)]
            s1 = _filter(s1, set(s1_ids_selected))
            gt_all = {k: gt_all.get(k, set()) for k in s1_ids_selected}

    if loader_precheck_only:
        return {
            "loader_precheck_only": True,
            "runtime_seconds": time.perf_counter() - started,
            "n_s1": len(s1),
            "n_s2": len(s2),
            "n_s3": len(s3),
            "loader_diagnostics": loader_diagnostics,
            "gt_for_s1_positive_count": sum(
                1 for v in gt_all.values() if v
            ),
            "gt_for_s1_match_ids_s2": sum(
                1 for v in gt_all.values() for mid in v if str(mid).startswith("S2-")
            ),
            "gt_for_s1_match_ids_s3": sum(
                1 for v in gt_all.values() for mid in v if str(mid).startswith("S3-")
            ),
        }

    s1_ids = s1_ids_selected

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
    tfidf_cfg = blocking_cfg.get("char_tfidf", {})
    if smoke_kgrid is not None and int(smoke_kgrid) > 0:
        k_grid = [int(smoke_kgrid)]
        k_final = int(smoke_kgrid)
    else:
        k_grid = list(tfidf_cfg.get("k_grid", [10, 20, 50, 100]))
        k_final = int(tfidf_cfg.get("k_final", 50))
    max_cands = blocking_cfg.get("max_candidates_per_s1", 250)
    if smoke_max_features is not None and int(smoke_max_features) > 0:
        effective_max_features = int(smoke_max_features)
    else:
        effective_max_features = int(tfidf_cfg.get("max_features", 50000))
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
        min_df=int(tfidf_cfg.get("min_df", 2)),
        max_features=effective_max_features,
    )
    s1_records = records_by_id(s1)
    recall_by_k = {}
    tfidf_runtime_total = float(sum(tfidf_runtime.values())) if tfidf_runtime else 0.0
    for k in k_grid:
        mixed = union_exact_with_tfidf_k(exact_val, val_pool, val_meta, k if use_tfidf else 0)
        capped = cap_preferring_exact(mixed, val_meta, max_cands)
        per_source = evaluate_candidate_recall_per_source(
            capped, gt_val, n_s2=len(s2), n_s3=len(s3), s1_records=s1_records
        )
        per_source["tfidf_runtime_seconds"] = tfidf_runtime_total
        recall_by_k[str(k)] = per_source

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
        min_df=int(tfidf_cfg.get("min_df", 2)),
        max_features=effective_max_features,
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
        feature_names=list(X_train.columns) if len(X_train.columns) else list(FEATURE_NAMES),
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
        "loader_diagnostics": loader_diagnostics,
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
            "notes": f"limit_s1={limit_s1}; limit_s2={limit_s2}; limit_s3={limit_s3}; skip_tfidf={skip_tfidf}; smoke_kgrid={smoke_kgrid}; smoke_max_features={smoke_max_features}",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E0-E5 baseline pipeline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--loader-precheck-only", action="store_true", help="Only run S1/S2/S3 smoke loader (no blocking/model)")
    parser.add_argument("--limit-s1", type=int, default=None, help="Optional cap on S1 rows for a measured slice")
    parser.add_argument("--limit-s2", type=int, default=None, help="Diagnostic cap on S2 rows (seed=42 stream-sample, keep positives)")
    parser.add_argument("--limit-s3", type=int, default=None, help="Diagnostic cap on S3 rows (seed=42 stream-sample, keep positives)")
    parser.add_argument("--smoke-kgrid", type=int, default=None, help="Override k_grid/k_final with a single K (micro-smoke only)")
    parser.add_argument("--smoke-max-features", type=int, default=None, help="Override TF-IDF max_features (micro-smoke only)")
    parser.add_argument("--skip-tfidf", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = load_config(repo / args.config)
    if args.audit_only:
        path = run_audit(config, repo)
        print(f"Wrote {path}")
        return
    result = run_baseline(
        config,
        repo,
        limit_s1=args.limit_s1,
        limit_s2=args.limit_s2,
        limit_s3=args.limit_s3,
        skip_tfidf=args.skip_tfidf,
        smoke_kgrid=args.smoke_kgrid,
        smoke_max_features=args.smoke_max_features,
        loader_precheck_only=args.loader_precheck_only,
    )
    print(json.dumps({k: result[k] for k in ("split", "best_threshold", "val_recall", "runtime_seconds", "loader_precheck_only", "loader_diagnostics", "n_s1", "n_s2", "n_s3", "gt_for_s1_positive_count", "gt_for_s1_match_ids_s2", "gt_for_s1_match_ids_s3") if k in result}, default=str, indent=2))


if __name__ == "__main__":
    main()
