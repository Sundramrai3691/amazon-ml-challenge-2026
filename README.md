# Business Entity Resolution — team repository

Team implementation for **Amazon ML Challenge 2026: Business Entity Resolution**.

This repository is an engineering foundation for a 72-hour competition. It does **not** claim a leaderboard score.

## 1. Purpose

Provide a clean, testable, reproducible pipeline:

```
data → normalization → blocking → pair features → pair classifier → threshold → submission → validator
```

so teammates and coding agents can iterate independently.

## 2. Challenge summary

Match noisy Source 2 / Source 3 business records to **every** Source 1 reference entity.

- IDs: `S1-...`, `S2-...`, `S3-...`
- Fields: `entity_id`, `business_name`, `business_address`, `country`
- Cardinality: zero, one, or many matches per S1 (never assume 1:1)
- Country is **open-set** (train: US, India; test also: France)
- Primary metric: **macro F0.5 per Source 1 entity** (precision-heavy)
- Public leaderboard is a subset; local S1-entity validation is mandatory

Official challenge files (problem statement, validator, methodology template) are **authoritative** when present. Team code must not reinterpret those rules.

At initialization, the official validator and official problem README were **not** in this folder. `utils/validate_submission.py` is a stdlib-only validator matching the documented CLI and checks so development can proceed. **Replace it with the official file without rewriting logic** when the organizers provide it. Root `Documentation_template.md` is a heading-only placeholder for the same reason.

## 3. Repository structure

```
.
├── AGENTS.md
├── README.md                 # this file (team README)
├── Documentation_template.md
├── requirements.txt
├── configs/baseline.yaml
├── src/                      # production logic
├── tests/                    # synthetic unit tests only
├── notebooks/                # EDA only; import src/
├── experiments/              # experiment_log.csv
├── docs/methodology_notes.md
├── output/                   # generated TSV (gitignored except .gitkeep)
├── dataset/                  # local data placement (gitignored)
└── utils/validate_submission.py
```

**Competition-provided (when available):** problem statement, `utils/validate_submission.py`, `Documentation_template.md`, dataset TSV files.

**Team-owned:** `src/`, `tests/`, `configs/`, `AGENTS.md`, this README, experiment tracking.

## 4. Local environment setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Do not add torch / transformers / LightGBM / XGBoost until an experiment justifies them (license + install + parameter constraints).

## 5. Data placement

Place organizer TSV files here (not committed):

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

All files are TAB-separated. Loaders always use `sep="\t"`.

Override paths with `BER_TRAIN_DIR`, `BER_TEST_DIR`, `BER_OUTPUT_DIR` or `configs/baseline.yaml`.

## 6. Data loading

```python
from pathlib import Path
from src.load_data import load_training_data, load_test_data, parse_matched_ids

train = load_training_data(Path("dataset/train"))
test = load_test_data(Path("dataset/test"))
```

Loaders validate schema, preserve strings, and support column selection / chunked reads. Do not reload full test tables in every module; pass DataFrames or indexes.

## 7. Validation split

Split **Source 1 entity IDs** with the seed in `configs/baseline.yaml` (`seed: 42`, `validation_fraction: 0.2`).

```
train S1 → block → pair labels/features → fit pair model
val S1   → block (no Cartesian product) → score → threshold → macro F0.5
```

Do not use a random pair-row split as the primary protocol. Do not silently change the seed.

## 8. Baseline pipeline

Interfaces exist for the experimental ladder E0–E5. The initial matcher is a small sklearn classifier (logistic regression by default). No embeddings, transformers, or GNNs in this foundation.

## 9. Candidate generation

`src/blocking.py` — inverted-index blocking, not S1×S2 products.

Initial methods: exact normalized name; name without legal suffix; exact normalized address; combined name+address key.

Always measure **before** matching:

- `candidate_recall`
- `average_candidates_per_s1`
- `median_candidates_per_s1`
- `max_candidates_per_s1`
- `reduction_ratio`

If a true match is never a candidate, the matcher cannot recover it.

## 10. Pair features

`src/features.py` — pair-level feature dict/matrix. Baseline: name/address equality, lengths, token overlap, country equality, candidate source, blocking method, rank. RapidFuzz and TF-IDF cosine are extension points, not required at init.

## 11. Training

`src/train.py` — build labels from candidates ∩ ground truth (hard negatives from blocking only), fit a config-driven sklearn model, save/load with joblib.

## 12. Prediction

`src/predict.py` — `predict_proba` then a **configured** threshold (default `0.7`, not assumed optimal). Supports zero, one, and many matches. Includes a threshold-sweep helper for validation S1 entities.

## 13. Submission generation

```python
from src.submission import write_submission

write_submission(
    s1_ids=["S1-1", "S1-2"],
    candidates={"S1-1": ["S2-1"], "S1-2": []},
    matches={"S1-1": ["S2-1"], "S1-2": []},
    output_dir="output",
)
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` with `sep="\t"` and `index=False`.

## 14. Official validator

**Fast (default development):**

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Diagnostic ID check (manual / memory-aware only):**

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

Do **not** run `--check-ids` on the full ~1.7M test set by default.

## 15. Experiment logging

Append rows to `experiments/experiment_log.csv`. Leave unmeasured metrics blank. Never fabricate values.

## 16. Git workflow

Never develop on `main`. Feature branches, small commits, PR, no force-push. Humans request commits; agents do not commit unless asked.

## 17. Reproducibility

Fixed seeds in config. Deterministic blocking (sorted IDs). Unit tests use tiny synthetic TSV only.

## 18. Fair-play restrictions

No web business lookup, registries, geocoding, commercial ER APIs, or external identity data. See `AGENTS.md`.

## 19. Final packaging

Assemble later:

```
<team_name>_submission.zip
  output/matching_results.tsv
  output/candidate_pairs.tsv
  code/business_entity_resolution/{src,README.md,requirements.txt}
  Documentation_template.md
```

Fill methodology from `docs/methodology_notes.md` using measured evidence only.
