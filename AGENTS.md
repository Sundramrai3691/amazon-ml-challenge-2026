# AGENTS.md

Instructions for Cursor, Codex, Antigravity, and human teammates working on the Amazon ML Challenge 2026 **Business Entity Resolution** repository.

Agents are coding assistants. They are **not** business-data lookup agents.

---

## Problem

Business Entity Resolution.

Three independent sources:

- **Source 1 (S1):** deduplicated reference entities (`S1-...`)
- **Source 2 (S2):** noisy entity records (`S2-...`)
- **Source 3 (S3):** noisy entity records (`S3-...`)

Fields: `entity_id`, `business_name`, `business_address`, `country`.

For every Source 1 entity, predict **all** matching Source 2 and/or Source 3 records.

An S1 entity may have:

- zero matches
- one match
- many matches

Never assume one-to-one matching. Never force top-1.

---

## Evaluation

Primary local metric: **macro F0.5 per Source 1 entity**.

```
F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
```

Computed per S1 entity, then macro-averaged. Singletons are included.

| true matches | predicted matches | entity score |
|---|---|---|
| `{}` | `{}` | `1.0` |
| non-empty | `{}` | `0.0` |
| `{}` | non-empty | `0.0` |

The competition is **precision-heavy**. False positives on singletons are especially damaging. Empty predictions are legitimate.

Do **not** replace this with pairwise micro F0.5.

The public leaderboard is only a subset of full test evaluation. Do **not** use leaderboard feedback as the main training/validation loop. Build local validation from training data at the **Source 1 entity** level. Do not use a random row-level pair split as the primary protocol.

The validation split must be reproducible with a fixed seed. Do not silently change it between experiments.

---

## Pipeline

```
data -> normalization -> blocking -> pair features -> pair model -> threshold -> submission
```

Blocking and matching remain strictly separate.

Candidate generation determines the upper bound of recall. The matcher only sees candidate pairs. Every final matched ID **must** appear in `candidate_pairs.tsv`.

Experimental ladder (do not skip to the end):

- E0 data sanity + metric
- E1 exact normalized matching
- E2 simple blocking
- E3 string similarity features
- E4 pair classifier
- E5 threshold optimization
- E6 hard-negative mining
- E7 advanced blocking
- E8 embeddings / advanced models **only if justified**

---

## Data rules

- All challenge files are TAB-separated TSV.
- Every loader must use `sep="\t"`. Never silently fall back to comma-separated parsing.
- Never modify raw challenge data.
- Never commit raw challenge data to Git (`dataset/` is gitignored).
- Treat **country as open-set**. Training has US and India; test also has France. Never hard-code only US/India. Never filter away France. Never assume a closed country universe.
- Prefer schema summaries, sampled examples, and aggregates. Do not paste massive raw datasets into prompts.

---

## Fair play (strictly prohibited)

- business lookup on the web
- company/business registry lookup
- commercial entity resolution APIs
- geocoding APIs
- external business databases
- external business identity resolution
- internet-based entity augmentation
- sending challenge records to internet APIs for resolution

Library and programming documentation is allowed for implementation.

---

## Model constraints

Any final model must satisfy:

- MIT or Apache 2.0 compatible license
- <= 8 billion parameters

Before introducing any pretrained model, verify license, parameter count, reproducibility, and challenge-rule compliance. Do not download pretrained models during initialization-style work.

---

## Scale

The test set is approximately 1.7 million entities across sources.

Never:

- full S1 × S2 or S1 × S3 Cartesian products
- dense similarity matrices over entire datasets
- unrestricted pair generation
- unnecessary duplicated DataFrames

Prefer indexed lookup, normalized keys, batch/chunked processing, compact representations, streaming output, deterministic partitioning, cached indexes.

Do not automatically run `utils/validate_submission.py --check-ids` on the full test set (can require several GB of RAM). Default development validation is **fast mode**.

---

## Submission

`output/matching_results.tsv`:

- exactly one row per test S1
- header: `source1_entity_id`, `matched_entity_ids`
- TAB-separated
- matched IDs only S2/S3
- no duplicate matched IDs
- empty list allowed

`output/candidate_pairs.tsv`:

- exactly one row per test S1
- header: `source1_entity_id`, `candidate_entity_ids`
- candidate list is the exact set passed into the final matcher
- final matches must be a subset of candidates
- no duplicate candidate IDs

Final package shape (later):

```
<team_name>_submission.zip
  output/matching_results.tsv
  output/candidate_pairs.tsv
  code/business_entity_resolution/src/
  code/business_entity_resolution/README.md
  code/business_entity_resolution/requirements.txt
  Documentation_template.md
```

---

## Git

- Never work directly on `main`.
- Use feature branches (`feat/validation`, `feat/blocking`, `feat/pair-model`, `feat/submission`, ...).
- One task = one branch. Small meaningful commits. PR before merge.
- No force pushes. Do not overwrite teammate work.
- Pull/rebase before starting new work from `main`.
- Do not commit unless a human explicitly asks.

---

## Agent behavior

Before major changes:

1. Inspect the existing implementation.
2. State the hypothesis.
3. State the experiment.
4. Implement the smallest change.

After changes:

1. Run tests.
2. Report failures honestly.
3. Report measured results only.
4. List changed files.

Do **not**:

- fabricate scores
- claim leaderboard improvement without evidence
- change multiple major experimental variables without documenting them
- silently change the validation split
- add unrelated refactors
- introduce large models without evidence they address a measured bottleneck
- create AWS resources or upload submissions unless explicitly asked
