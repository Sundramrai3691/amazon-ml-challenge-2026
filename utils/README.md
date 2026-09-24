# Submission validator

## Status

The official Amazon ML Challenge 2026 `validate_submission.py` was **not in the repository** at initialization.

`validate_submission.py` in this folder is a **stdlib-only** validator that implements the documented CLI and checks so local development can proceed:

- `matching_results.tsv` required
- `candidate_pairs.tsv` when provided
- required Source 1 rows from `test_source1.tsv`
- duplicate rows / duplicate IDs in lists
- S1 self-matches
- invalid prefixes (matches/candidates must be S2/S3)
- optional S2/S3 ID existence (`--check-ids`)
- warning if a final match is missing from the candidate list

When the official file is provided, **replace this file as-is**. Do not rewrite official validation logic.

## Fast mode (default)

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Diagnostic ID check

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

`--check-ids` may load all S2/S3 IDs and can require several GB of memory on the full test set. Do not run it automatically in development.
