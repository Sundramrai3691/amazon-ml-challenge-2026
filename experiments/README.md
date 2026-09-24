# Experiment log

Log every meaningful experiment in `experiment_log.csv`.

Rules:

- Do not invent metric values.
- Leave a cell blank if it was not measured.
- Keep `validation_split_seed` stable unless the split change is the experiment.
- Change one major variable at a time when possible.
- Record blocking recall **before** matcher metrics.

Suggested experiment IDs follow the ladder in the root README: `E0`, `E1`, ...
