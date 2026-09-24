# Methodology notes (evidence tracker)

Fill this file with **measured** observations during the competition.
Do not invent results. Empty sections are expected until experiments run.

## Executive summary notes

-

## Problem analysis

- S1 is a deduplicated reference set.
- S2 and S3 are noisy sources.
- Cardinality is zero/one/many matches per S1 entity.
- Country is open-set (train: US, India; test also: France).
- Primary metric: macro F0.5 per Source 1 entity (precision-heavy).

## Solution strategy

- Pipeline: data -> normalize -> block -> pair features -> pair model -> threshold -> submission.
- Blocking and matching stay separate.
- Local validation is an S1-entity split with a fixed seed.

## Candidate generation / blocking

-

## Matching model

-

## Features

-

## Threshold method

-

## Results and error analysis

-

## Conclusion

-

## Code artefact description

- Implementation lives under `src/`.
- Submission writers live in `src/submission.py`.
- Official/fast validator: `utils/validate_submission.py`.

## Additional results

-

## Open questions

-
