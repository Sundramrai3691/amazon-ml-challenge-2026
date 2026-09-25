# Baseline error analysis

Validation-only. Counts are measured. Examples are IDs plus compact feature snapshots.
Raw business strings are not dumped.

Labeled errors: **17**

## singleton_false_positive

- count: **16**
- fraction: **0.9412**
- examples:
  - `{'s1': 'S1-232999426', 'id': 'S2-170817805', 'score': 0.9999999999556377, 'name_sim': 0.731, 'addr_sim': 0.36}`
  - `{'s1': 'S1-913042783', 'id': 'S3-564669171', 'score': 0.9999999999584055, 'name_sim': 0.308, 'addr_sim': 0.529}`
  - `{'s1': 'S1-765734254', 'id': 'S2-681560600', 'score': 0.9999999999333153, 'name_sim': 0.667, 'addr_sim': 0.324}`
  - `{'s1': 'S1-765734254', 'id': 'S2-97470579', 'score': 0.9999999999999996, 'name_sim': 0.717, 'addr_sim': 0.39}`
  - `{'s1': 'S1-765734254', 'id': 'S3-793221535', 'score': 0.9991059284671499, 'name_sim': 0.324, 'addr_sim': 0.674}`
  - `{'s1': 'S1-765734254', 'id': 'S3-968537137', 'score': 0.9999665265392299, 'name_sim': 0.345, 'addr_sim': 0.651}`
  - `{'s1': 'S1-493059116', 'id': 'S3-517113780', 'score': 1.0, 'name_sim': 0.448, 'addr_sim': 0.444}`
  - `{'s1': 'S1-208550511', 'id': 'S2-248479986', 'score': 1.0, 'name_sim': 0.377, 'addr_sim': 0.464}`

## noisy_name_case

- count: **1**
- fraction: **0.0588**
- examples:
  - `{'s1': 'S1-535605149', 'id': 'S2-751164711', 'score': 0.0, 'in_candidates': True, 'name_sim': 0.085, 'addr_sim': 0.577}`

