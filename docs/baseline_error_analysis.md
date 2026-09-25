# Baseline error analysis

Validation-only. Counts are measured. Examples are IDs plus compact feature snapshots.
Raw business strings are not dumped.

Labeled errors: **15**

## noisy_name_case

- count: **8**
- fraction: **0.5333**
- examples:
  - `{'s1': 'S1-535605149', 'id': 'S2-751164711', 'score': 0.136101910682624, 'in_candidates': True, 'name_sim': 0.085, 'addr_sim': 0.577}`
  - `{'s1': 'S1-630223771', 'id': 'S3-587424087', 'score': 0.14375137205153776, 'in_candidates': True, 'name_sim': 0.457, 'addr_sim': 0.322}`
  - `{'s1': 'S1-630223771', 'id': 'S3-697578371', 'score': 0.0004461697832037913, 'in_candidates': True, 'name_sim': 0.103, 'addr_sim': 0.422}`
  - `{'s1': 'S1-630223771', 'id': 'S3-815538600', 'score': 0.00038690508995558443, 'in_candidates': True, 'name_sim': 0.103, 'addr_sim': 0.322}`
  - `{'s1': 'S1-306095185', 'id': 'S3-261575641', 'score': 0.00014681069682711756, 'in_candidates': True, 'name_sim': 0.488, 'addr_sim': 0.277}`
  - `{'s1': 'S1-86094936', 'id': 'S3-505894931', 'score': 0.00010031506081165666, 'in_candidates': True, 'name_sim': 0.071, 'addr_sim': 0.721}`
  - `{'s1': 'S1-305403960', 'id': 'S2-771841018', 'score': 0.2281035588009851, 'in_candidates': True, 'name_sim': 0.115, 'addr_sim': 0.65}`
  - `{'s1': 'S1-791867209', 'id': 'S3-805705209', 'score': 7.177597637826074e-05, 'in_candidates': True, 'name_sim': 0.235, 'addr_sim': 0.885}`

## candidate_retrieved_low_score

- count: **2**
- fraction: **0.1333**
- examples:
  - `{'s1': 'S1-883468618', 'id': 'S3-149911356', 'score': 0.011194843038571947, 'in_candidates': True, 'name_sim': 0.541, 'addr_sim': 0.537}`
  - `{'s1': 'S1-567349457', 'id': 'S3-70405454', 'score': 0.0029075709741333915, 'in_candidates': True, 'name_sim': 0.692, 'addr_sim': 0.595}`

## missing_field_case

- count: **2**
- fraction: **0.1333**
- examples:
  - `{'s1': 'S1-387694500', 'id': 'S3-77197957', 'score': 5.4827599773446994e-05, 'in_candidates': True, 'name_sim': 0.737, 'addr_sim': 0.0}`
  - `{'s1': 'S1-338840619', 'id': 'S2-711927056', 'score': 0.0009243351160724962, 'in_candidates': True, 'name_sim': 0.625, 'addr_sim': 0.0}`

## common_token_false_positive

- count: **1**
- fraction: **0.0667**
- examples:
  - `{'s1': 'S1-954595905', 'id': 'S3-982616638', 'score': 0.7728189739623099, 'name_sim': 0.824, 'addr_sim': 0.0}`

## missed_by_blocking

- count: **1**
- fraction: **0.0667**
- examples:
  - `{'s1': 'S1-387694500', 'id': 'S3-9126403', 'score': None, 'in_candidates': False}`

## noisy_address_case

- count: **1**
- fraction: **0.0667**
- examples:
  - `{'s1': 'S1-447103941', 'id': 'S3-556786353', 'score': 0.002624227788711678, 'in_candidates': True, 'name_sim': 0.531, 'addr_sim': 0.288}`

