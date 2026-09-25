# Data audit

Measured locally from challenge TSV files. Values are not invented.
Country is treated as an open-set string. No ownership constraint is applied.

## Train source1

- rows: **2206821**
- columns: `['entity_id', 'business_name', 'business_address', 'country']`
- dtypes: `{'entity_id': 'str', 'business_name': 'str', 'business_address': 'str', 'country': 'str'}`
- missingness: `{'business_name': 0, 'business_address': 0, 'country': 0}`
- country distribution: `{'US': 1323633, 'India': 883188}`
- duplicate raw IDs: **0** (unique=2206821)
- duplicate normalized names: **180374**
- duplicate normalized addresses: **40635**

## Train source2

- rows: **5034616**
- columns: `['entity_id', 'business_name', 'business_address', 'country']`
- dtypes: `{'entity_id': 'str', 'business_name': 'str', 'business_address': 'str', 'country': 'str'}`
- missingness: `{'business_name': 1, 'business_address': 168967, 'country': 0}`
- country distribution: `{'US': 3016817, 'India': 2017799}`
- duplicate raw IDs: **0** (unique=5034616)
- duplicate normalized names: **391942**
- duplicate normalized addresses: **565454**

## Train source3

- rows: **5285603**
- columns: `['entity_id', 'business_name', 'business_address', 'country']`
- dtypes: `{'entity_id': 'str', 'business_name': 'str', 'business_address': 'str', 'country': 'str'}`
- missingness: `{'business_name': 0, 'business_address': 175916, 'country': 0}`
- country distribution: `{'US': 3170056, 'India': 2115547}`
- duplicate raw IDs: **0** (unique=5285603)
- duplicate normalized names: **397583**
- duplicate normalized addresses: **497241**

## Train ground truth

- rows / S1 entities: **2206821**
- match-count distribution: `{'0': 123247, '1': 119157, '2': 375212, '3+': 1589205}`
- singleton percentage: **5.5848%**
- S1 with both S2 and S3 matches: **1776047**
- S2 ownership: `{'n_distinct_s2_in_gt': 3693619, 'max_s1_owners': 1, 'owner_count_distribution': {1: 3693619}, 'n_s2_with_multiple_s1': 0}`
- S3 ownership: `{'n_distinct_s3_in_gt': 3944746, 'max_s1_owners': 1, 'owner_count_distribution': {1: 3944746}, 'n_s3_with_multiple_s1': 0}`
- note: Ownership is measured only. No uniqueness constraint is imposed.

## Test source1

- rows: **1732544**
- columns: `['entity_id', 'business_name', 'business_address', 'country']`
- dtypes: `{'entity_id': 'str', 'business_name': 'str', 'business_address': 'str', 'country': 'str'}`
- missingness: `{'business_name': 0, 'business_address': 0, 'country': 0}`
- country distribution: `{'India': 809986, 'US': 663106, 'France': 259452}`
- duplicate raw IDs: **0** (unique=1732544)
- duplicate normalized names: **131662**
- duplicate normalized addresses: **34327**

## Test source2 / source3 (row counts)

- test_source2 rows: **4887273**
- test_source3 rows: **5082316**
- Full missingness/country diagnostics for test S2/S3 skipped by default (~1GB). Row counts only. Pass include_test_s2_s3=True for a full scan.

