from src.split import entity_stratum, stratified_source1_split


def test_stratified_split_reproducible_and_disjoint() -> None:
    ids = [f"S1-{i}" for i in range(40)]
    strata = {}
    for i, s1 in enumerate(ids):
        country = "US" if i % 2 == 0 else "India"
        true = set() if i % 5 == 0 else {f"S2-{i}"}
        strata[s1] = entity_stratum(country, true)
    a_train, a_val, a_sum = stratified_source1_split(ids, strata, seed=42, validation_fraction=0.2)
    b_train, b_val, _ = stratified_source1_split(ids, strata, seed=42, validation_fraction=0.2)
    assert a_train == b_train and a_val == b_val
    assert set(a_train).isdisjoint(a_val)
    assert len(a_train) + len(a_val) == 40
    assert a_sum["method"] in {"stratified_entity", "random_entity"}
    assert a_sum["seed"] == 42
