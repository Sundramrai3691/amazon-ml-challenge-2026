from src.metrics import detailed_entity_metrics, entity_f05, macro_f05, precision_recall


def test_empty_empty_is_one() -> None:
    assert entity_f05(set(), set()) == 1.0
    assert precision_recall(set(), set()) == (1.0, 1.0)


def test_true_positive_empty_pred_is_zero() -> None:
    assert entity_f05({"S2-1"}, set()) == 0.0


def test_singleton_false_positive_is_zero() -> None:
    assert entity_f05(set(), {"S2-1"}) == 0.0


def test_exact_match() -> None:
    assert entity_f05({"S2-1", "S3-1"}, {"S3-1", "S2-1"}) == 1.0


def test_partial_overlap() -> None:
    score = entity_f05({"S2-1", "S2-2"}, {"S2-1"})
    # P=1, R=0.5 -> F0.5 = 1.25*1*0.5 / (0.25*1 + 0.5) = 0.625 / 0.75 = 5/6
    assert abs(score - (5.0 / 6.0)) < 1e-9


def test_multiple_matches_and_macro() -> None:
    gt = {
        "S1-1": set(),
        "S1-2": {"S2-1"},
        "S1-3": {"S2-2", "S3-1"},
    }
    pred = {
        "S1-1": set(),
        "S1-2": {"S2-1"},
        "S1-3": {"S2-2"},
    }
    # entity scores: 1.0, 1.0, F0.5(P=1,R=0.5)=5/6
    expected = (1.0 + 1.0 + 5.0 / 6.0) / 3.0
    assert abs(macro_f05(gt, pred) - expected) < 1e-9
    detail = detailed_entity_metrics(gt, pred)
    assert detail["n_entities"] == 3
    assert detail["n_singletons"] == 1
    assert abs(detail["macro_f0_5"] - expected) < 1e-9
