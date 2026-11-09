"""Tests for the small helper functions in src/evalkit.py."""
import pandas as pd
import evalkit


def test_persist_needs_k_alarms_in_a_row():
    d = pd.DataFrame({"flight": ["a"] * 6})
    pred = [0, 1, 1, 1, 0, 1]
    out = evalkit.persist(d, pred, 3)
    # only the 4th second has 3 alarms in a row (seconds 2, 3, 4)
    assert out.tolist() == [False, False, False, True, False, False]


def test_persist_does_not_cross_flights():
    d = pd.DataFrame({"flight": ["a", "a", "b", "b"]})
    out = evalkit.persist(d, [1, 1, 1, 1], 3)
    # each flight has only 2 rows, so 3 in a row is impossible
    assert out.tolist() == [False, False, False, False]


def test_persist_k1_changes_nothing():
    d = pd.DataFrame({"flight": ["a"] * 4})
    out = evalkit.persist(d, [0, 1, 0, 1], 1)
    assert out.tolist() == [False, True, False, True]


def test_feature_lists_splits_sensor_and_cause():
    df = pd.DataFrame(columns=["flight", "t", "x__mean5", "cause_a__mean5", "rc_out_1__mean5"])
    sym, feats = evalkit.feature_lists(df)
    assert sym == ["x__mean5", "rc_out_1__mean5"]
    assert feats == ["x__mean5", "cause_a__mean5", "rc_out_1__mean5"]
    # NOTE: sym still contains rc_out_1 (we drop rc_out later, in the scripts)
