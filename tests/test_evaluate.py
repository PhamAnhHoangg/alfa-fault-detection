"""Tests for evalkit.evaluate on tiny flights where the right answer is known by hand."""
import numpy as np
import pandas as pd
import pytest
import evalkit

T = [-2, -1, 0, 1, 2, 3]
LABEL = [0, 0, 1, 1, 1, 1]


def one_flight(label=LABEL, ftype="engine"):
    return pd.DataFrame({"flight": "a", "fault_type": ftype,
                         "t_rel_fault": T, "label": label})


def test_clean_detection():
    res = evalkit.evaluate(one_flight(), [0, 0, 0, 1, 1, 1])
    assert res["prec"] == pytest.approx(1.0)
    assert res["rec_eng"] == pytest.approx(0.75)
    assert res["f1"] == pytest.approx(6 / 7)
    assert res["fa_s_per_min"] == pytest.approx(0.0)
    assert res["det_eng"] == "1/1"
    assert res["pre_eng"] == "0/1"
    assert res["med_delay_s"] == pytest.approx(1.0)
    assert np.isnan(res["rec_oth"])
    assert res["det_oth"] == "0/0"


def test_false_alarm_before_fault():
    res = evalkit.evaluate(one_flight(), [1, 0, 0, 1, 1, 1])
    assert res["prec"] == pytest.approx(0.75)
    assert res["f1"] == pytest.approx(0.75)
    assert res["fa_s_per_min"] == pytest.approx(30.0)
    assert res["pre_eng"] == "1/1"
    assert res["det_eng"] == "1/1"


def test_persistence_delays_and_cuts_recall():
    res = evalkit.evaluate(one_flight(), [0, 0, 1, 1, 1, 1], k_persist=3)
    assert res["prec"] == pytest.approx(1.0)
    assert res["rec_eng"] == pytest.approx(0.5)
    assert res["f1"] == pytest.approx(2 / 3)
    assert res["med_delay_s"] == pytest.approx(2.0)


def test_engine_and_other_types_are_separated():
    a = pd.DataFrame({"flight": "a", "fault_type": "engine",
                      "t_rel_fault": [-1, 0, 1], "label": [0, 1, 1]})
    b = pd.DataFrame({"flight": "b", "fault_type": "aileron",
                      "t_rel_fault": [-1, 0, 1], "label": [0, 1, 1]})
    d = pd.concat([a, b], ignore_index=True)
    res = evalkit.evaluate(d, [0, 1, 1, 0, 0, 0])
    assert res["rec_eng"] == pytest.approx(1.0)
    assert res["rec_oth"] == pytest.approx(0.0)
    assert res["det_eng"] == "1/1"
    assert res["det_oth"] == "0/1"
    assert res["med_delay_s"] == pytest.approx(0.0)
