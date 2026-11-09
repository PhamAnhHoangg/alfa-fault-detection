"""Tests that need the real data (skipped when the data is not available)."""
import pytest
import evalkit


@pytest.fixture(scope="module")
def df():
    try:
        return evalkit.load(cap=30)
    except FileNotFoundError:
        pytest.skip("ALFA data not available")


def test_row_counts(df):
    assert len(df) == 4113
    assert len(df[df["role"] == "test"]) == 834
    assert len(df[df["role"] != "test"]) == 3279


def test_flight_counts(df):
    assert df[df["role"] == "test"]["flight"].nunique() == 9
    assert df[df["role"] != "test"]["flight"].nunique() == 37


def test_cap_removes_rows_after_30s(df):
    assert (df["t_rel_fault"].dropna() <= 30).all()


def test_no_flight_is_in_two_roles(df):
    roles_per_flight = df.groupby("flight")["role"].nunique()
    assert (roles_per_flight == 1).all()
