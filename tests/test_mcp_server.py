"""Checks for src/mcp_server.py. Skipped when the mcp package, the saved model or the data are missing."""
import pytest

mcp = pytest.importorskip("mcp")
import mcp_server  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

if not mcp_server.MODEL.exists():
    pytest.skip("models/forest.joblib not found (run src/18_train_final.py)", allow_module_level=True)


def test_missing_features_are_refused():
    with pytest.raises(ToolError, match="missing features"):
        mcp_server.score_rows([{"alt_error__mean5": 1.0}])


def test_flight_and_rows_agree():
    try:
        d = mcp_server.table()
    except FileNotFoundError:
        pytest.skip("data not found")
    flight = d[d["role"] != "test"]["flight"].iloc[0]
    rows = d[d["flight"] == flight][mcp_server.bundle()["features"]].to_dict("records")
    a = mcp_server.score_flight(flight)
    b = mcp_server.score_rows(rows)
    assert a["alarm_seconds_before_fault"] + a["alarm_seconds_after_fault"] == b["alarm_seconds"]
    assert a["seen_in_training"] is True


def test_test_flights_are_refused():
    try:
        d = mcp_server.table()
    except FileNotFoundError:
        pytest.skip("data not found")
    with pytest.raises(ToolError, match="locked test flight"):
        mcp_server.score_flight(d[d["role"] == "test"]["flight"].iloc[0])
