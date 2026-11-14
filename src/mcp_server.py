"""MCP server: lets an AI app (Claude Desktop, Claude Code and others) score flights with the saved forest.

Tools
  model_info      what the model reads, its threshold and what it was trained on
  list_flights    development flights in table.csv, with fault type and length
  score_flight    score one development flight from table.csv (the model has seen these flights, see below)
  score_rows      score new per-second feature rows, the same input as the /predict API

The saved forest (src/18_train_final.py) was trained on all development flights, so score_flight results are
in-sample and look better than on unseen data. The locked test flights are refused on purpose.
Needs: pip install -r requirements-mcp.txt, then src/18_train_final.py once to create models/forest.joblib.
Run:  python src/mcp_server.py   (stdio transport; an MCP client starts it, see docs/code_index.md)"""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import config
import evalkit

MODEL = Path(os.environ.get("ALFA_MODEL") or config.PROJECT / "models" / "forest.joblib")
server = MCPServer("alfa-fault-detection", instructions=(
    "Scores fixed-wing UAV flights from the ALFA dataset second by second with a random forest. "
    "score_flight results are in-sample because the model was trained on those flights."))


@lru_cache(maxsize=1)
def bundle():
    if not MODEL.exists():
        raise RuntimeError(f"model file not found: {MODEL} (run src/18_train_final.py first)")
    return joblib.load(MODEL)


@lru_cache(maxsize=1)
def table():
    return evalkit.load(cap=30)


def alarms(X, flight_ids):
    b = bundle()
    proba = b["model"].predict_proba(np.asarray(X, dtype=float))[:, 1]
    raw = proba >= b["threshold"]
    alarm = evalkit.persist(pd.DataFrame({"flight": flight_ids}), raw, b["k_persist"])
    return proba, alarm


@server.tool()
def model_info() -> dict:
    """What the model reads, its alarm rule and what it was trained on."""
    b = bundle()
    return {"trained_on": b["trained_on"], "threshold": b["threshold"],
            "k_persist": b["k_persist"], "features": b["features"]}


@server.tool()
def list_flights() -> list[dict]:
    """Development flights that score_flight accepts, with fault type and number of seconds."""
    dev = table()[table()["role"] != "test"]
    g = dev.groupby("flight").agg(fault_type=("fault_type", "first"), seconds=("t", "size"))
    return [{"flight": f, **r} for f, r in g.reset_index().set_index("flight").to_dict("index").items()]


@server.tool()
def score_flight(flight: str) -> dict:
    """Score one development flight second by second. Locked test flights are refused.
    The model was trained on all development flights, so this result is in-sample."""
    d = table()[table()["flight"] == flight]
    if d.empty:
        raise ToolError(f"unknown flight: {flight} (use list_flights)")
    if (d["role"] == "test").any():
        raise ToolError("this is a locked test flight; the test set is scored once, by 04_final_test.py")
    proba, alarm = alarms(d[bundle()["features"]], d["flight"].to_numpy())
    rel = d["t_rel_fault"].to_numpy()
    has_fault = not np.isnan(rel).all()
    before = alarm & (rel < 0) if has_fault else alarm
    after = alarm & (rel >= 0) if has_fault else np.zeros_like(alarm)
    first = round(float(rel[after][0]), 1) if after.any() else None
    return {"flight": flight, "fault_type": d["fault_type"].iloc[0], "seconds": len(d),
            "seen_in_training": True,
            "alarm_seconds_before_fault": int(before.sum()),
            "alarm_seconds_after_fault": int(after.sum()),
            "first_alarm_seconds_after_fault": first,
            "max_probability": round(float(proba.max()), 4)}


@server.tool()
def score_rows(rows: list[dict[str, float]]) -> dict:
    """Score new data: one flight as a list of per-second dicts with the 21 feature values, in time order."""
    feats = bundle()["features"]
    missing = sorted({f for r in rows for f in feats if f not in r})
    if missing:
        raise ToolError(f"missing features: {missing}")
    X = [[r[f] for f in feats] for r in rows]
    proba, alarm = alarms(X, np.zeros(len(rows)))
    return {"seconds": len(rows), "alarm_seconds": int(alarm.sum()),
            "first_alarm_second": int(np.argmax(alarm)) if alarm.any() else None,
            "probability": [round(float(p), 4) for p in proba], "alarm": [bool(a) for a in alarm]}


if __name__ == "__main__":
    server.run()
