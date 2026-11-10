"""FastAPI demo: per-second fault alarm from the saved random forest.

Send ONE flight, seconds in time order, with the 21 feature values per second (5-second mean,
5-second std and 10-second change of each signal; see /health for the names). The reply holds the fault
probability, the raw alarm and the final alarm (k raw alarms in a row) for every second.

Limits: the API does not build the features from raw sensor logs (that is done by the table-building
script), and the model was trained on the public ALFA dev flights only."""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "forest.joblib"

app = FastAPI(title="ALFA fault alarm demo", version="0.1")


@lru_cache(maxsize=1)
def load_bundle():
    path = Path(os.environ.get("ALFA_MODEL") or DEFAULT_MODEL)
    if not path.exists():
        raise RuntimeError(f"model file not found: {path} (run src/18_train_final.py first)")
    return joblib.load(path)


def persistent_alarm(raw, k):
    """Alarm at second i only if the raw alarms at the last k seconds (including i) are all on.
    Same rule as evalkit.persist, for one flight."""
    s = pd.Series(np.asarray(raw, dtype=int))
    return s.rolling(k, min_periods=k).min().fillna(0).astype(bool).to_numpy()


class FlightRequest(BaseModel):
    rows: list[dict[str, float]] = Field(
        min_length=1, description="One flight, one dict of feature values per second, in time order.")
    threshold: float | None = Field(
        default=None, ge=0, le=1, description="Optional override of the model threshold (default 0.5).")


@app.get("/health")
def health():
    b = load_bundle()
    return {"status": "ok", "trained_on": b["trained_on"], "threshold": b["threshold"],
            "k_persist": b["k_persist"], "features": b["features"]}


@app.post("/predict")
def predict(req: FlightRequest):
    b = load_bundle()
    feats = b["features"]
    missing = sorted({f for r in req.rows for f in feats if f not in r})
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    X = np.array([[r[f] for f in feats] for r in req.rows], dtype=float)
    proba = b["model"].predict_proba(X)[:, 1]
    thr = req.threshold if req.threshold is not None else b["threshold"]
    raw = proba >= thr
    alarm = persistent_alarm(raw, b["k_persist"])
    return {
        "threshold": thr,
        "k_persist": b["k_persist"],
        "n_seconds": len(req.rows),
        "n_alarm_seconds": int(alarm.sum()),
        "first_alarm_second": int(np.argmax(alarm)) if alarm.any() else None,
        "seconds": [{"second": i, "proba": round(float(p), 4), "raw_alarm": bool(r), "alarm": bool(a)}
                    for i, (p, r, a) in enumerate(zip(proba, raw, alarm))],
    }
