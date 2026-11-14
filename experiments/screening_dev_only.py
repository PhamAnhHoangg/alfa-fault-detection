"""Redo of the step6 to step8 signal screening, once on all fault flights and once on dev flights only (ROADMAP item 32).
Lift = median after-fault change minus the same at a placebo time 30 s earlier, best of shift and stdx for engine and other faults.
Run from the repo root with ALFA_DATA set; needs split.csv from 02_make_split.py."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(os.environ["ALFA_DATA"])
ROOT = DATA / "alfa" / "processed" / "processed"
split = pd.read_csv(DATA / "split.csv")
role = dict(zip(split["flight"], split["role"]))
CHOSEN = {"alt_error", "xtrack_error", "hud_altitude", "vel_lin_z", "roll_gap", "pitch_gap", "yaw_gap"}


def col(n):
    return lambda d: d["field." + n]


def gap(d):
    return d["field.measured"] - d["field.commanded"]


S = {}
for nm in ["roll", "pitch", "yaw", "airspeed"]:
    S[nm + "_gap"] = (f"mavros-nav_info-{nm}", gap)
    S[nm + "_measured"] = (f"mavros-nav_info-{nm}", col("measured"))
for c in ["alt_error", "aspd_error", "xtrack_error"]:
    S[c] = ("mavros-nav_info-errors", col(c))
for c in ["airspeed", "groundspeed", "climb", "altitude"]:
    S["hud_" + c] = ("mavros-vfr_hud", col(c))
for c in ["voltage", "current"]:
    S["batt_" + c] = ("mavros-battery", col(c))
for a in "xyz":
    S["gyro_" + a] = ("mavros-imu-data_raw", col("angular_velocity." + a))
    S["accel_" + a] = ("mavros-imu-data_raw", col("linear_acceleration." + a))
    S["vel_lin_" + a] = ("mavros-local_position-velocity", col("twist.linear." + a))

PRE, POST = 10e9, 15e9
rows = []
for f in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    fs = sorted(f.glob("*failure_status*.csv"))
    if not fs:
        continue
    ft0 = min(pd.read_csv(c, usecols=["%time"])["%time"].iloc[0] for c in fs)
    grp = "engine" if "engine_failure" in f.name else "other"
    r = role.get(f.name.replace("carbonZ_", ""))
    cache = {}
    for kind, off in (("real", 0), ("placebo", -30e9)):
        ft = ft0 + off
        for name, (topic, fn) in S.items():
            if topic not in cache:
                cache[topic] = pd.read_csv(f / f"{f.name}-{topic}.csv")
            d = cache[topic]
            try:
                v = pd.to_numeric(fn(d), errors="coerce").to_numpy(dtype=float)
            except KeyError:
                continue
            t = d["%time"].to_numpy()
            pre = v[(t >= ft - PRE) & (t < ft)]
            post = v[(t >= ft) & (t <= ft + POST)]
            if len(pre) < 5 or len(post) < 5:
                continue
            sd = np.nanstd(pre)
            rows.append(dict(signal=name, group=grp, kind=kind, role=r,
                             shift=abs(np.nanmean(post) - np.nanmean(pre)) / (sd + 1e-3),
                             stdx=(np.nanstd(post) + 1e-3) / (sd + 1e-3)))
tab = pd.DataFrame(rows)


def lift(t):
    m = t.pivot_table(index="signal", columns=["group", "kind"], values=["shift", "stdx"], aggfunc="median")
    return pd.concat([m[(s, g, "real")] - m[(s, g, "placebo")]
                      for s in ("shift", "stdx") for g in ("engine", "other")], axis=1).max(axis=1)


A, D = lift(tab), lift(tab[tab["role"] != "test"])
out = pd.DataFrame({"all: lift": A, "rank all": A.rank(ascending=False).astype(int),
                    "dev: lift": D, "rank dev": D.rank(ascending=False).astype(int)})
out["chosen"] = ["yes" if s in CHOSEN else "" for s in out.index]
print(out.sort_values("rank dev").round(1).to_string())
print("\nchosen ranks, all flights:", sorted(out[out.chosen == "yes"]["rank all"]))
print("chosen ranks, dev only:   ", sorted(out[out.chosen == "yes"]["rank dev"]))
