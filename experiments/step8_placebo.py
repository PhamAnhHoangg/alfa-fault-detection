from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"E:\ml-course\data\alfa\processed\processed")
flights = sorted(p for p in ROOT.iterdir() if p.is_dir())

def col(name):
    return lambda d: d["field." + name]

def wrap(x):
    return (x + 180) % 360 - 180

def gap(d):
    return wrap(d["field.measured"] - d["field.commanded"])

SIGNALS = {}
for nm in ["roll", "pitch", "yaw"]:
    SIGNALS[nm + "_gap"] = (f"mavros-nav_info-{nm}", gap)
for c in ["alt_error", "xtrack_error"]:
    SIGNALS[c] = ("mavros-nav_info-errors", col(c))
for c in ["altitude", "groundspeed", "throttle"]:
    SIGNALS["hud_" + c] = ("mavros-vfr_hud", col(c))
for a in "xz":
    SIGNALS["vel_lin_" + a] = ("mavros-local_position-velocity", col("twist.linear." + a))
for i in [1, 2]:
    SIGNALS[f"rc_out_{i}"] = ("mavros-rc-out", col(f"channels{i}"))

PRE = 10_000_000_000
POST = 15_000_000_000
rows = []
for f in flights:
    fs = sorted(f.glob("*failure_status*.csv"))
    if not fs:
        continue
    ft0 = min(pd.read_csv(c, usecols=["%time"])["%time"].iloc[0] for c in fs)
    group = "engine" if "engine_failure" in f.name else "other"
    cache = {}
    for kind, off in (("real", 0), ("placebo", -30_000_000_000)):
        ft = ft0 + off
        for name, (topic, fn) in SIGNALS.items():
            if topic not in cache:
                cache[topic] = pd.read_csv(f / f"{f.name}-{topic}.csv")
            d = cache[topic]
            v = pd.to_numeric(fn(d), errors="coerce").to_numpy(dtype=float)
            t = d["%time"].to_numpy()
            pre = v[(t >= ft - PRE) & (t < ft)]
            post = v[(t >= ft) & (t <= ft + POST)]
            if len(pre) < 5 or len(post) < 5:
                continue
            sd_pre = np.nanstd(pre)
            shift = abs(np.nanmean(post) - np.nanmean(pre)) / (sd_pre + 1e-3)
            stdx = (np.nanstd(post) + 1e-3) / (sd_pre + 1e-3)
            rows.append(dict(signal=name, group=group, kind=kind, shift=shift, stdx=stdx))

tab = pd.DataFrame(rows)
for stat in ("shift", "stdx"):
    p = tab.pivot_table(index="signal", columns=["group", "kind"], values=stat, aggfunc="median")
    print(f"\n=== {stat} (median): real fault vs placebo 30 s earlier ===")
    print(p.round(1).to_string())
