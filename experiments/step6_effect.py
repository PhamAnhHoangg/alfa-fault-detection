from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"E:\ml-course\data\alfa\processed\processed")
flights = sorted(p for p in ROOT.iterdir() if p.is_dir())

def col(name):
    return lambda d: d["field." + name]

def gap(d):
    return d["field.measured"] - d["field.commanded"]

SIGNALS = {}
for nm in ["roll", "pitch", "yaw", "airspeed"]:
    SIGNALS[nm + "_gap"] = (f"mavros-nav_info-{nm}", gap)
    SIGNALS[nm + "_measured"] = (f"mavros-nav_info-{nm}", col("measured"))
for c in ["alt_error", "aspd_error", "xtrack_error"]:
    SIGNALS[c] = ("mavros-nav_info-errors", col(c))
for c in ["airspeed", "groundspeed", "throttle", "climb", "altitude"]:
    SIGNALS["hud_" + c] = ("mavros-vfr_hud", col(c))
for c in ["voltage", "current"]:
    SIGNALS["batt_" + c] = ("mavros-battery", col(c))
for a in "xyz":
    SIGNALS["gyro_" + a] = ("mavros-imu-data_raw", col("angular_velocity." + a))
    SIGNALS["accel_" + a] = ("mavros-imu-data_raw", col("linear_acceleration." + a))
    SIGNALS["vel_lin_" + a] = ("mavros-local_position-velocity", col("twist.linear." + a))
for i in range(8):
    SIGNALS[f"rc_out_{i}"] = ("mavros-rc-out", col(f"channels{i}"))

rows = []
for f in flights:
    fs = sorted(f.glob("*failure_status*.csv"))
    if not fs:
        continue
    ft = min(pd.read_csv(c, usecols=["%time"])["%time"].iloc[0] for c in fs)
    group = "engine" if "engine_failure" in f.name else "other"
    cache = {}
    for name, (topic, fn) in SIGNALS.items():
        if topic not in cache:
            cache[topic] = pd.read_csv(f / f"{f.name}-{topic}.csv")
        d = cache[topic]
        try:
            v = pd.to_numeric(fn(d), errors="coerce").to_numpy(dtype=float)
        except KeyError:
            continue
        t = d["%time"].to_numpy()
        pre = v[(t >= ft - 10_000_000_000) & (t < ft)]
        post = v[(t >= ft) & (t <= ft + 7_000_000_000)]
        if len(pre) < 3 or len(post) < 3:
            continue
        dead = bool(np.nanstd(v) == 0)
        shift = abs(np.nanmean(post) - np.nanmean(pre)) / (np.nanstd(pre) + 1e-3)
        rows.append(dict(signal=name, group=group, shift=shift, dead=dead))

tab = pd.DataFrame(rows)
res = tab.pivot_table(index="signal", columns="group", values="shift", aggfunc="median")
res["dead_flights"] = tab.groupby("signal")["dead"].sum()
res["flights"] = tab.groupby("signal")["dead"].count()
res = res.sort_values("engine", ascending=False)
print("shift = how many normal wiggles the average moves after the fault (median over flights)")
print(res.round(1).to_string())
