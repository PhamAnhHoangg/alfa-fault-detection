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
for c in ["alt_error", "aspd_error", "xtrack_error"]:
    SIGNALS[c] = ("mavros-nav_info-errors", col(c))
for c in ["altitude", "groundspeed", "climb", "throttle"]:
    SIGNALS["hud_" + c] = ("mavros-vfr_hud", col(c))
for a in "xyz":
    SIGNALS["gyro_" + a] = ("mavros-imu-data_raw", col("angular_velocity." + a))
    SIGNALS["accel_" + a] = ("mavros-imu-data_raw", col("linear_acceleration." + a))
    SIGNALS["vel_lin_" + a] = ("mavros-local_position-velocity", col("twist.linear." + a))
for i in [1, 2, 3, 4, 5]:
    SIGNALS[f"rc_out_{i}"] = ("mavros-rc-out", col(f"channels{i}"))

PRE = 10_000_000_000
POST = 15_000_000_000
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
        v = pd.to_numeric(fn(d), errors="coerce").to_numpy(dtype=float)
        t = d["%time"].to_numpy()
        pre = v[(t >= ft - PRE) & (t < ft)]
        post = v[(t >= ft) & (t <= ft + POST)]
        if len(pre) < 5 or len(post) < 5:
            continue
        sd_pre = np.nanstd(pre)
        shift = abs(np.nanmean(post) - np.nanmean(pre)) / (sd_pre + 1e-3)
        stdx = (np.nanstd(post) + 1e-3) / (sd_pre + 1e-3)
        rows.append(dict(signal=name, group=group, shift=shift, stdx=stdx))

tab = pd.DataFrame(rows)
a = tab.pivot_table(index="signal", columns="group", values="shift", aggfunc="median").add_prefix("shift_")
b = tab.pivot_table(index="signal", columns="group", values="stdx", aggfunc="median").add_prefix("stdx_")
res = a.join(b).sort_values("shift_engine", ascending=False)
print("shift = average moved by N normal wiggles | stdx = variability after / before (1.0 = same)")
print(res.round(1).to_string())

print("\n=== airspeed sensor dead (constant) per recording date ===")
by = {}
for f in flights:
    d = pd.read_csv(f / f"{f.name}-mavros-nav_info-airspeed.csv", usecols=["field.measured"])
    day = f.name.replace("carbonZ_", "")[:10]
    k = by.setdefault(day, [0, 0])
    k[1] += 1
    k[0] += int(d["field.measured"].std() == 0)
for day, (dead, n) in sorted(by.items()):
    print(day, f"dead {dead}/{n}")
