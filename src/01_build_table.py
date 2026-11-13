import config
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(str(config.DATA / r"alfa\processed\processed"))
OUT = Path(str(config.DATA / r"table.csv"))
flights = sorted(p for p in ROOT.iterdir() if p.is_dir())

def wrap(x):
    return (x + 180) % 360 - 180

def col(name):
    return lambda d: d["field." + name]

def gap(d):
    return wrap(d["field.measured"] - d["field.commanded"])

SIGNALS = {
    "alt_error": ("mavros-nav_info-errors", col("alt_error")),
    "xtrack_error": ("mavros-nav_info-errors", col("xtrack_error")),
    "hud_altitude": ("mavros-vfr_hud", col("altitude")),
    "vel_lin_z": ("mavros-local_position-velocity", col("twist.linear.z")),
    "rc_out_1": ("mavros-rc-out", col("channels1")),
    "roll_gap": ("mavros-nav_info-roll", gap),
    "pitch_gap": ("mavros-nav_info-pitch", gap),
    "yaw_gap": ("mavros-nav_info-yaw", gap),
    "cause_rc_out_2": ("mavros-rc-out", col("channels2")),
    "cause_hud_throttle": ("mavros-vfr_hud", col("throttle")),
}

def fault_type(name):
    if "no_failure" in name:
        return "none"
    for k in ("engine", "rudder", "elevator", "aileron"):
        if k in name:
            return k
    return "unknown"

frames = []
for f in flights:
    if "no_ground_truth" in f.name:
        print("skip (no labels):", f.name)
        continue
    imu_t = pd.read_csv(f / f"{f.name}-mavros-imu-data.csv", usecols=["%time"])["%time"].to_numpy()
    start, end = imu_t[0], imu_t[-1]
    n = int((end - start) // 1_000_000_000) + 1
    fs = sorted(f.glob("*failure_status*.csv"))
    fault_sec = None
    if fs:
        ft = min(pd.read_csv(c, usecols=["%time"])["%time"].iloc[0] for c in fs)
        fault_sec = (ft - start) / 1e9
    cache = {}
    parts = []
    for name, (topic, fn) in SIGNALS.items():
        if topic not in cache:
            cache[topic] = pd.read_csv(f / f"{f.name}-{topic}.csv")
        d = cache[topic]
        s = pd.Series(pd.to_numeric(fn(d), errors="coerce").to_numpy(dtype=float))
        sec = ((d["%time"].to_numpy() - start) // 1_000_000_000).astype(int)
        x = s.groupby(sec).mean().reindex(range(n)).ffill(limit=5)
        parts.append(pd.DataFrame({
            f"{name}__mean5": x.rolling(5, min_periods=3).mean(),
            f"{name}__std5": x.rolling(5, min_periods=3).std(),
            f"{name}__delta10": x - x.shift(10)}))
    df = pd.concat(parts, axis=1)
    df.index.name = "t"
    df = df.reset_index()
    m = re.search(r"carbonZ_(\d{4}-\d\d-\d\d-\d\d-\d\d-\d\d)", f.name)
    df.insert(0, "flight", f.name.replace("carbonZ_", ""))
    df.insert(1, "session", m.group(1))
    df.insert(2, "day", m.group(1)[:10])
    df.insert(3, "fault_type", fault_type(f.name))
    if fault_sec is None:
        df["label"] = 0
        df["t_rel_fault"] = np.nan
    else:
        df["label"] = (df["t"] + 1 > fault_sec).astype(int)
        df["t_rel_fault"] = df["t"] - fault_sec
    frames.append(df)

table = pd.concat(frames, ignore_index=True)
table = table[table["t"] >= 10]
feat = [c for c in table.columns if "__" in c]
before = len(table)
table = table.dropna(subset=feat)
print("rows dropped because of missing values:", before - len(table))
table.to_csv(OUT, index=False)

print("\nsaved:", OUT)
print("table shape (rows, columns):", table.shape, " features:", len(feat))
print("flights:", table["flight"].nunique(), " sessions:", table["session"].nunique())
print("\nlabel counts (0 = normal, 1 = fault):")
print(table["label"].value_counts().to_string())
print("\nrows per fault type (and how many are label 1):")
print(table.groupby("fault_type")["label"].agg(rows="count", fault_rows="sum").to_string())
print("\nsanity: label-1 rows inside normal flights (must be 0):",
      int(table[table.fault_type == "none"]["label"].sum()))
print("rows per flight  min/max:",
      table.groupby("flight").size().min(), table.groupby("flight").size().max())

