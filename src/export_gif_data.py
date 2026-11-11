"""Exports small snippets of the public ALFA data, used to draw the explanation GIFs.

Run from the project folder:  python src/export_gif_data.py
Writes results/gif_data/*.csv (a few KB in total).
"""
import shutil

import numpy as np
import pandas as pd

import config
import evalkit

OUT = config.PROJECT / "results" / "gif_data"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. one flight, second by second
df = evalkit.load(cap=30)
dev = df[df["role"] != "test"].copy()
flight = dev[dev["fault_type"] == "engine"]["flight"].iloc[0]
folder = config.DATA / "alfa" / "processed" / "processed" / f"carbonZ_{flight}"
print("example flight:", flight)

imu_t = pd.read_csv(folder / f"{folder.name}-mavros-imu-data.csv", usecols=["%time"])["%time"].to_numpy()
start, end = imu_t[0], imu_t[-1]
n = int((end - start) // 1_000_000_000) + 1
fs_files = sorted(folder.glob("*failure_status*.csv"))
ft = min(pd.read_csv(c, usecols=["%time"])["%time"].iloc[0] for c in fs_files)
fault_sec = (ft - start) / 1e9

d = pd.read_csv(folder / f"{folder.name}-mavros-nav_info-errors.csv")
sec = ((d["%time"].to_numpy() - start) // 1_000_000_000).astype(int)
raw = pd.to_numeric(d["field.alt_error"], errors="coerce")
x = raw.groupby(sec).mean().reindex(range(n)).ffill(limit=5)
per_sec = pd.DataFrame({
    "t": range(n),
    "alt_error_mean": x.to_numpy(),
    "mean5": x.rolling(5, min_periods=3).mean().to_numpy(),
    "std5": x.rolling(5, min_periods=3).std().to_numpy(),
    "delta10": (x - x.shift(10)).to_numpy(),
})
per_sec["label"] = (per_sec["t"] + 1 > fault_sec).astype(int)
per_sec["t_rel_fault"] = per_sec["t"] - fault_sec
f0 = int(fault_sec)
snip = per_sec[(per_sec["t"] >= f0 - 12) & (per_sec["t"] <= f0 + 3)]
snip.to_csv(OUT / "per_second_example.csv", index=False)
pd.DataFrame({"flight": [flight], "fault_sec": [fault_sec]}).to_csv(OUT / "example_info.csv", index=False)

# raw readings that fall inside the second before the fault second
mask = sec == (f0 - 1)
rr = pd.DataFrame({"time_s": (d["%time"].to_numpy()[mask] - start) / 1e9,
                   "alt_error": raw.to_numpy()[mask]})
rr.to_csv(OUT / "raw_readings_one_second.csv", index=False)

# check against table.csv (the two must agree)
tab = pd.read_csv(config.DATA / "table.csv")
tab = tab[tab["flight"] == flight][["t", "alt_error__mean5"]]
chk = snip.merge(tab, on="t")
print("check against table.csv, max abs difference of mean5:",
      float((chk["mean5"] - chk["alt_error__mean5"]).abs().max()))

# ---------------------------------------------------------------- 2. split
split = pd.read_csv(config.DATA / "split.csv")
split.head(14).to_csv(OUT / "split_head.csv", index=False)
(split.groupby("role").agg(flights=("flight", "size"), sessions=("session", "nunique"))
 .reset_index().to_csv(OUT / "split_counts.csv", index=False))

# ---------------------------------------------------------------- 3. the fig7 engine cell
col = "alt_error__mean5"
g = dev[dev["fault_type"] == "engine"].copy()
g["x"] = g[col]
g["x_mean_of_flight"] = g.groupby("flight")[col].transform("mean")
g["x_tilde"] = g["x"] - g["x_mean_of_flight"]
g["y_mean_of_flight"] = g.groupby("flight")["label"].transform("mean")
g["y_tilde"] = g["label"] - g["y_mean_of_flight"]
g["xy"] = g["x_tilde"] * g["y_tilde"]
g["x2"] = g["x_tilde"] ** 2
g["y2"] = g["y_tilde"] ** 2
keep = ["flight", "t", "label", "x", "x_mean_of_flight", "x_tilde", "y_mean_of_flight",
        "y_tilde", "xy", "x2", "y2"]
g[keep].head(12).to_csv(OUT / "fig7_trace_head.csv", index=False)
sxy, sxx, syy = float(g["xy"].sum()), float(g["x2"].sum()), float(g["y2"].sum())
pd.DataFrame({"rows": [len(g)], "flights": [g["flight"].nunique()], "Sxy": [sxy], "Sxx": [sxx],
              "Syy": [syy], "r": [sxy / np.sqrt(sxx * syy)]}).to_csv(OUT / "fig7_trace_sums.csv", index=False)
print("fig7 engine cell: r =", round(sxy / np.sqrt(sxx * syy), 2), " (the figure shows 0.65)")

# ---------------------------------------------------------------- 4. model comparison
mc = config.PROJECT / "results" / "model_comparison.csv"
if mc.exists():
    shutil.copy(mc, OUT / "model_comparison.csv")

print("\nwritten to", OUT)
for p in sorted(OUT.glob("*.csv")):
    print("  ", p.name, p.stat().st_size, "bytes")
