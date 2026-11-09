import os
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
root = r"E:\ml-course\data\alfa\processed\processed"
split = pd.read_csv(r"E:\ml-course\data\split.csv")
dev = split[split["role"] != "test"]            # test set NOT used
CH = ["field.channels%d" % i for i in (1, 2, 3, 4, 5)]

rows = []
for _, r in dev.iterrows():
    folder = os.path.join(root, "carbonZ_" + r["flight"])
    files = os.listdir(folder)
    fs = [f for f in files if "failure_status" in f.lower() and f.endswith(".csv")]
    rc = [f for f in files if f.endswith("-mavros-rc-out.csv")]
    if not rc:
        print("skipped (no rc-out)", r["flight"]); continue
    d = pd.read_csv(os.path.join(folder, rc[0]))
    if fs:
        ref = min(pd.read_csv(os.path.join(folder, f), usecols=["%time"])["%time"].min() for f in fs)
    else:
        ref = d["%time"].min()                    # normal flight: time from the start
    d["s"] = np.floor((d["%time"] - ref) / 1e9).astype(int)
    g = d.groupby("s")[CH]
    sd, cnt = g.std(ddof=0), g.size()
    sd = sd.reindex(range(sd.index.min(), sd.index.max() + 1))
    cnt = cnt.reindex(sd.index)
    fr = (sd == 0).astype(float)
    fr[(cnt.fillna(0) < 2).to_numpy()] = np.nan   # seconds with too few samples
    fr3 = fr.rolling(3, min_periods=3).min()
    s0 = sd.index.min()
    if fs:
        base = fr3[(fr3.index >= s0 + 10) & (fr3.index < -20)]
        flt = fr3[(fr3.index >= 2) & (fr3.index < 30)]
    else:
        base = fr3[fr3.index >= 10]
        flt = None
    row = {"type": r["fault_type"]}
    for c in CH:
        k = c.replace("field.channels", "ch")
        row["b_" + k] = 100 * base[c].mean()
        row["f_" + k] = 100 * flt[c].mean() if flt is not None else np.nan
    rows.append(row)

R = pd.DataFrame(rows)
n = R.groupby("type").size()
for pre, title in (("b_", "BASELINE frozen % (before the fault / whole normal flight)"),
                   ("f_", "FAULT WINDOW frozen % (seconds 2-30)")):
    cols = [c for c in R.columns if c.startswith(pre)]
    for fn, label in (("mean", "mean over flights"), ("max" if pre == "b_" else "min", "worst case over flights")):
        T = R.groupby("type")[cols].agg(fn).round(0)
        T.columns = [c[2:] for c in T.columns]
        T.index = [f"{t} (n={n[t]})" for t in T.index]
        print(f"\n=== {title} - {label} ===")
        print(T.to_string())
