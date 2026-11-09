import os
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
root = r"E:\ml-course\data\alfa\processed\processed"
split = pd.read_csv(r"E:\ml-course\data\split.csv")
dev = split[(split["role"] != "test") & (~split["fault_type"].isin(["engine", "none"]))]   # test set NOT used
flights = sorted(dev["flight"])
ftype = dict(zip(dev["flight"], dev["fault_type"]))
CH = ["field.channels%d" % i for i in (1, 2, 3, 4, 5, 7)]

real, plc, bsd = {}, {}, {}
for n, fl in enumerate(flights):
    folder = os.path.join(root, "carbonZ_" + fl)
    files = os.listdir(folder)
    fs = [f for f in files if "failure_status" in f.lower() and f.endswith(".csv")]
    rc = [f for f in files if f.endswith("-mavros-rc-out.csv")]
    name = f"{n}:{ftype[fl][:5]}"
    if not fs or not rc:
        print("skipped", name); continue
    fault_ns = min(pd.read_csv(os.path.join(folder, f), usecols=["%time"])["%time"].min() for f in fs)
    d = pd.read_csv(os.path.join(folder, rc[0]))
    d["rel"] = (d["%time"] - fault_ns) / 1e9
    t0 = d["rel"].min()
    base = d[(d["rel"] >= t0 + 10) & (d["rel"] < -20)][CH]
    pre = d[(d["rel"] >= -20) & (d["rel"] < 0)][CH]
    flt = d[(d["rel"] >= 0) & (d["rel"] < 30)][CH]
    if len(base) < 50 or len(pre) < 20 or len(flt) < 20:
        print("skipped (too short)", name, len(base), len(pre), len(flt)); continue
    sb = base.std().replace(0, np.nan)
    real[name] = flt.std() / sb
    plc[name] = pre.std() / sb
    bsd[name] = base.std()

short = lambda c: c.replace("field.channels", "ch")
print("=== REAL: spread in the fault window / baseline spread (1 = unchanged, ~0 = flatline) ===")
print(pd.DataFrame(real).T.rename(columns=short).round(2).to_string())
print("\n=== PLACEBO: same ratio for the 20 s before the fault ===")
print(pd.DataFrame(plc).T.rename(columns=short).round(2).to_string())
print("\n=== baseline spread in microseconds (for scale) ===")
print(pd.DataFrame(bsd).T.rename(columns=short).round(1).to_string())
