import os, textwrap
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
root = r"E:\ml-course\data\alfa\processed\processed"
split = pd.read_csv(r"E:\ml-course\data\split.csv")
dev = split[(split["role"] != "test") & (~split["fault_type"].isin(["engine", "none"]))]   # test set NOT used
flights = sorted(dev["flight"])
ftype = dict(zip(dev["flight"], dev["fault_type"]))

# --- part 1: topic names of one flight
fl0 = flights[0]
pref = "carbonZ_" + fl0
names = []
for f in sorted(os.listdir(os.path.join(root, pref))):
    if f.endswith(".csv"):
        f = f[:-4]
        names.append(f[len(pref):].lstrip("-") if f.startswith(pref) else f)
print("=== topics in flight", fl0, "===")
print(textwrap.fill(", ".join(names), 160))

# --- part 2: rc-out channels, real vs placebo
real, plc, delta = {}, {}, {}
for n, fl in enumerate(flights):
    folder = os.path.join(root, "carbonZ_" + fl)
    files = os.listdir(folder)
    fs = [f for f in files if "failure_status" in f.lower() and f.endswith(".csv")]
    rc = [f for f in files if f.endswith("-mavros-rc-out.csv")]
    name = f"{n}:{ftype[fl][:5]}"
    if not fs or not rc:
        print("skipped", name, "failure files:", len(fs), "rc-out files:", len(rc))
        continue
    fault_ns = min(pd.read_csv(os.path.join(folder, f), usecols=["%time"])["%time"].min() for f in fs)
    d = pd.read_csv(os.path.join(folder, rc[0]))
    ch = [c for c in d.columns if c.startswith("field.channels")]
    d["s"] = np.floor((d["%time"] - fault_ns) / 1e9).astype(int)
    per = d.groupby("s")[ch].mean()
    base = per[(per.index >= per.index.min() + 10) & (per.index < -20)]
    pre = per[(per.index >= -20) & (per.index < 0)]
    flt = per[(per.index >= 0) & (per.index < 30)]
    if len(base) < 15 or len(pre) < 10 or len(flt) < 5:
        print("skipped (too short)", name, len(base), len(pre), len(flt))
        continue
    sd = base.std().replace(0, np.nan)
    real[name] = ((flt.mean() - base.mean()) / sd).abs()
    plc[name] = ((pre.mean() - base.mean()) / sd).abs()
    delta[name] = flt.mean() - base.mean()

short = lambda c: c.replace("field.channels", "ch")
print("\n=== REAL: |shift| in baseline std units, first 30 s of fault ===")
print(pd.DataFrame(real).T.rename(columns=short).round(1).to_string())
print("\n=== PLACEBO: 20 s before the fault ===")
print(pd.DataFrame(plc).T.rename(columns=short).round(1).to_string())
print("\n=== SIGNED change in the fault window (raw units, microseconds PWM) ===")
print(pd.DataFrame(delta).T.rename(columns=short).round(0).to_string())
