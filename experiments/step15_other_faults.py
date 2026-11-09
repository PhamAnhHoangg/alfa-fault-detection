import numpy as np
import pandas as pd
from evalkit import load

pd.set_option("display.width", 250)
df = load(cap=30)
dev = df[(df["role"] != "test") & (~df["fault_type"].isin(["engine", "none"]))]   # test set NOT used
sig = [c for c in df.columns if c.endswith("__mean5")]

real, plc = {}, {}
for n, (fl, g) in enumerate(dev.groupby("flight")):
    base = g[g["t_rel_fault"] < -20]
    pre = g[(g["t_rel_fault"] >= -20) & (g["t_rel_fault"] < 0)]
    flt = g[g["label"] == 1]
    name = f"{n}:{g['fault_type'].iloc[0][:5]} {fl.split('_')[1][5:]}"
    if len(base) < 15 or len(pre) < 10 or len(flt) < 5:
        print("skipped (too short):", name, len(base), len(pre), len(flt))
        continue
    sd = base[sig].std().replace(0, np.nan)
    real[name] = ((flt[sig].mean() - base[sig].mean()) / sd).abs()
    plc[name] = ((pre[sig].mean() - base[sig].mean()) / sd).abs()

short = {c: c.replace("__mean5", "")[:11] for c in sig}
print("\n=== REAL: |shift| in baseline std units, first 30 s of the fault ===")
print(pd.DataFrame(real).T.rename(columns=short).round(1).to_string())
print("\n=== PLACEBO: same measure, 20 s before the fault ===")
print(pd.DataFrame(plc).T.rename(columns=short).round(1).to_string())
