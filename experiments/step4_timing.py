from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\ml-course\data\alfa\processed\processed")
flights = sorted(p for p in ROOT.iterdir() if p.is_dir())

print("=== A) fault timing in every flight ===")
rows = []
for f in flights:
    tag = f.name.replace("carbonZ_", "")
    imu = pd.read_csv(f / f"{f.name}-mavros-imu-data.csv", usecols=["%time"])
    start, end = imu["%time"].iloc[0], imu["%time"].iloc[-1]
    fs = sorted(f.glob("*failure_status*.csv"))
    if not fs:
        rows.append(dict(flight=tag, status_files="none", fault_at_s=None,
                         post_fault_s=None, values="", total_s=round((end-start)/1e9, 1)))
        continue
    for c in fs:
        d = pd.read_csv(c)
        ft = d["%time"].iloc[0]
        rows.append(dict(flight=tag,
                         status_files=c.name.split("failure_status-")[1],
                         fault_at_s=round((ft-start)/1e9, 1),
                         post_fault_s=round((end-ft)/1e9, 1),
                         values=sorted(d["field.data"].unique().tolist()),
                         total_s=round((end-start)/1e9, 1)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== B) rate of every topic in one normal flight ===")
nf = next(f for f in flights if f.name.endswith("_no_failure"))
out = []
for c in sorted(nf.glob("*.csv")):
    t = pd.read_csv(c, usecols=["%time"])["%time"]
    dur = (t.iloc[-1] - t.iloc[0]) / 1e9
    out.append(dict(topic=c.name.replace(nf.name + "-", ""), rows=len(t),
                    hz=round(len(t)/dur, 1) if dur > 0 else None))
print(pd.DataFrame(out).to_string(index=False))

print("\n=== C) topics present in ALL flights vs only some ===")
from collections import Counter
cnt = Counter()
for f in flights:
    for n in {c.name.replace(f.name + "-", "") for c in f.glob("*.csv")}:
        cnt[n] += 1
for n, k in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"{k:3d}/{len(flights)}  {n}")
