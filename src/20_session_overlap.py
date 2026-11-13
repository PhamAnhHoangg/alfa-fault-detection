"""Check whether sessions overlap in time (ROADMAP item 31).
Prints each session's IMU time range and flags sessions that start before the previous one ends."""
import re
import pandas as pd
import config

ROOT = config.DATA / "alfa" / "processed" / "processed"
split = pd.read_csv(config.DATA / "split.csv")
role = dict(zip(split["session"], split["role"]))

rows = []
for f in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    m = re.search(r"carbonZ_(\d{4}-\d\d-\d\d-\d\d-\d\d-\d\d)", f.name)
    t = pd.read_csv(f / f"{f.name}-mavros-imu-data.csv", usecols=["%time"])["%time"]
    rows.append((m.group(1), t.iloc[0] / 1e9, t.iloc[-1] / 1e9))

s = (pd.DataFrame(rows, columns=["session", "start", "end"])
     .groupby("session").agg(start=("start", "min"), end=("end", "max")).sort_values("start"))
s["role"] = s.index.map(role)
s["gap_s"] = s["start"] - s["end"].shift()
s["overlap"] = s["gap_s"] < 0
print(s[["role", "gap_s", "overlap"]].round(1).to_string())
bad = s[s["overlap"] & (s["role"] != s["role"].shift())]
print("\noverlapping sessions in different groups:", len(bad))
if len(bad):
    print(bad[["role", "gap_s"]].round(1).to_string())
