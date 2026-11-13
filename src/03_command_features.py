import config
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
ROOT = config.DATA / "alfa" / "processed" / "processed"
table = pd.read_csv(str(config.DATA / r"table.csv"))
split = pd.read_csv(str(config.DATA / r"split.csv"))[["flight", "role"]]

frames = []
for flight in table["flight"].unique():
    f = ROOT / ("carbonZ_" + flight)
    imu_t = pd.read_csv(f / f"{f.name}-mavros-imu-data.csv", usecols=["%time"])["%time"].to_numpy()
    start, end = imu_t[0], imu_t[-1]
    n = int((end - start) // 1_000_000_000) + 1
    d = pd.read_csv(f / f"{f.name}-mavros-rc-out.csv")
    sec = ((d["%time"].to_numpy() - start) // 1_000_000_000).astype(int)
    out = pd.DataFrame({"flight": flight, "t": np.arange(n)})
    for i in (1, 3, 4, 5):
        g = d[f"field.channels{i}"].groupby(sec)
        sd = g.std(ddof=0).reindex(range(n))
        cnt = g.size().reindex(range(n)).fillna(0)
        fr = (sd == 0).astype(float)
        fr[cnt.to_numpy() < 2] = np.nan                 # too few samples: no evidence
        out[f"cmd_frozen_ch{i}"] = fr.rolling(3, min_periods=3).min().fillna(0).to_numpy()
    frames.append(out)

cmd = pd.concat(frames, ignore_index=True)
cols = [c for c in cmd.columns if c.startswith("cmd_frozen_ch")]
cmd["cmd_frozen_any"] = cmd[cols].max(axis=1)
allc = cols + ["cmd_frozen_any"]

t2 = table.merge(cmd, on=["flight", "t"], how="left")
print("rows without a match (filled with 0):", int(t2["cmd_frozen_any"].isna().sum()))
t2[allc] = t2[allc].fillna(0)
t2.to_csv(str(config.DATA / r"table2.csv"), index=False)
print("saved table2.csv, shape:", t2.shape)

chk = t2.merge(split, on="flight")
chk = chk[(chk["role"] != "test") & ~(chk["t_rel_fault"] > 30)]      # test set NOT used
rows = []
for ft, g in chk.groupby("fault_type"):
    b, a = g[g["label"] == 0], g[g["label"] == 1]
    r = {"type": ft, "normal_rows": len(b), "false_alarm_%": 100 * b["cmd_frozen_any"].mean(),
         "fault_rows": len(a), "any_%": 100 * a["cmd_frozen_any"].mean() if len(a) else np.nan}
    for c in cols:
        r[c.replace("cmd_frozen_", "") + "_%"] = 100 * a[c].mean() if len(a) else np.nan
    rows.append(r)
print("\n=== share of seconds with a frozen channel (development rows, first 30 s of the fault) ===")
print(pd.DataFrame(rows).round(1).to_string(index=False))

