import pandas as pd

pd.set_option("display.width", 250)
t = pd.read_csv(r"E:\ml-course\data\table2.csv")
s = pd.read_csv(r"E:\ml-course\data\split.csv")[["flight", "role"]]
t = t.merge(s, on="flight")

print("=== all rudder flights (role, name) ===")
for fl, r in t[t["fault_type"] == "rudder"][["flight", "role"]].drop_duplicates().itertuples(index=False):
    print(r.ljust(5), fl)

cols = ["cmd_frozen_ch1", "cmd_frozen_ch3", "cmd_frozen_ch4", "cmd_frozen_ch5"]
g = t[(t["role"] == "test") & t["fault_type"].isin(["rudder", "aileron"])
      & (t["label"] == 1) & ~(t["t_rel_fault"] > 30)]
out = (g.groupby(["flight", "fault_type"])[cols].mean() * 100).round(0)
out.columns = [c.replace("cmd_frozen_", "") for c in out.columns]
print("\n=== test flights: % of fault seconds (first 30 s) with each command channel frozen ===")
print(out.to_string())
