import config
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

ASSIGN = {
    "2018-07-18-15-53-31": "cv1",
    "2018-07-18-16-22-01": "cv4",
    "2018-07-18-16-37-39": "test",
    "2018-07-30-16-29-45": "cv1",
    "2018-07-30-16-39-00": "cv4",
    "2018-07-30-17-10-45": "cv1",
    "2018-07-30-17-20-01": "cv2",
    "2018-07-30-17-36-35": "cv1",
    "2018-07-30-17-46-31": "cv2",
    "2018-09-11-11-56-30": "test",
    "2018-09-11-14-16-55": "cv2",
    "2018-09-11-14-22-07": "test",
    "2018-09-11-14-41-38": "cv3",
    "2018-09-11-14-41-51": "test",
    "2018-09-11-14-52-54": "cv3",
    "2018-09-11-15-05-11": "cv3",
    "2018-09-11-15-06-34": "cv4",
    "2018-09-11-17-27-13": "test",
    "2018-09-11-17-55-30": "cv2",
    "2018-10-05-14-34-20": "cv1",
    "2018-10-05-14-37-22": "cv0",
    "2018-10-05-15-52-12": "cv0",
    "2018-10-05-15-55-10": "cv4",
    "2018-10-05-16-04-46": "test",
    "2018-10-18-11-03-57": "cv1",
    "2018-10-18-11-04-00": "cv0",
    "2018-10-18-11-04-08": "cv2",
    "2018-10-18-11-04-35": "cv3",
    "2018-10-18-11-06-06": "cv3",
    "2018-10-18-11-08-24": "cv3",
}

table = pd.read_csv(str(config.DATA / r"table.csv"))
fl = (table.groupby("flight")
      .agg(session=("session", "first"), fault_type=("fault_type", "first"),
           rows=("t", "size"), fault_rows=("label", "sum"))
      .reset_index())
fl["role"] = fl["session"].map(ASSIGN)
if fl["role"].isna().any():
    print(fl[fl["role"].isna()])
    raise SystemExit("some sessions have no assignment: add them to ASSIGN")
fl["fault_group"] = np.where(fl.fault_type == "none", "none",
                             np.where(fl.fault_type == "engine", "engine", "other"))

for k in ["engine", "aileron", "rudder", "elevator", "none"]:
    fl["n_" + k] = (fl.fault_type == k).astype(int)
fl["share"] = fl["fault_rows"] / fl.groupby("role")["fault_rows"].transform("sum")
summary = fl.groupby("role").agg(
    flights=("flight", "size"), engine=("n_engine", "sum"), aileron=("n_aileron", "sum"),
    rudder=("n_rudder", "sum"), elevator=("n_elevator", "sum"), none=("n_none", "sum"),
    rows=("rows", "sum"), fault_rows=("fault_rows", "sum"),
    biggest_flight_share=("share", "max"))
print(summary.round(2).to_string())
n_split = int((fl.groupby("session")["role"].nunique() > 1).sum())
print("\nsessions split across more than one part (must be 0):", n_split)
if n_split:
    raise SystemExit("a session appears in two groups")
print("total rows:", int(fl["rows"].sum()), " total fault rows:", int(fl["fault_rows"].sum()))
print("\nLOCKED TEST FLIGHTS (do not tune on these):")
for _, r in fl[fl.role == "test"].iterrows():
    print("  ", r.fault_type.ljust(9), r.flight)

fl[["flight", "session", "fault_type", "fault_group", "role", "rows", "fault_rows"]].to_csv(
    str(config.DATA / r"split.csv"), index=False)
print("\nsaved split.csv")


