import os
import pandas as pd

root = r"E:\ml-course\data\alfa\processed\processed"
split = pd.read_csv(r"E:\ml-course\data\split.csv")
fl = split[(split["role"] != "test") & (split["fault_type"] == "aileron")]["flight"].iloc[0]
folder = next((p for p in (os.path.join(root, fl), os.path.join(root, "carbonZ_" + fl)) if os.path.isdir(p)), None)
print("flight:", fl)
print("folder:", folder)
if folder is None:
    raise SystemExit("folder not found")

print("\n=== csv files: name | n_columns | first columns ===")
for f in sorted(os.listdir(folder)):
    if f.endswith(".csv"):
        cols = list(pd.read_csv(os.path.join(folder, f), nrows=1).columns)
        print(f[:-4][:40].ljust(40), str(len(cols)).rjust(3), "|", ", ".join(cols[:6])[:95])

print("\n=== rc-out / servo files: all columns ===")
for f in sorted(os.listdir(folder)):
    if f.endswith(".csv") and any(w in f.lower() for w in ("rc", "servo", "actuator", "output")):
        cols = list(pd.read_csv(os.path.join(folder, f), nrows=1).columns)
        print(f[:-4], "->", ", ".join(cols)[:300])
