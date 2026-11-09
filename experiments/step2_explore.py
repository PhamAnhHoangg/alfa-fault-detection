from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\ml-course\data\alfa\processed\processed")

flights = sorted(p for p in ROOT.iterdir() if p.is_dir())
print("flight folders:", len(flights))

# 1) how are the folders named?
for f in flights:
    n = len(list(f.glob("*.csv")))
    print(f"{n:3d} csv  {f.name}")

# 2) look inside ONE flight: list its csv files (short names)
one = flights[0]
print("\n--- inside:", one.name)
for c in sorted(one.glob("*.csv")):
    print("  ", c.name.replace(one.name + "-", ""))

# 3) open one csv and look at it
csv = sorted(one.glob("*.csv"))[0]
df = pd.read_csv(csv)
print("\n--- file:", csv.name[-60:])
print("rows, columns:", df.shape)
print("columns:", list(df.columns)[:12])
print(df.head(3).to_string())