from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\ml-course\data\alfa\processed\processed")
flights = sorted(p for p in ROOT.iterdir() if p.is_dir())

def names(folder):
    return {c.name.replace(folder.name + "-", "") for c in folder.glob("*.csv")}

normal = next(f for f in flights if f.name.endswith("_no_failure"))
faulty = next(f for f in flights if f.name.endswith("_engine_failure"))

print("normal :", normal.name)
print("faulty :", faulty.name)
print("\nfiles ONLY in the faulty flight:")
for n in sorted(names(faulty) - names(normal)):
    print("  ", n)
print("\nfiles ONLY in the normal flight:")
for n in sorted(names(normal) - names(faulty)):
    print("  ", n)

for n in sorted(names(faulty) - names(normal)):
    df = pd.read_csv(faulty / f"{faulty.name}-{n}")
    print("\n=====", n, df.shape)
    print("columns:", list(df.columns))
    print(df.head(4).to_string())
    print(df.tail(2).to_string())

imu = pd.read_csv(faulty / f"{faulty.name}-mavros-imu-data.csv")
t = imu["%time"]
print("\nIMU rows:", len(imu), " duration (s):", (t.iloc[-1] - t.iloc[0]) / 1e9)
print("IMU start time:", t.iloc[0])
