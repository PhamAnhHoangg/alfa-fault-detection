from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"E:\ml-course\data\alfa\processed\processed")
F = ROOT / "carbonZ_2018-07-18-15-53-31_1_engine_failure"
FAULT_AT = 116.2   # seconds after flight start (from Step 4)

def load(topic):
    return pd.read_csv(F / f"{F.name}-{topic}.csv")

start = load("mavros-imu-data")["%time"].iloc[0]

SKIP = ("header", "covariance", "seq", "stamp", "frame", "%time")
def signal_cols(df):
    return [c for c in df.columns
            if not any(s in c for s in SKIP) and pd.api.types.is_numeric_dtype(df[c])]

print("mavlink-from columns:", list(load("mavlink-from").columns)[:15])
print()

TOPICS = ["mavros-nav_info-roll", "mavros-nav_info-pitch", "mavros-nav_info-yaw",
          "mavros-nav_info-errors", "mavros-nav_info-airspeed", "mavros-vfr_hud",
          "mavros-battery", "mavros-imu-data_raw", "mavros-rc-out",
          "mavros-local_position-velocity"]

fig, axes = plt.subplots(len(TOPICS), 1, figsize=(11, 2.2 * len(TOPICS)), sharex=True)
for ax, topic in zip(axes, TOPICS):
    df = load(topic)
    cols = signal_cols(df)
    print(f"{topic}: {len(cols)} signals ->", cols[:6])
    t = (df["%time"] - start) / 1e9
    for c in cols[:3]:
        ax.plot(t, df[c], label=c.replace("field.", ""), linewidth=0.8)
    ax.axvline(FAULT_AT, color="red", linestyle="--")
    ax.set_ylabel(topic.replace("mavros-", ""), fontsize=7)
    ax.legend(fontsize=6, loc="upper left")
axes[-1].set_xlabel("seconds since flight start (red line = engine fails)")
plt.tight_layout()
Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/engine_flight.png", dpi=110)
print("\nsaved plots/engine_flight.png")
