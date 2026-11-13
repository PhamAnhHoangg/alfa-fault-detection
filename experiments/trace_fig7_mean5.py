import numpy as np
import pandas as pd
import evalkit

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"]
signals = ["alt_error", "xtrack_error", "hud_altitude", "vel_lin_z", "roll_gap", "pitch_gap", "yaw_gap"]
types = ["engine", "elevator", "aileron", "rudder"]

rows = []
for s in signals:
    col = s + "__mean5"
    for ft in types:
        g = dev[dev["fault_type"] == ft]
        x = g[col] - g.groupby("flight")[col].transform("mean")
        y = g["label"] - g.groupby("flight")["label"].transform("mean")
        sxy = float((x * y).sum())
        sxx = float((x ** 2).sum())
        syy = float((y ** 2).sum())
        den = float(np.sqrt(sxx * syy))
        rows.append({"feature": col, "type": ft, "rows": len(g), "flights": g["flight"].nunique(),
                     "Sxy": round(sxy, 3), "Sxx": round(sxx, 3), "Syy": round(syy, 3),
                     "bottom": round(den, 3), "r": round(sxy / den, 2)})

out = pd.DataFrame(rows)
out.to_csv("results/fig7_mean5_numbers.csv", index=False)
print(out.to_string(index=False))