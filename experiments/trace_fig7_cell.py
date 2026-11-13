import numpy as np
import evalkit

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"]
g = dev[dev["fault_type"] == "engine"]
print("engine rows:", len(g), "| flights:", g["flight"].nunique())
print(g[["flight", "t", "t_rel_fault", "alt_error__mean5", "label"]].head(8).to_string(index=False))

x = g["alt_error__mean5"] - g.groupby("flight")["alt_error__mean5"].transform("mean")
y = g["label"] - g.groupby("flight")["label"].transform("mean")
num = float((x * y).sum())
den = float(np.sqrt((x ** 2).sum() * (y ** 2).sum()))
print("sum(x*y) =", round(num, 3), "| sqrt(sum(x^2)*sum(y^2)) =", round(den, 3))
print("r =", round(num / den, 3), "  (fig7 shows 0.65)")