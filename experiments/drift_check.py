"""Start-of-flight drift check for the fig7 engine column (ROADMAP item 21).
Drops the first 30 s of each engine dev flight and compares with dropping the same number of random normal seconds.
Run from the repo root with PYTHONPATH=src."""
import numpy as np
import pandas as pd
import evalkit

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"]
sym, feats = evalkit.feature_lists(dev)
cols = [c for c in sym if not c.startswith("rc_out")]
g0 = dev[dev["fault_type"] == "engine"]


def r(g):
    x = g[cols] - g.groupby("flight")[cols].transform("mean")
    y = g["label"] - g.groupby("flight")["label"].transform("mean")
    return x.corrwith(y)


CUT = 30
early = g0[g0["t"] < CUT]
assert early["label"].sum() == 0, "a fault starts inside the cut"
n_drop = early.groupby("flight").size()
rng = np.random.default_rng(0)
rand = []
for _ in range(50):
    drop = []
    for fl, k in n_drop.items():
        idx = g0[(g0.flight == fl) & (g0.label == 0)].index
        drop += list(rng.choice(idx, size=k, replace=False))
    rand.append(r(g0.drop(index=drop)))
R = pd.concat(rand, axis=1)
t = pd.DataFrame({"all rows": r(g0), "drop first 30 s": r(g0[g0["t"] >= CUT]),
                  "drop random (mean of 50)": R.mean(axis=1),
                  "random 5%": R.quantile(.05, axis=1), "random 95%": R.quantile(.95, axis=1)})
t["first minus random"] = t["drop first 30 s"] - t["drop random (mean of 50)"]
print(t.round(2).sort_values("first minus random", key=abs, ascending=False).to_string())
