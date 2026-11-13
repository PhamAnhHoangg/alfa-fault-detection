import config
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import evalkit
from evalkit import load, feature_lists, fit_rule, apply_rule, persist

evalkit.TABLE = str(config.DATA / r"table2.csv")
ROOT = config.DATA / "alfa" / "processed" / "processed"
os.makedirs(str(config.PROJECT / "figures"), exist_ok=True)

df = load()
sym, _ = feature_lists(df)
sensors = [c for c in sym if not c.startswith("rc_out")]
dev = df[df["role"] != "test"]
test = df[df["role"] == "test"].copy()

ruleA = fit_rule(dev, sensors)
predA = persist(test, apply_rule(test, ruleA), 3)
forest = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                              class_weight="balanced_subsample",
                                              random_state=0, n_jobs=-1)).fit(dev[sensors], dev["label"])
predF = persist(test, forest.predict_proba(test[sensors])[:, 1] >= 0.5, 3)
ruleT = fit_rule(dev, ["cause_rc_out_2__mean5"])
thr = persist(test, apply_rule(test, ruleT), 3)
predD = (test["cmd_frozen_any"].to_numpy() == 1) | thr
test = test.assign(A=predA, F=predF, D=predD)

# ---- figure 1: one test engine flight
eng = test[test["fault_type"] == "engine"]
fl = eng.groupby("flight")["label"].sum().idxmax()
g = test[test["flight"] == fl]
x = g["t_rel_fault"]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 2, 1.3]})
ax[0].plot(x, g["vel_lin_z__mean5"], color="tab:blue")
ax[0].axhline(ruleA[3], ls="--", color="gray")
ax[0].set_ylabel("vertical velocity\n(5 s mean)")
ax[0].set_title("Test engine flight " + fl[:19] + " - dashed = threshold fitted on development data")
ax[1].plot(x, g["cause_rc_out_2__mean5"], color="tab:orange")
ax[1].axhline(ruleT[3], ls="--", color="gray")
ax[1].set_ylabel("throttle output\n(microseconds)")
for k, (col, name) in enumerate((("A", "A sensor rule"), ("F", "F sensor forest"), ("D", "D2 command-side"))):
    xs = x[g[col].to_numpy()]
    ax[2].scatter(xs, np.full(len(xs), k), marker="|", s=250)
ax[2].set_yticks([0, 1, 2])
ax[2].set_yticklabels(["A sensor rule", "F sensor forest", "D2 command-side"])
ax[2].set_ylim(-0.7, 2.7)
ax[2].set_xlabel("seconds relative to fault start")
for a in ax:
    a.axvline(0, color="red")
ax[0].text(0.3, ax[0].get_ylim()[1], " fault starts", color="red", va="top")
fig.tight_layout()
fig.savefig(str(config.PROJECT / "figures" / r"fig1_engine_flight.png"), dpi=150)

# ---- figure 2: raw command channels in the dual-fault test flight
fl2 = test[test["fault_type"] == "rudder"]["flight"].iloc[0]
folder = ROOT / ("carbonZ_" + fl2)
fs = list(folder.glob("*failure_status*.csv"))
fault_ns = min(pd.read_csv(f, usecols=["%time"])["%time"].min() for f in fs)
d = pd.read_csv(folder / f"{folder.name}-mavros-rc-out.csv")
d["rel"] = (d["%time"] - fault_ns) / 1e9
d = d[(d["rel"] >= -20) & (d["rel"] <= 30)]
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 6))
for a, ch, name in zip(ax, (3, 4, 5), ("ch3 - rudder (inferred)", "ch4 - left aileron (inferred)", "ch5 - right aileron (inferred)")):
    a.plot(d["rel"], d[f"field.channels{ch}"], lw=1)
    a.axvline(0, color="red")
    a.set_ylabel(name + "\nPWM (us)", fontsize=8)
ax[0].set_title("Test flight " + fl2, fontsize=9)
ax[2].set_xlabel("seconds relative to fault start")
fig.tight_layout()
fig.savefig(str(config.PROJECT / "figures" / r"fig2_frozen_channels.png"), dpi=150)
print("saved:", os.listdir(str(config.PROJECT / "figures")))

