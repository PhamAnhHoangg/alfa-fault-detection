import config
import numpy as np
import pandas as pd
import evalkit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from evalkit import load, feature_lists, fit_rule, apply_rule, rule_text, persist, evaluate

evalkit.TABLE = str(config.DATA / r"table2.csv")
pd.set_option("display.width", 250)
df = load()
sym, _ = feature_lists(df)
sensors = [c for c in sym if not c.startswith("rc_out")]
dev = df[df["role"] != "test"]
test = df[df["role"] == "test"].copy()
print("dev rows:", len(dev), "| test rows:", len(test), "| test fault rows:", int(test["label"].sum()))
print("test flights by type:", test.groupby("fault_type")["flight"].nunique().to_dict())

# A: one rule on sensors
ruleA = fit_rule(dev, sensors)
predA = persist(test, apply_rule(test, ruleA), 3)
# F: forest on sensors
forest = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                              class_weight="balanced_subsample",
                                              random_state=0, n_jobs=-1)).fit(dev[sensors], dev["label"])
predF = persist(test, forest.predict_proba(test[sensors])[:, 1] >= 0.5, 3)
# D2: frozen channel OR throttle level
ruleT = fit_rule(dev, ["cause_rc_out_2__mean5"])
thr = persist(test, apply_rule(test, ruleT), 3)
frozen = test["cmd_frozen_any"].to_numpy() == 1
predD = frozen | thr
print("rule A:", rule_text(ruleA), "| D2 throttle rule:", rule_text(ruleT))

COLS = ["detector", "f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min",
        "det_eng", "pre_eng", "det_oth", "pre_oth", "med_delay_s"]
rows = []
for name, pred in (("A sensor rule", predA), ("F sensor forest", predF), ("D2 command-side", predD)):
    r = evaluate(test, pred)
    r["detector"] = name
    rows.append(r)
print("\n=== LOCKED TEST SET, overall ===")
print(pd.DataFrame(rows)[COLS].round(2).to_string(index=False))

fl_rows = []
for fl, g in test.assign(A=predA, F=predF, D=predD).groupby("flight"):
    row = {"flight": fl[:19], "type": g["fault_type"].iloc[0], "rows": len(g),
           "fault_rows": int((g["label"] == 1).sum())}
    for k in ("A", "F", "D"):
        hit = g[(g["label"] == 1) & g[k]]
        row[k + "_delay_s"] = max(hit["t_rel_fault"].min(), 0) if len(hit) else np.nan
        row[k + "_false_s"] = int(((g["label"] == 0) & g[k]).sum())
    fl_rows.append(row)
print("\n=== per flight: delay of first alarm after the fault (NaN = missed), false-alarm seconds ===")
print(pd.DataFrame(fl_rows).round(1).to_string(index=False))

hat = pd.Series("none", index=test.index)
hat[thr] = "engine"
hat[test["cmd_frozen_ch1"].to_numpy() == 1] = "elevator"
hat[test["cmd_frozen_ch3"].to_numpy() == 1] = "rudder"
hat[(test["cmd_frozen_ch4"] + test["cmd_frozen_ch5"]).to_numpy() > 0] = "aileron"
m = (test["label"].to_numpy() == 1) & predD
print("\n=== D2 isolation on the test set: true type (rows) vs guessed surface (columns) ===")
print(pd.crosstab(test["fault_type"].to_numpy()[m], hat.to_numpy()[m]).to_string())

