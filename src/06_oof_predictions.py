import numpy as np
from sklearn.ensemble import RandomForestClassifier
import config
import evalkit

d = evalkit.load(cap=30)
dev = d[d["role"] != "test"].reset_index(drop=True)
sym, _ = evalkit.feature_lists(dev)
feats = [c for c in sym if not c.startswith("rc_out")]
print("features:", len(feats))

folds = sorted(dev["role"].unique())
dev["proba"] = np.nan
for k in folds:
    tr = dev["role"] != k
    te = ~tr
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                min_samples_leaf=5, n_jobs=-1, random_state=0)
    rf.fit(dev.loc[tr, feats], dev.loc[tr, "label"])
    dev.loc[te, "proba"] = rf.predict_proba(dev.loc[te, feats])[:, 1]

dev["pred_raw"] = dev["proba"] >= 0.5
dev["pred"] = evalkit.persist(dev, dev["pred_raw"], 3)

f1s = []
for k in folds:
    m = dev[dev["role"] == k]
    r = evalkit.evaluate(m, m["pred_raw"].to_numpy(), k_persist=3)
    f1s.append(r["f1"])
    print(k, "F1 %.2f  precision %.2f" % (r["f1"], r["prec"]))
print("mean fold F1 %.2f, std %.2f  (earlier dev result: 0.51 +/- 0.16)" % (np.mean(f1s), np.std(f1s)))

y = dev["label"] == 1
print("recall by fault type (per-second / after 3-in-a-row):")
for t in ["engine", "elevator", "aileron", "rudder"]:
    m = y & (dev["fault_type"] == t)
    print("  %-9s %.2f / %.2f  (%d fault seconds)" % (t, dev.loc[m, "pred_raw"].mean(), dev.loc[m, "pred"].mean(), m.sum()))

out = config.PROJECT / "results"
out.mkdir(exist_ok=True)
cols = ["flight", "role", "day", "fault_type", "t", "t_rel_fault", "label", "proba", "pred_raw", "pred"]
dev[cols].to_csv(out / "oof_predictions.csv", index=False)
print("saved", out / "oof_predictions.csv")
