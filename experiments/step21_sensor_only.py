import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from evalkit import load, feature_lists, evaluate

pd.set_option("display.width", 250)
df = load()
sym, allf = feature_lists(df)
dev = df[df["role"] != "test"]          # locked test set is NOT used

def make():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced_subsample",
                                                random_state=0, n_jobs=-1))

COLS = ["fold", "f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min",
        "det_eng", "pre_eng", "det_oth", "pre_oth", "med_delay_s", "ap"]

def run(feats, title):
    rows = []
    for i in range(5):
        tr = dev[dev["role"] != f"cv{i}"]
        va = dev[dev["role"] == f"cv{i}"]
        m = make().fit(tr[feats], tr["label"])
        proba = m.predict_proba(va[feats])[:, 1]
        r = evaluate(va, proba >= 0.5, k_persist=3)
        r["ap"] = average_precision_score(va["label"], proba)
        r["fold"] = f"cv{i}"
        rows.append(r)
    out = pd.DataFrame(rows)[COLS]
    print("\n=== " + title + " ===")
    print(out.round(2).to_string(index=False))
    print("mean F1: %.2f (std %.2f) | mean AP: %.2f" % (out["f1"].mean(), out["f1"].std(), out["ap"].mean()))

sensors = [c for c in sym if not c.startswith("rc_out")]
print("sensor-only features:", len(sensors), "| with rc_out_1:", len(sym))
run(sensors, "forest, SENSORS ONLY (no rc_out)")
run(sym, "forest, sensors + rc_out_1 (as in Step 13)")
