import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from evalkit import load, feature_lists, evaluate

pd.set_option("display.width", 250)
df = load()
sym, allf = feature_lists(df)
dev = df[df["role"] != "test"]          # locked test set is NOT used

def make(name):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced_subsample",
                                                random_state=0, n_jobs=-1))

COLS = ["fold", "f1", "prec", "rec_eng", "rec_oth", "fa_s_per_min",
        "det_eng", "pre_eng", "det_oth", "pre_oth", "med_delay_s", "ap"]

def run(name, feats, k=3):
    rows = []
    for i in range(5):
        tr = dev[dev["role"] != f"cv{i}"]
        va = dev[dev["role"] == f"cv{i}"]
        m = make(name).fit(tr[feats], tr["label"])
        proba = m.predict_proba(va[feats])[:, 1]
        r = evaluate(va, proba >= 0.5, k_persist=k)
        r["ap"] = average_precision_score(va["label"], proba)
        r["fold"] = f"cv{i}"
        rows.append(r)
    out = pd.DataFrame(rows)[COLS]
    print(f"\n=== {name}, symptom features, 3 seconds in a row ===")
    print(out.round(2).to_string(index=False))
    print("mean F1: %.2f (std %.2f) | mean AP: %.2f" % (out["f1"].mean(), out["f1"].std(), out["ap"].mean()))

for name in ("logreg", "forest"):
    run(name, sym)

print("\n=== top features (model fitted on all development rows) ===")
m = make("logreg").fit(dev[sym], dev["label"])
coef = pd.Series(m[-1].coef_[0], index=sym)
print("logreg, biggest weights (standardized):")
print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(6).round(2).to_string())
m = make("forest").fit(dev[sym], dev["label"])
imp = pd.Series(m[-1].feature_importances_, index=sym).sort_values(ascending=False)
print("forest, biggest importances:")
print(imp.head(6).round(3).to_string())
