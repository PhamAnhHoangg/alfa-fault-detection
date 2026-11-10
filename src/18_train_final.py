"""Train the final forest on ALL dev flights and save it for the API demo.

Same settings as the model comparison (300 trees, min_samples_leaf=5, balanced_subsample, seed 0).
The locked test set is not used. The saved file stays out of git (models/ is ignored)."""
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
import evalkit

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"].reset_index(drop=True)
sym, feats = evalkit.feature_lists(dev)
cols = [c for c in sym if not c.startswith("rc_out")]

X = dev[cols].to_numpy(float)
y = dev["label"].to_numpy(int)
model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                               class_weight="balanced_subsample",
                               random_state=0, n_jobs=-1).fit(X, y)

bundle = {"model": model, "features": cols, "threshold": 0.5, "k_persist": 3,
          "trained_on": f"{dev['flight'].nunique()} dev flights, {len(dev)} seconds"}
os.makedirs("models", exist_ok=True)
path = os.path.join("models", "forest.joblib")
joblib.dump(bundle, path)

print("trained on:", bundle["trained_on"])
print("saved:", path, "|", round(os.path.getsize(path) / 1e6, 1), "MB")
print("features (%d):" % len(cols))
for c in cols:
    print("  ", c)
