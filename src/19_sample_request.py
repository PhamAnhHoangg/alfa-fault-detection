"""Write a sample /predict request from one dev engine-fault flight (demo only).
The flight was in the final forest's training data, so this is not a performance test."""
import json
import os, config
import numpy as np
import evalkit

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"]
sym, feats = evalkit.feature_lists(dev)
cols = [c for c in sym if not c.startswith("rc_out")]

flight = dev[dev["fault_type"] == "engine"]["flight"].unique()[0]
d = dev[dev["flight"] == flight].sort_values("t_rel_fault")
onset = int(np.argmax(d["label"].to_numpy() == 1))

payload = {
    "flight": str(flight),
    "onset_second": onset,
    "rows": [{c: float(v) for c, v in zip(cols, row)} for row in d[cols].to_numpy(float)],
}
os.makedirs(config.PROJECT / "results", exist_ok=True)
with open(os.path.join(config.PROJECT, "results", "sample_request.json"), "w") as fh:
    json.dump(payload, fh)
print("flight:", flight, "| seconds:", len(d), "| fault starts at second index:", onset,
      "| fault seconds:", int((d["label"] == 1).sum()))
