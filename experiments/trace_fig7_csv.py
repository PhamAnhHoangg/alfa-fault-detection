import evalkit

df = evalkit.load(cap=30)
dev = df[df["role"] != "test"]
g = dev[dev["fault_type"] == "engine"].copy()
col = "alt_error__mean5"

out = g[["flight", "t", "t_rel_fault", col, "label"]].copy()
out["flight_mean_x"] = g.groupby("flight")[col].transform("mean")
out["x_tilde"] = out[col] - out["flight_mean_x"]
out["flight_mean_y"] = g.groupby("flight")["label"].transform("mean")
out["y_tilde"] = out["label"] - out["flight_mean_y"]
out["x_tilde_times_y_tilde"] = out["x_tilde"] * out["y_tilde"]

out.to_csv("results/fig7_engine_trace.csv", index=False)
print("rows:", len(out))
print("sum of last column:", round(float(out["x_tilde_times_y_tilde"].sum()), 3))