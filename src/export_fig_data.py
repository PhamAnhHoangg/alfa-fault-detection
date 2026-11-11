"""Exports small real-data snippets for the figure GIFs (fig1 to fig6, fig8).

Run on your PC (needs results/oof_predictions.csv, results/model_comparison.csv,
table.csv, table2.csv, split.csv):
    python src/export_fig_data.py

Writes results/gif_data/fig*_*.csv. Only a few rows of public ALFA data and a few
summary numbers. Nothing is trained and nothing is scored.
"""
import numpy as np
import pandas as pd
import config
import evalkit

OUT = config.PROJECT / "results" / "gif_data"
OUT.mkdir(parents=True, exist_ok=True)
oof = pd.read_csv(config.PROJECT / "results" / "oof_predictions.csv")
y = oof["label"] == 1

# ---- fig3: counts behind each bar
rows = []
for t in ["engine", "elevator", "aileron", "rudder"]:
    m = y & (oof["fault_type"] == t)
    rows.append({"fault_type": t, "fault_seconds": int(m.sum()),
                 "raw_hits": int(oof.loc[m, "pred_raw"].sum()), "alarm_hits": int(oof.loc[m, "pred"].sum()),
                 "raw_recall": round(oof.loc[m, "pred_raw"].mean(), 4),
                 "alarm_recall": round(oof.loc[m, "pred"].mean(), 4)})
pd.DataFrame(rows).to_csv(OUT / "fig3_counts.csv", index=False)

# ---- fig4: raw counts behind the four cells
p = oof["pred"].astype(bool)
tn, fp, fn, tp = int((~y & ~p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((y & p).sum())
pd.DataFrame([{"tn": tn, "fp": fp, "fn": fn, "tp": tp,
               "normal_total": tn + fp, "fault_total": fn + tp}]).to_csv(OUT / "fig4_counts.csv", index=False)

# ---- fig5: rows of the first engine flight (what one grey bar and its red ticks are made of)
f = oof[(oof["fault_type"] == "engine") & oof["t_rel_fault"].notna()]
first = sorted(f["flight"].unique())[0]
g = f[f["flight"] == first].sort_values("t")
g[["flight", "t", "t_rel_fault", "label", "proba", "pred_raw", "pred"]].round(3).to_csv(OUT / "fig5_flight_rows.csv", index=False)
pd.DataFrame([{"flight": first, "rows": len(g), "t_rel_min": g["t_rel_fault"].min(),
               "t_rel_max": g["t_rel_fault"].max(), "alarm_ticks": int(g["pred"].sum())}]).to_csv(OUT / "fig5_flight_info.csv", index=False)

# ---- fig6: one cell, alt_error__std5 vs hud_altitude__std5, dev rows pooled
df = evalkit.load()
dev = df[df["role"] != "test"] if "role" in df.columns else df
a, b = "alt_error__std5", "hud_altitude__std5"
d = dev[[a, b]].dropna()
x, z = d[a].to_numpy(), d[b].to_numpy()
dx, dz = x - x.mean(), z - z.mean()
sxy, sxx, szz = float((dx * dz).sum()), float((dx ** 2).sum()), float((dz ** 2).sum())
d.head(8).round(4).to_csv(OUT / "fig6_rows_head.csv", index=False)
pd.DataFrame([{"x": a, "z": b, "n_rows": len(d), "mean_x": x.mean(), "mean_z": z.mean(), "sxy": sxy, "sxx": sxx, "szz": szz,
               "r": sxy / np.sqrt(sxx * szz), "r_pandas": float(np.corrcoef(x, z)[0, 1])}]).to_csv(OUT / "fig6_cell_sums.csv", index=False)

# ---- fig1 and fig2: rows of the two test flights drawn in the figures (already public in the figures)
t2 = pd.read_csv(config.DATA / "table2.csv")
sp = pd.read_csv(config.DATA / "split.csv")[["flight", "role"]]
t2 = t2.merge(sp, on="flight")
test = t2[t2["role"] == "test"]
eng = test[test["fault_type"] == "engine"]
fl = eng.groupby("flight")["label"].sum().idxmax()
g1 = test[test["flight"] == fl].sort_values("t")
cols = [c for c in ["flight", "t", "t_rel_fault", "label", "vel_lin_z__mean5", "cause_rc_out_2__mean5", "cmd_frozen_any"] if c in g1.columns]
g1[g1["t_rel_fault"].between(-3, 8)][cols].round(3).to_csv(OUT / "fig1_rows.csv", index=False)
fl2 = test[test["fault_type"] == "rudder"]["flight"].iloc[0]
g2 = test[test["flight"] == fl2].sort_values("t")
cols2 = [c for c in ["flight", "t", "t_rel_fault", "label", "cmd_frozen_any"] if c in g2.columns]
g2[g2["t_rel_fault"].between(-3, 5)][cols2].round(3).to_csv(OUT / "fig2_rows.csv", index=False)

# ---- fig8: per fold mean over seeds (the dots) and the three means (the diamonds)
mc = pd.read_csv(config.PROJECT / "results" / "model_comparison.csv")
fm = mc.groupby(["model", "fold"])[["f1", "rec_eng", "fa_s_per_min"]].mean().round(3).reset_index()
fm.to_csv(OUT / "fig8_fold_means.csv", index=False)
mc[(mc["fold"] == mc["fold"].iloc[0]) & (mc["model"] == "forest")][["model", "fold", "seed", "f1", "rec_eng", "fa_s_per_min"]].round(3).to_csv(OUT / "fig8_seed_rows.csv", index=False)
print("written to", OUT)
for f_ in sorted(OUT.glob("fig[1-8]_*.csv")):
    print(" ", f_.name)
