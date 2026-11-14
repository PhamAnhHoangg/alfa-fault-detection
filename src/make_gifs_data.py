"""GIFs that show where the numbers come from, with spreadsheet-style tables of real ALFA rows.

Needs the small CSV snippets in docs/gif_data/ (made by src/export_gif_data.py).
Run:  python src/make_gifs_data.py
Writes figures/gif3_split.gif, gif5_numbers.gif, gif6_fig7_cell.gif and
figures/walkthrough_numbers.mp4 (gif5 and gif6 joined).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

import make_gifs as mg
from make_gifs import ORANGE, base, to_image

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "docs" / "gif_data"
MONO = "DejaVu Sans Mono"
HI = {"in": ("#fde047", "#ca8a04"), "out": ("#86efac", "#16a34a"), "aux": ("#bfdbfe", "#2563eb")}
W_IN, H_IN = 12.74, 5.29          # axes size in inches: data units are inches


# ------------------------------------------------------------------ spreadsheet helper
def sheet(ax, x0, y0, headers, rows, widths, fmts=None, hi=None, new_cols=(), row_h=0.27,
          head_h=0.32, fs=9, rn_start=1, grey_cols=(), gap_after=None):
    """Draw a spreadsheet. widths[0] is the row-number column. hi maps (row, col) to in/out/aux.
    Returns the y of the bottom edge."""
    hi = hi or {}
    fmts = fmts or [str] * len(headers)
    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + w)
    y = y0
    # column letters
    ax.add_patch(Rectangle((xs[0], y), xs[-1] - xs[0], 0.2, fc="#e5e7eb", ec="#d1d5db", lw=0.6))
    for c in range(len(headers)):
        ax.text((xs[c + 1] + xs[c + 2]) / 2, y + 0.1, chr(65 + c), ha="center", va="center",
                fontsize=7.5, color="#6b7280")
    y += 0.2
    # header row
    ax.add_patch(Rectangle((xs[0], y), widths[0], head_h, fc="#e5e7eb", ec="#d1d5db", lw=0.6))
    for c, h in enumerate(headers):
        new = c in new_cols
        ax.add_patch(Rectangle((xs[c + 1], y), widths[c + 1], head_h, fc="#f3f4f6",
                               ec=ORANGE if new else "#9ca3af", lw=2.6 if new else 0.8,
                               zorder=3 if new else 1))
        ax.text((xs[c + 1] + xs[c + 2]) / 2, y + head_h / 2, h, ha="center", va="center",
                fontsize=fs - 0.5, fontweight="bold", color="#111827", zorder=4,
                linespacing=1.15)
    y += head_h
    top_rows = y
    n = 0
    for r, row in enumerate(rows):
        gap = row == "gap"
        ax.add_patch(Rectangle((xs[0], y), widths[0], row_h, fc="#e5e7eb", ec="#d1d5db", lw=0.6))
        ax.text(xs[0] + widths[0] / 2, y + row_h / 2, "..." if gap else str(rn_start + n),
                ha="center", va="center", fontsize=fs - 1, color="#6b7280")
        for c in range(len(headers)):
            ax.add_patch(Rectangle((xs[c + 1], y), widths[c + 1], row_h, fc="white",
                                   ec="#d1d5db", lw=0.6))
            if gap:
                ax.text((xs[c + 1] + xs[c + 2]) / 2, y + row_h / 2, "...", ha="center",
                        va="center", fontsize=fs, color="#9ca3af")
                continue
            val = row[c]
            txt = fmts[c](val) if not isinstance(val, str) else val
            left = isinstance(val, str)
            ax.text(xs[c + 1] + (0.07 if left else widths[c + 1] - 0.07), y + row_h / 2, txt,
                    ha="left" if left else "right", va="center", fontsize=fs,
                    color="#9ca3af" if c in grey_cols else "#111827", zorder=2)
        if not gap:
            n += 1
        y += row_h
    # highlighted cells on top
    for (r, c), key in hi.items():
        fc, ec = HI[key]
        yy = top_rows + r * row_h
        ax.add_patch(Rectangle((xs[c + 1], yy), widths[c + 1], row_h, fc=fc, ec=ec, lw=2.4,
                               zorder=5))
        val = rows[r][c]
        txt = fmts[c](val) if not isinstance(val, str) else val
        left = isinstance(val, str)
        ax.text(xs[c + 1] + (0.07 if left else widths[c + 1] - 0.07), yy + row_h / 2, txt,
                ha="left" if left else "right", va="center", fontsize=fs, fontweight="bold",
                color="#111827", zorder=6)
    return y


def panel(ax, x, y, w, h, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                fc="white", ec="#cbd5e1", lw=1.4, zorder=1))
    ax.add_patch(Rectangle((x, y + 0.05), 0.07, h - 0.1, fc="#2563eb", ec="none", zorder=2))
    total = sum(kw.get("dy", kw.get("fs", 11) * 1.85 / 72) for _, kw in lines)
    widest = max((len(s) * kw.get("fs", 11) * (0.62 if kw.get("mono") else 0.56) / 72 for s, kw in lines), default=0)
    k = min(1.0, (h - 0.4) / total if total else 1.0, (w - 0.45) / widest if widest else 1.0)  # shrink only to fit
    cur = y + 0.28
    for s, kw in lines:
        fs = kw.get("fs", 11) * k
        ax.text(x + 0.25, cur, s, ha="left", va="center", fontsize=fs,
                fontweight="bold" if kw.get("bold") else "normal", color=kw.get("color", "#1f2937"),
                family=MONO if kw.get("mono") else "DejaVu Sans", zorder=3)
        cur += kw.get("dy", kw.get("fs", 11) * 1.85 / 72) * k


PANEL_SCALE = 4.4  # side-panel text scale, so the explanation stays readable when the GIF is shown small


def Hd(s):
    return (s, dict(fs=12.5 * PANEL_SCALE, bold=True, color="#1e3a8a", dy=0.34 * PANEL_SCALE))


def Tx(s, dy=None):
    return (s, dict(fs=10.5 * PANEL_SCALE, **({"dy": dy * PANEL_SCALE} if dy else {})))


def Fm(s):
    return (s, dict(fs=10 * PANEL_SCALE, mono=True, color="#92400e"))


def Rs(s):
    return (s, dict(fs=12.5 * PANEL_SCALE, mono=True, bold=True, color="#15803d", dy=0.34 * PANEL_SCALE))


def Sp(dy=0.12):
    return ("", dict(fs=6, dy=dy))


f4 = lambda x: f"{x:.4f}"
par = lambda x: f"({x:.4f})"


def scene(title, step, n, cap, size=(13, 7.2)):
    return base(title, step, n, cap, size, 100, (0, W_IN), (H_IN, 0))


# ================================================================== GIF 5: where the numbers come from
P = pd.read_csv(D / "per_second_example.csv")
R = pd.read_csv(D / "raw_readings_one_second.csv")
INFO = pd.read_csv(D / "example_info.csv").iloc[0]
FAULT = float(INFO["fault_sec"])
FLIGHT = INFO["flight"]
X = P.set_index("t")["alt_error_mean"]
TS = list(P["t"])
COLW = {"t": 0.6, "alt_error_mean": 1.55, "mean5": 1.2, "std5": 1.05, "delta10": 1.05, "label": 0.7}
HEAD = {"t": "t", "alt_error_mean": "alt_error\n(per-second mean)", "mean5": "alt_error\n__mean5",
        "std5": "alt_error\n__std5", "delta10": "alt_error\n__delta10", "label": "label"}
FMT = {"t": str, "alt_error_mean": f4, "mean5": f4, "std5": f4, "delta10": f4, "label": str}


def raw_scene(step, n, cap, mark, lines):
    fig, ax = scene("Where the numbers come from", step, n, cap)
    rows = R.values.tolist()
    hi = {(r, 1): "in" for r in range(len(rows))} if mark else {}
    sheet(ax, 0.2, 0.05, ["time_s", "field.alt_error"], rows, [0.5, 1.6, 1.7],
          [lambda v: f"{v:.3f}", f4], hi=hi, row_h=0.24, head_h=0.3, fs=8.5, rn_start=2)
    panel(ax, 4.4, 0.05, 8.2, 4.95, lines)
    return to_image(fig)


def sec_scene(step, n, cap, visible, hi_by, new_cols, lines, grey=()):
    fig, ax = scene("Where the numbers come from", step, n, cap)
    rows = [[int(row[c]) if c in ("t", "label") else row[c] for c in visible] for _, row in P.iterrows()]
    hi = {}
    for (t, col), key in hi_by.items():
        if col in visible:
            hi[(TS.index(t), visible.index(col))] = key
    widths = [0.5] + [COLW[c] for c in visible]
    bottom = sheet(ax, 0.2, 0.05, [HEAD[c] for c in visible], rows, widths,
                   [FMT[c] for c in visible], hi=hi, rn_start=2,
                   new_cols=[visible.index(c) for c in new_cols], row_h=0.265, head_h=0.42,
                   grey_cols=[visible.index(c) for c in grey if c in visible])
    left = 0.2 + sum(widths) + 0.3
    panel(ax, left, 0.05, W_IN - left - 0.15, bottom - 0.05, lines)
    return to_image(fig)


def build_gif5():
    base_cols = ["t", "alt_error_mean"]
    w5 = [par(X.loc[t]) for t in range(112, 117)]
    mean116 = X.loc[112:116].mean()
    mean117 = X.loc[113:117].mean()
    dev = [X.loc[t] - mean116 for t in range(112, 117)]
    ss = sum(d * d for d in dev)
    raw_mean = R["alt_error"].mean()
    d10 = X.loc[116] - X.loc[106]
    n = 9
    imgs = []

    imgs.append(raw_scene(0, n,
        "The aircraft logs the signal alt_error about 19 times inside second 115.\n"
        "This is the raw file, one row per reading.", False, [
            Hd("Raw file"), Tx(f"flight: {FLIGHT}"),
            Tx("file: ...-mavros-nav_info-errors.csv"), Tx("column: field.alt_error"), Sp(),
            Tx(f"These {len(R)} readings all fall inside second t = 115"),
            Tx("(time 115.0 to 116.0 s after the start of the flight)."), Sp(),
            Tx("(the first row of the real file is a header, so the data starts in row 2)")]))

    nums = " + ".join(par(v) for v in R["alt_error"].head(2)) + " + ... + " + par(R["alt_error"].iloc[-1])
    imgs.append(raw_scene(1, n,
        "Step 1: add up the 19 readings and divide by 19.\n"
        "That average is the value for second t = 115.", True, [
            Hd("Average of one second"), Fm(f"({nums}) / {len(R)}"), Sp(),
            Tx(f"sum of the {len(R)} readings = {R['alt_error'].sum():.4f}"),
            Rs(f"{R['alt_error'].sum():.4f} / {len(R)} = {f4(raw_mean)}"), Sp(),
            Tx("This number goes into the per-second table"),
            Tx("as the value for t = 115.")]))

    imgs.append(sec_scene(2, n,
        "Each second of the flight becomes one row: a time t and one averaged value.\n"
        "This is the per-second table that 01_build_table.py works with.", base_cols,
        {(115, "alt_error_mean"): "out"}, ["alt_error_mean"], [
            Hd("One number per second"), Rs(f"t = 115   ->   {f4(X.loc[115])}"), Sp(),
            Tx("Same recipe for every second of the flight:"),
            Tx("average all readings inside that second."), Sp(),
            Tx("(missing seconds are filled from the last known value,"),
            Tx("for at most 5 seconds)")]))

    imgs.append(sec_scene(3, n,
        "mean5 at t = 116: the average of the last five seconds, 112 to 116.\n"
        "The five yellow cells are averaged into the green cell.", base_cols + ["mean5"],
        {**{(t, "alt_error_mean"): "in" for t in range(112, 117)}, (116, "mean5"): "out"},
        ["mean5"], [
            Hd("mean5 at t = 116"), Fm("(x112 + x113 + x114 + x115 + x116) / 5"), Sp(),
            Fm("(" + " + ".join(w5[:3])), Fm(" + " + " + ".join(w5[3:]) + ") / 5"), Sp(),
            Rs("= " + f4(mean116)), Sp(),
            Tx("A rolling window of 5 seconds that ends at t.")]))

    imgs.append(sec_scene(4, n,
        "One row further: mean5 at t = 117 uses seconds 113 to 117.\n"
        "The window moves down by one second.", base_cols + ["mean5"],
        {**{(t, "alt_error_mean"): "in" for t in range(113, 118)}, (117, "mean5"): "out",
         (116, "mean5"): "aux"}, [], [
            Hd("mean5 at t = 117"), Fm("(x113 + x114 + x115 + x116 + x117) / 5"), Sp(),
            Rs("= " + f4(mean117)), Sp(),
            Tx("The row for t = 116 (blue) keeps its own value."),
            Tx("Every row has its own window.")]))

    imgs.append(sec_scene(5, n,
        "std5 at t = 116 measures how much those same five values spread out.\n"
        "It is the sample standard deviation (divide by n - 1 = 4).",
        base_cols + ["mean5", "std5"],
        {**{(t, "alt_error_mean"): "in" for t in range(112, 117)}, (116, "mean5"): "aux",
         (116, "std5"): "out"}, ["std5"], [
            Hd("std5 at t = 116"), Tx(f"mean5 = {f4(mean116)}, the differences x - mean5:"),
            Fm("  " + ", ".join(f4(d) for d in dev[:3])), Fm("  " + ", ".join(f4(d) for d in dev[3:])),
            Tx(f"squares add up to {f4(ss)}"), Sp(),
            Fm(f"std5 = sqrt( {f4(ss)} / 4 )"), Rs("= " + f4(np.sqrt(ss / 4))), Sp(),
            Tx("Small number: the five seconds were similar.")]))

    imgs.append(sec_scene(6, n,
        "delta10 at t = 116: the value now minus the value 10 seconds earlier.\n"
        "It shows how much the signal changed over 10 seconds.",
        base_cols + ["mean5", "std5", "delta10"],
        {(116, "alt_error_mean"): "in", (106, "alt_error_mean"): "in", (116, "delta10"): "out"},
        ["delta10"], [
            Hd("delta10 at t = 116"), Fm("x116 - x106"), Sp(),
            Fm(f"{par(X.loc[116])} - {par(X.loc[106])}"), Rs("= " + f4(d10)), Sp(),
            Tx("Row 106 is 10 rows above row 116.")]))

    imgs.append(sec_scene(7, n,
        f"The label: fault_sec = {FAULT:.3f} s. A second gets label 1 once it ends after the fault started.\n"
        "t = 115 ends at 116, before the fault: 0.  t = 116 ends at 117, after the fault: 1.",
        base_cols + ["mean5", "std5", "delta10", "label"],
        {(115, "label"): "out", (116, "label"): "out"}, ["label"], [
            Hd("label"), Tx(f"fault_sec = {FAULT:.3f}"),
            Tx("(from the failure_status file of the flight)"), Sp(),
            Fm(f"t = 115:  116 > {FAULT:.3f} ?  no   -> label 0"),
            Fm(f"t = 116:  117 > {FAULT:.3f} ?  yes  -> label 1"), Sp(),
            Tx("The rule is label = 1 when  t + 1 > fault_sec.")]))

    keep = ["t", "mean5", "std5", "delta10", "label"]
    imgs.append(sec_scene(8, n,
        "table.csv keeps only the rolling columns and the label, not the per-second average.\n"
        "The same is done for 10 signals, which gives 30 rolling columns.",
        keep, {(116, c): "out" for c in keep}, [], [
            Hd("table.csv, row t = 116"), Tx("t, mean5, std5, delta10 and label are saved."), Sp(),
            Tx("The per-second average (alt_error) is not saved."), Sp(),
            Tx("Ten signals x 3 rolling columns = 30 columns per row,"),
            Tx("one row for every second of every flight."), Sp(),
            Tx("Rows with t < 10 are dropped from the table.")]))
    return imgs


# ================================================================== GIF 6: the fig7 cell
T = pd.read_csv(D / "fig7_trace_head.csv").head(8)
S = pd.read_csv(D / "fig7_trace_sums.csv").iloc[0]
COLS7 = ["t", "label", "x", "x_mean_of_flight", "y_mean_of_flight", "x_tilde", "y_tilde", "xy",
         "x2", "y2"]
HEAD7 = {"t": "t", "label": "y\n(label)", "x": "x\n(alt_error__mean5)", "x_mean_of_flight": "mean x\nof flight",
         "y_mean_of_flight": "mean y\nof flight", "x_tilde": "x~ =\nx - mean x", "y_tilde": "y~ =\ny - mean y",
         "xy": "x~ * y~", "x2": "x~ squared", "y2": "y~ squared"}
W7 = {"t": 0.6, "label": 0.75, "x": 1.65, "x_mean_of_flight": 1.2, "y_mean_of_flight": 1.15, "x_tilde": 1.2,
      "y_tilde": 1.2, "xy": 1.1, "x2": 1.2, "y2": 1.15}
FMT7 = {"t": str, "label": str, "x": f4, "x_mean_of_flight": f4, "y_mean_of_flight": f4,
        "x_tilde": f4, "y_tilde": f4, "xy": f4, "x2": f4, "y2": f4}


def fig7_scene(step, n, cap, visible, hi_by, new_cols, lines, show_sum=False):
    fig, ax = scene("Where r = 0.65 comes from (fig7)", step, n, cap)
    rows = []
    for _, r in T.iterrows():
        rows.append([int(r[c]) if c in ("t", "label") else float(r[c]) for c in visible])
    rows.append("gap")
    if show_sum:
        sumrow = []
        for c in visible:
            sumrow.append({"xy": float(S["Sxy"]), "x2": float(S["Sxx"]), "y2": float(S["Syy"])}.get(c, ""))
        rows.append(sumrow)
    hi = {}
    for (rr, col), key in hi_by.items():
        if col in visible:
            hi[(rr, visible.index(col))] = key
    widths = [0.5] + [W7[c] for c in visible]
    fm = [FMT7[c] if c not in ("xy", "x2", "y2") else (lambda v: f"{v:.3f}" if v else "") for c in visible]
    fm = [FMT7[c] for c in visible]
    bottom = sheet(ax, 0.2, 0.05, [HEAD7[c] for c in visible], rows, widths,
                   [(lambda v, f=f: f(v) if v != "" else "") for f in fm], hi=hi,
                   new_cols=[visible.index(c) for c in new_cols], row_h=0.28, head_h=0.46, fs=9,
                   rn_start=2)
    if show_sum:
        # label the sum row
        ytxt = bottom - 0.14
        ax.text(0.2 + widths[0] + 0.07, ytxt, f"SUM of all {int(S['rows'])} rows", ha="left", va="center",
                fontsize=9.5, fontweight="bold", color="#15803d", zorder=7)
    panel(ax, 0.2, bottom + 0.2, W_IN - 0.35, H_IN - bottom - 0.25, lines)
    return to_image(fig)


def fig7_image_scene(step, n, cap):
    fig, ax = scene("Where r = 0.65 comes from (fig7)", step, n, cap)
    img = Image.open(ROOT / "figures" / "fig7_label_correlation.png").convert("RGB")
    iw, ih = img.size
    h = 5.1
    w = h * iw / ih
    ax.imshow(np.asarray(img), extent=(0.3, 0.3 + w, 0.1 + h, 0.1), zorder=1)
    ax.set_aspect("auto")
    ax.set_xlim(0, W_IN)
    ax.set_ylim(H_IN, 0)
    x0, x1, y0, y1 = 222, 362, 53, 113          # the 0.65 cell, measured in the png
    ax.add_patch(Rectangle((0.3 + w * x0 / iw, 0.1 + h * y0 / ih), w * (x1 - x0) / iw,
                           h * (y1 - y0) / ih, fc="none", ec=ORANGE, lw=3.5, zorder=5))
    left = 0.3 + w + 0.4
    panel(ax, left, 0.1, W_IN - left - 0.15, 5.1, [
        Hd("The cell in fig7"), Tx("row: alt_error__mean5"), Tx("column: engine"), Sp(),
        Rs("r = 0.65"), Sp(),
        Tx("Positive r: inside a flight, x tends to be"), Tx("above its own average when the label is 1."), Sp(),
        Tx("The colour scale stops at 0.6, so 0.65"), Tx("shows as the darkest red."), Sp(),
        Tx("The other 27 cells are worked out the same way."), Tx("(the mean5 cells, 7 signals x 4 fault types)")])
    return to_image(fig)


def build_gif6():
    r0 = T.iloc[0]
    n = 9
    imgs = []
    v0 = ["t", "label", "x"]
    imgs.append(fig7_scene(0, n,
        "These are the first rows of the data behind the fig7 cell alt_error__mean5 / engine.\n"
        f"x is the feature alt_error__mean5, y is the label. {int(S['rows'])} rows from {int(S['flights'])} engine-fault flights.",
        v0, {}, ["x"], [
            Hd("Engine-fault flights, dev part"),
            Tx(f"{int(S['rows'])} rows (seconds) from {int(S['flights'])} flights."),
            Tx("Each row: x = alt_error__mean5, y = label (0 normal, 1 fault)."), Sp(),
            Tx(f"Shown: the first flight, {FLIGHT}, from t = 10.")]))
    v1 = v0 + ["x_mean_of_flight", "y_mean_of_flight"]
    imgs.append(fig7_scene(1, n,
        "Each flight gets its own average of x and of y.\n"
        "Flight 1 has 123 rows, 17 of them with label 1, so its mean of y is 17 / 123 = 0.1382.",
        v1, {(r, c): "aux" for r in range(len(T)) for c in ("x_mean_of_flight", "y_mean_of_flight")},
        ["x_mean_of_flight", "y_mean_of_flight"], [
            Hd("Why a flight average?"),
            Tx("Every flight has its own offset (this one sits near x = -13.28)."),
            Tx("We want changes inside a flight, so each flight's own average is removed."), Sp(),
            Fm(f"mean y = 17 / 123 = {r0['y_mean_of_flight']:.4f}")]))
    v2 = v1 + ["x_tilde"]
    imgs.append(fig7_scene(2, n,
        "x~ is x minus its flight average.\n"
        f"First row: {r0['x']:.4f} - ({r0['x_mean_of_flight']:.4f}) = {r0['x_tilde']:.4f}.",
        v2, {(0, "x"): "in", (0, "x_mean_of_flight"): "in", (0, "x_tilde"): "out"}, ["x_tilde"], [
            Hd("x~ (x tilde)"), Fm(f"{par(r0['x'])} - {par(r0['x_mean_of_flight'])}"),
            Rs("= " + f4(r0["x_tilde"]))]))
    v3 = v2 + ["y_tilde"]
    imgs.append(fig7_scene(3, n,
        "y~ is y minus its flight average.\n"
        f"First row: 0 - {r0['y_mean_of_flight']:.4f} = {r0['y_tilde']:.4f}.",
        v3, {(0, "label"): "in", (0, "y_mean_of_flight"): "in", (0, "y_tilde"): "out"}, ["y_tilde"], [
            Hd("y~ (y tilde)"), Fm(f"0 - {par(r0['y_mean_of_flight'])}"), Rs("= " + f4(r0["y_tilde"]))]))
    v4 = v3 + ["xy"]
    imgs.append(fig7_scene(4, n,
        "Multiply the two: x~ * y~.\n"
        f"First row: ({r0['x_tilde']:.4f}) * ({r0['y_tilde']:.4f}) = {r0['xy']:.4f}.",
        v4, {(0, "x_tilde"): "in", (0, "y_tilde"): "in", (0, "xy"): "out"}, ["xy"], [
            Hd("x~ * y~"), Fm(f"{par(r0['x_tilde'])} * {par(r0['y_tilde'])}"), Rs("= " + f4(r0["xy"])), Sp(),
            Tx("Positive when x and y are on the same side of their averages.")]))
    v5 = v4 + ["x2", "y2"]
    imgs.append(fig7_scene(5, n,
        "Also square each one: x~ squared and y~ squared.\n"
        f"First row: {r0['x2']:.4f} and {r0['y2']:.4f}.",
        v5, {(0, "x_tilde"): "in", (0, "x2"): "out", (0, "y_tilde"): "in", (0, "y2"): "out"},
        ["x2", "y2"], [Hd("Squares"), Fm(f"x~ squared = {par(r0['x_tilde'])}^2 = {f4(r0['x2'])}"),
                       Fm(f"y~ squared = {par(r0['y_tilde'])}^2 = {f4(r0['y2'])}"), Sp(),
                       Tx("These measure how big the changes are.")]))
    sums_hi = {(len(T) + 1, "xy"): "out", (len(T) + 1, "x2"): "out", (len(T) + 1, "y2"): "out"}
    imgs.append(fig7_scene(6, n,
        f"Now add up every column over all {int(S['rows'])} rows.\n"
        f"Sxy = {S['Sxy']:.3f},  Sxx = {S['Sxx']:.3f},  Syy = {S['Syy']:.3f}.",
        v5, sums_hi, [], [
            Hd("Three sums"), Fm(f"Sxy = sum of x~ * y~       = {S['Sxy']:.3f}"),
            Fm(f"Sxx = sum of x~ squared   = {S['Sxx']:.3f}"), Fm(f"Syy = sum of y~ squared   = {S['Syy']:.3f}")],
        show_sum=True))
    den = float(np.sqrt(S["Sxx"] * S["Syy"]))
    imgs.append(fig7_scene(7, n,
        f"r = Sxy / sqrt(Sxx * Syy) = {S['Sxy']:.3f} / {den:.2f} = {S['Sxy'] / den:.2f}.",
        v5, sums_hi, [], [
            Hd("The correlation r"), Fm("r = Sxy / sqrt( Sxx * Syy )"),
            Fm(f"  = {S['Sxy']:.3f} / sqrt( {S['Sxx']:.3f} * {S['Syy']:.3f} )"),
            Fm(f"  = {S['Sxy']:.3f} / {den:.2f}"), Rs(f"  = {S['Sxy'] / den:.2f}")], show_sum=True))
    imgs.append(fig7_image_scene(8, n,
        "That is the dark red cell in fig7: alt_error__mean5 and engine, r = 0.65.\n"
        "The colour scale stops at 0.6, so 0.65 shows as the darkest red."))
    return imgs


# ================================================================== GIF 3 with real split tables
def build_gif3():
    SH = pd.read_csv(D / "split_head.csv")
    SC = pd.read_csv(D / "split_counts.csv")
    caps = mg.SPLIT_CAPS
    n = len(caps) + 2
    imgs = [mg.draw_split(0, 0, n, caps[0]), mg.draw_split(1, 1, n, caps[1])]

    fig, ax = base("How the data is split", 2, n,
                   "split.csv lists every flight with its role: cv0 to cv4, or test.\n"
                   "(The first 14 rows are shown. rows = seconds in the table, fault_rows = seconds with label 1.)",
                   (12, 6.4), 100, (0, 12), (5.0, 0))
    rows = [[r.flight, r.fault_type, r.role, int(r.rows), int(r.fault_rows)] for r in SH.itertuples()]
    hi = {(i, 2): "out" if rows[i][2] != "test" else "in" for i in range(len(rows))}
    sheet(ax, 0.2, 0.05, ["flight", "fault_type", "role", "rows", "fault_rows"], rows,
          [0.45, 4.3, 1.2, 0.9, 0.9, 1.3], [str, str, str, str, str], hi=hi, row_h=0.28, head_h=0.32,
          fs=8.5)
    imgs.append(to_image(fig))

    dev = SC[SC.role != "test"]
    tot_f, tot_s = int(SC.flights.sum()), int(SC.sessions.sum())
    fig, ax = base("How the data is split", 3, n,
                   f"Counted from split.csv: dev has {' + '.join(str(v) for v in dev.flights)} = {int(dev.flights.sum())} flights, "
                   f"test has {int(SC[SC.role == 'test'].flights.iloc[0])}.\n"
                   f"{tot_s} sessions in all. Whole sessions stay together.",
                   (12, 6.4), 100, (0, 12), (5.0, 0))
    rows = [[r.role, int(r.flights), int(r.sessions)] for r in SC.itertuples()]
    rows += [["dev total", int(dev.flights.sum()), int(dev.sessions.sum())],
             ["all", tot_f, tot_s]]
    hi = {(len(SC), 1): "out", (len(SC) + 1, 1): "out", (len(SC), 2): "out", (len(SC) + 1, 2): "out"}
    sheet(ax, 0.2, 0.05, ["role", "flights", "sessions"], rows, [0.45, 1.6, 1.2, 1.2],
          [str, str, str], hi=hi, row_h=0.34, head_h=0.34, fs=10)
    imgs.append(to_image(fig))

    for k in range(2, len(caps)):
        imgs.append(mg.draw_split(k, k + 2, n, caps[k]))
    return imgs


if __name__ == "__main__":
    g3 = build_gif3()
    mg.save_gif(g3, "gif3_split.gif")
    f5, d5 = mg.save_gif(build_gif5(), "gif5_numbers.gif")
    f6, d6 = mg.save_gif(build_gif6(), "gif6_fig7_cell.gif")
    mg.save_mp4(f5 + f6, d5 + d6, "walkthrough_numbers.mp4")
