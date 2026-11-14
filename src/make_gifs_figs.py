"""Step-by-step GIFs for the result figures (fig1 to fig6 and fig8).

Run:  python src/make_gifs_figs.py
Reads  figures/fig*.png  and the real-data snippets in docs/gif_data/fig*_*.csv
       (written by src/export_fig_data.py)
Writes figures/gif_fig1.gif ... gif_fig6.gif, gif_fig8.gif

Every GIF has the same order:
  1. look at the finished figure
  2. real rows that the numbers come from (highlighted cells)
  3. the formula worked out with those numbers
  4. back to the figure: where the result sits
  5. why we do it this way (short)
3 seconds per step, short cross-fade between steps.
"""
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FIG, DATA = ROOT / "figures", ROOT / "docs" / "gif_data"
W, H = 1280, 620
HEADER_H, BAR_H = 56, 10
AREA = (20, HEADER_H + 16, 800, H - BAR_H - 16)
PANEL_X = 826
STEP_MS, LAST_MS, FADE_N, FADE_MS = 3000, 5000, 6, 50
ORANGE = "#f97316"
COLORS = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#db2777", "#0284c7", "#dc2626"]
YEL, BLU, ORA, GRN = "#fde68a", "#bfdbfe", "#fed7aa", "#bbf7d0"
FD = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf") + "/"  # DejaVu ships with matplotlib, so this works on Windows too


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(FD + name + ("-Bold.ttf" if bold else ".ttf"), size)


def wrap(draw, text, fnt, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            test = (cur + " " + word).strip()
            if draw.textlength(test, font=fnt) <= width:
                cur = test
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def csv(name):
    return pd.read_csv(DATA / name)


# ------------------------------------------------------------------ left-area drawers
def draw_fig(img, d, png, box):
    ax0, ay0, ax1, ay1 = AREA
    s = min((ax1 - ax0) / png.width, (ay1 - ay0) / png.height)
    pw, ph = int(png.width * s), int(png.height * s)
    px, py = ax0 + (ax1 - ax0 - pw) // 2, ay0 + (ay1 - ay0 - ph) // 2
    fig = png.resize((pw, ph), Image.LANCZOS).convert("RGB")
    bxs = []
    if box is not None:
        boxes = box if isinstance(box, list) else [box]
        bxs = [[int(v * s) for v in b] for b in boxes]
        out = Image.blend(fig, Image.new("RGB", fig.size, "white"), 0.72)
        for bx in bxs:
            out.paste(fig.crop(bx), (bx[0], bx[1]))
        fig = out
    img.paste(fig, (px, py))
    d.rectangle((px - 1, py - 1, px + pw, py + ph), outline="#cbd5e1", width=1)
    for bx in bxs:
        d.rectangle((px + bx[0], py + bx[1], px + bx[2], py + bx[3]), outline=ORANGE, width=4)


def draw_table(img, d, spec):
    """spec: file, header, rows, hl {(row, col): color}, widths optional."""
    ax0, ay0, ax1, ay1 = AREA
    d.rounded_rectangle((ax0, ay0, ax1, ay1), radius=10, fill="#ffffff", outline="#cbd5e1")
    d.text((ax0 + 14, ay0 + 10), spec["file"], font=font(17, bold=True), fill="#475569")
    fnt, fb = font(17, mono=True), font(17, bold=True, mono=True)
    header, rows, hl = spec["header"], spec["rows"], spec.get("hl", {})
    ncol = len(header)
    cw = [max(len(str(header[c])), *(len(str(r[c])) for r in rows)) * 11 + 22 for c in range(ncol)]
    scale = min(1.0, (ax1 - ax0 - 70) / sum(cw))
    cw = [int(w * scale) for w in cw]
    rh = min(34, (ay1 - ay0 - 90) // (len(rows) + 1))
    x0, y0 = ax0 + 14, ay0 + 44
    # column letters + header
    x = x0 + 36
    for c in range(ncol):
        d.rectangle((x, y0, x + cw[c], y0 + rh), fill="#e2e8f0", outline="#94a3b8")
        tx = header[c]
        d.text((x + 6, y0 + rh // 2 - 10), tx, font=fb, fill="#1f2937")
        x += cw[c]
    for r, row in enumerate(rows):
        y = y0 + rh * (r + 1)
        d.rectangle((x0, y, x0 + 36, y + rh), fill="#e2e8f0", outline="#94a3b8")
        d.text((x0 + 6, y + rh // 2 - 10), str(r + 1), font=fnt, fill="#475569")
        x = x0 + 36
        for c in range(ncol):
            col = hl.get((r, c)) or hl.get((None, c))
            d.rectangle((x, y, x + cw[c], y + rh), fill=col or "white", outline="#94a3b8",
                        width=3 if hl.get((r, c)) else 1)
            d.text((x + 6, y + rh // 2 - 10), str(row[c]), font=fnt, fill="#111827")
            x += cw[c]
    if spec.get("legend"):
        d.text((ax0 + 14, ay1 - 30), spec["legend"], font=font(16), fill="#475569")


def draw_text(img, d, lines):
    ax0, ay0, ax1, ay1 = AREA
    d.rounded_rectangle((ax0, ay0, ax1, ay1), radius=10, fill="#fffbeb", outline="#d97706", width=3)
    scale = 4.2 * W / 1280  # bigger explanation text, shrunk only if it would not fit the box
    while True:
        y, out = ay0 + 24, []
        for text, color, size in lines:
            f = font(max(12, round(size * scale)), bold=(color != "#1f2937"), mono=True)
            for line in wrap(d, text, f, ax1 - ax0 - 40):
                out.append((y, line, f, color))
                y += int(f.size * 1.5)
            y += 8
        if y <= ay1 - 10 or scale <= 0.8:
            break
        scale -= 0.05
    for y, line, f, color in out:
        d.text((ax0 + 20, y), line, font=f, fill=color)


def frame(png, title, steps, i):
    n = len(steps)
    kind, payload, head, text = steps[i]
    color = COLORS[i % len(COLORS)]
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, HEADER_H), fill="#1e3a8a")
    d.text((20, 14), title, font=font(24, True), fill="white")
    tag = f"Step {i + 1} of {n}"
    d.text((W - 20 - d.textlength(tag, font=font(22, True)), 16), tag, font=font(22, True), fill="#fde047")
    if kind == "fig":
        draw_fig(img, d, png, payload)
    elif kind == "table":
        draw_table(img, d, payload)
    else:
        draw_text(img, d, payload)
    pw2 = W - PANEL_X - 20
    d.rounded_rectangle((PANEL_X, HEADER_H + 16, PANEL_X + pw2, H - BAR_H - 16), radius=14,
                        fill="#f8fafc", outline=color, width=4)
    d.rounded_rectangle((PANEL_X, HEADER_H + 16, PANEL_X + pw2, HEADER_H + 62), radius=14, fill=color)
    hs = round(28 * W / 1280)
    while hs > 16 and d.textlength(head, font=font(hs, True)) > pw2 - 32:  # long titles shrink to fit
        hs -= 1
    d.text((PANEL_X + 16, HEADER_H + 39 - hs // 2), head, font=font(hs, True), fill="white")
    size, bottom = round(91 * W / 1280), H - BAR_H - 30
    while True:  # biggest size that still fits the panel
        fnt = font(size)
        lines = wrap(d, text, fnt, pw2 - 32)
        if HEADER_H + 80 + len(lines) * int(size * 1.45) <= bottom or size <= 17:
            break
        size -= 1
    y = HEADER_H + 80
    for line in lines:
        d.text((PANEL_X + 16, y), line, font=fnt, fill="#1f2937")
        y += int(size * 1.45)
    d.rectangle((0, H - BAR_H, W, H), fill="#e5e7eb")
    d.rectangle((0, H - BAR_H, int(W * (i + 1) / n), H), fill=color)
    return img


def build(name, png_name, title, steps):
    png = Image.open(FIG / png_name)
    frames = [frame(png, title, steps, i) for i in range(len(steps))]
    seq, dur = [], []
    for i, f in enumerate(frames):
        seq.append(f)
        dur.append(LAST_MS if i == len(frames) - 1 else STEP_MS)
        if i < len(frames) - 1:
            for k in range(1, FADE_N + 1):
                seq.append(Image.blend(f, frames[i + 1], k / (FADE_N + 1)))
                dur.append(FADE_MS)
    pal = [im.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for im in seq]
    out = FIG / name
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True, disposal=1)
    print(out.name, f"{out.stat().st_size / 1e6:.1f} MB", len(steps), "steps")


K, G, R, B = "#1f2937", "#15803d", "#b91c1c", "#1d4ed8"


def f3(v):
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1e4 else f"{v:.0f}"


# ------------------------------------------------------------------ fig 1
def gif1():
    t = csv("fig1_rows.csv")
    rows = [[int(r.t), f"{r.t_rel_fault:.1f}", int(r.label), f"{r.vel_lin_z__mean5:.3f}",
             f"{r.cause_rc_out_2__mean5:.1f}"] for r in t.itertuples()]
    header = ["t", "t_rel_fault", "label", "vel_lin_z__mean5", "throttle__mean5"]
    base = {(None, 3): None}
    s2 = dict(file="table2.csv, test flight 2018-09-11-11-56-30 (engine), rows t = 101 to 111", header=header, rows=rows,
              hl={(None, 3): BLU, (None, 4): ORA, (None, 1): "#f1f5f9"},
              legend="blue = panel 1 line, orange = panel 2 line, x axis = t_rel_fault")
    s3 = dict(file="the same rows: which seconds raise an alarm", header=header, rows=rows,
              hl={**{(r, 4): ORA for r in (5, 6, 7)}, **{(r, 3): BLU for r in (8, 9, 10)}},
              legend="orange rows 6 to 8: throttle low 3 in a row. blue rows 9 to 11: velocity low 3 in a row")
    return [
        ("fig", None, "1. Look at the figure",
         "One test engine flight. Panels 1 and 2 are signals over time, panel 3 shows when each detector is in alarm. "
         "0 on the x axis is the fault start."),
        ("table", s2, "2. Where the numbers come from",
         "Every point on the two lines is one row of the per-second table. Blue column: vertical velocity, 5 s mean. "
         "Orange column: throttle, 5 s mean. The x value is t_rel_fault. The label flips to 1 at t = 103."),
        ("table", s3, "3. Which rows trigger alarms",
         "Throttle rule: 1287.2, 1173.1, 1100.0 are all below the dashed line (about 1323). "
         "Sensor rule A: -0.903, -1.255, -0.893 are all at or below -0.833."),
        ("text", [("alarm_t = low_t AND low_(t-1) AND low_(t-2)", K, 22),
                  ("", K, 12),
                  ("throttle low:  t=106, 107, 108", ORA.replace("fed7aa", "c2410c"), 22),
                  ("  -> D2 alarm at t=108, 4.5 s after the fault", G, 22),
                  ("velocity low:  t=109, 110, 111", B, 22),
                  ("  -> A alarm at t=111, 7.5 s after the fault", G, 22),
                  ("", K, 12),
                  ("D2 = frozen_command OR throttle alarm", K, 20)],
         "4. The formula",
         "A rule says 'low' when its feature is under the threshold. An alarm needs 3 low seconds in a row. "
         "D2 also fires if a command channel is frozen."),
        ("fig", (232, 753, 1328, 962), "5. Back to the figure",
         "The ticks in panel 3 start at 4.5 s (D2) and 7.5 s (A), the same seconds we just found in the table."),
        ("text", [("Why 3 seconds in a row?", K, 24),
                  ("One noisy second should not raise an alarm.", K, 20),
                  ("", K, 12),
                  ("Why fit thresholds on development flights?", K, 24),
                  ("This is a test flight. The thresholds were chosen without seeing it.", K, 20),
                  ("", K, 12),
                  ("Why two detectors?", K, 24),
                  ("D2 reads commands, A reads sensors. The gap between them shows what each side sees.", K, 20)],
         "6. Why we do it", "Short reasons for the three choices."),
    ]


# ------------------------------------------------------------------ fig 2
def gif2():
    t = csv("fig2_rows.csv")
    rows = [[int(r.t), f"{r.t_rel_fault:.1f}", int(r.label), int(r.cmd_frozen_any)] for r in t.itertuples()]
    header = ["t", "t_rel_fault", "label", "cmd_frozen_any"]
    return [
        ("fig", None, "1. Look at the figure",
         "Test flight with rudder_zero and a left aileron failure. Three raw command channels (rc-out 3, 4, 5) from 20 s "
         "before to 30 s after the fault. Channels 3 and 4 go flat at 1500, channel 5 keeps moving."),
        ("text", [("raw file: ...-mavros-rc-out.csv", K, 20),
                  ("columns field.channels3, 4, 5 (PWM, microseconds)", K, 20),
                  ("", K, 12),
                  ("about 10 readings per second:", K, 22),
                  ("the lines in the figure are these readings", B, 22),
                  ("plotted against t_rel_fault", B, 22)],
         "2. Where the numbers come from",
         "The three lines are the raw rc-out readings of this flight. This is the only figure that uses raw readings "
         "instead of the per-second table."),
        ("table", dict(file="table2.csv, same flight, rows t = 114 to 121", header=header, rows=rows,
                       hl={(None, 3): YEL, **{(r, 3): GRN for r in (4, 5, 6, 7)}},
                       legend="cmd_frozen_any is 1 from t = 118 (rows shown)"),
         "3. The frozen flag",
         "From the raw readings 03_command_features.py builds cmd_frozen_any. It is 0 before the fault and 1.0 from "
         "t = 118, which is 1.9 s after the fault start."),
        ("text", [("per channel i, per second:", K, 22),
                  ("  frozen_i = (std of readings == 0)", B, 22),
                  ("  needs at least 2 readings", K, 20),
                  ("", K, 12),
                  ("frozen_i must hold 3 seconds in a row", K, 22),
                  ("", K, 12),
                  ("cmd_frozen_any = max over channels 1, 3, 4, 5", G, 22)],
         "4. The formula",
         "A channel with standard deviation 0 in a second did not move at all. Three such seconds in a row count as frozen."),
        ("fig", [(651, 130, 1280, 170), (651, 400, 1280, 430)], "5. Back to the figure",
         "These flat lines at 1500 are the frozen seconds that set the flag. Channel 5 moves, so it does not."),
        ("text", [("Why look for a frozen command?", K, 24),
                  ("A healthy flight controller keeps changing the commands. A failed channel stops.", K, 20),
                  ("", K, 12),
                  ("Why 3 seconds in a row?", K, 24),
                  ("A calm moment can have one still second.", K, 20)],
         "6. Why we do it", "The flag catches faults that the sensor features miss."),
    ]


# ------------------------------------------------------------------ fig 3
def gif3():
    t = csv("fig3_counts.csv")
    rows = [[r.fault_type, int(r.fault_seconds), int(r.raw_hits), int(r.alarm_hits)] for r in t.itertuples()]
    header = ["fault_type", "fault_seconds", "raw_hits", "alarm_hits"]
    lines = []
    for r in t.itertuples():
        lines.append((f"{r.fault_type}: {int(r.raw_hits)}/{int(r.fault_seconds)} = {r.raw_recall:.2f}   "
                      f"{int(r.alarm_hits)}/{int(r.fault_seconds)} = {r.alarm_recall:.2f}", K, 20))
    return [
        ("fig", None, "1. Look at the figure",
         "Random forest on development folds. Recall by fault type: the share of fault seconds that were detected. "
         "Light bar = raw per-second prediction, dark bar = alarm."),
        ("table", dict(file="oof_predictions.csv, rows with label = 1, counted per fault type", header=header, rows=rows,
                       hl={(None, 1): YEL, (None, 2): BLU, (None, 3): GRN},
                       legend="yellow = fault seconds, blue = raw hits, green = alarm hits"),
         "2. Where the numbers come from",
         "Each fault second is one row with label 1. We count how many were predicted fault (raw_hits) and how many "
         "were inside an alarm (alarm_hits). Predictions are out-of-fold."),
        ("text", [("recall = hits / fault seconds", K, 24), ("", K, 12),
                  ("      raw           alarm", "#475569", 20)] + lines,
         "3. The formula", "Each bar is one division. The bar labels in the figure are these results rounded to 2 decimals."),
        ("fig", (140, 150, 540, 545), "4. Back to the figure",
         "Engine: 225 / 301 = 0.75 and 185 / 301 = 0.61. Elevator has only 13 fault seconds, so its bars rest on very few seconds."),
        ("text", [("Why split by fault type?", K, 24),
                  ("One overall number hides that only engine faults are found.", K, 20), ("", K, 12),
                  ("Why two bars?", K, 24),
                  ("The alarm needs 3 seconds in a row. That removes false alarms but misses short detections.", K, 20)],
         "5. Why we do it", "The two bars show what the 3-in-a-row rule costs."),
    ]


# ------------------------------------------------------------------ fig 4
def gif4():
    c = csv("fig4_counts.csv").iloc[0]
    rows = [["normal", int(c.tn), int(c.fp), int(c.normal_total)], ["fault", int(c.fn), int(c.tp), int(c.fault_total)]]
    header = ["actual \\ alarm", "off", "on", "row total"]
    return [
        ("fig", None, "1. Look at the figure",
         "Confusion matrix of the alarm on development rows. Each row is an actual class and sums to 1."),
        ("table", dict(file="oof_predictions.csv, seconds counted by (label, alarm)", header=header, rows=rows,
                       hl={(0, 1): BLU, (0, 2): ORA, (1, 1): ORA, (1, 2): GRN},
                       legend="four counts, then each is divided by its row total"),
         "2. Where the numbers come from",
         "Every second of every development flight falls in one cell: normal or fault (label) and alarm off or on."),
        ("text", [("cell share = count / row total", K, 24), ("", K, 12),
                  (f"normal, off: {int(c.tn)} / {int(c.normal_total)} = {c.tn / c.normal_total:.3f} -> 0.99", K, 20),
                  (f"normal, on : {int(c.fp)} / {int(c.normal_total)} = {c.fp / c.normal_total:.3f} -> 0.01", K, 20),
                  (f"fault, off : {int(c.fn)} / {int(c.fault_total)} = {c.fn / c.fault_total:.3f} -> 0.62", K, 20),
                  (f"fault, on  : {int(c.tp)} / {int(c.fault_total)} = {c.tp / c.fault_total:.3f} -> 0.38", K, 20)],
         "3. The formula", "This is scikit-learn's normalize='true'. The 0.38 is the recall over all fault types together."),
        ("fig", [(393, 298, 636, 541), (150, 55, 393, 298)], "4. Back to the figure",
         "Bottom right is 0.38: caught fault seconds. Top left is 0.99: quiet normal seconds."),
        ("text", [("Why shares per row?", K, 24),
                  (f"There are {int(c.normal_total)} normal seconds and only {int(c.fault_total)} fault seconds.", K, 20),
                  ("Raw counts would hide how many faults are missed.", K, 20)],
         "5. Why we do it", "Dividing by the row total makes both classes comparable."),
    ]


# ------------------------------------------------------------------ fig 5
def gif5():
    t = csv("fig5_flight_rows.csv")
    info = csv("fig5_flight_info.csv").iloc[0]
    s = t[t.t.between(116, 125)]
    rows = [[int(r.t), f"{r.t_rel_fault:.1f}", int(r.label), f"{r.proba:.3f}", str(r.pred_raw), str(r.pred)] for r in s.itertuples()]
    header = ["t", "t_rel_fault", "label", "proba", "pred_raw", "pred"]
    raw_rows = [i for i, r in enumerate(s.itertuples()) if r.pred_raw]
    al_rows = [i for i, r in enumerate(s.itertuples()) if r.pred]
    return [
        ("fig", None, "1. Look at the figure",
         "One grey bar per development flight, from its first to its last second. The black line is the fault start. "
         "Red ticks are alarm seconds."),
        ("table", dict(file="oof_predictions.csv, flight 2018-07-18-15-53-31 (engine 1), t = 116 to 125", header=header, rows=rows,
                       hl={**{(r, 4): BLU for r in raw_rows}, (None, 3): "#f1f5f9"},
                       legend=f"whole flight: {int(info.rows)} rows, grey bar from {info.t_rel_min:.0f} to {info.t_rel_max:.0f} s"),
         "2. Where the numbers come from",
         "One bar is one flight. Each row has a fault probability (proba) from the forest that did not train on this flight. "
         "pred_raw is True when proba is at least 0.5."),
        ("table", dict(file="the same rows: the alarm needs 3 in a row", header=header, rows=rows,
                       hl={**{(r, 4): BLU for r in raw_rows}, **{(r, 5): GRN for r in al_rows}},
                       legend="pred is True only after three True pred_raw in a row"),
         "3. From probability to alarm",
         "pred_raw is True at t = 118 and 119, only 2 in a row, so no alarm. At t = 120 it is the third in a row: alarm. "
         "Each green cell is a red tick."),
        ("text", [("pred_raw_t = proba_t >= 0.5", K, 22),
                  ("pred_t     = pred_raw_t AND pred_raw_(t-1) AND pred_raw_(t-2)", K, 20), ("", K, 12),
                  ("t=118: 0.574, 0.638 are 2 seconds -> off", R, 20),
                  ("t=120: 0.574, 0.638, 0.551 -> alarm", G, 20),
                  ("t=121, 122 -> alarm", G, 20),
                  ("t=123: 0.456 -> off", R, 20), ("", K, 12),
                  (f"whole flight: {int(info.alarm_ticks)} alarm ticks", K, 22)],
         "4. The formula", "The tick is drawn at t_rel_fault of each True pred."),
        ("fig", (20, 98, 1240, 126), "5. Back to the figure",
         "This is the engine 1 bar: ticks at about 3.8, 4.8 and 5.8 s, and a second run from 11.8 to 15.8 s. 8 ticks in total."),
        ("text", [("Why out-of-fold?", K, 24),
                  ("The model never saw the flight it scores.", K, 20), ("", K, 12),
                  ("Why a timeline per flight?", K, 24),
                  ("The overall percentages hide when an alarm comes: before, right after or late.", K, 20)],
         "6. Why we do it", "The figure shows timing, which the bar chart cannot."),
    ]


# ------------------------------------------------------------------ fig 6
def gif6():
    t = csv("fig6_rows_head.csv")
    c = csv("fig6_cell_sums.csv").iloc[0]
    rows = [[f"{r.alt_error__std5:.4f}", f"{r.hud_altitude__std5:.4f}"] for r in t.itertuples()]
    header = ["alt_error__std5", "hud_altitude__std5"]
    den = (c.sxx * c.szz) ** 0.5
    return [
        ("fig", None, "1. Look at the figure",
         "Pearson correlation of every feature pair, development rows pooled. Red: rise together. Blue: one rises when the other falls."),
        ("table", dict(file=f"table.csv, first 8 of {int(c.n_rows)} development rows", header=header, rows=rows,
                       hl={(None, 0): YEL, (None, 1): BLU},
                       legend="one cell of the figure uses all 3279 rows of two columns"),
         "2. Where the numbers come from",
         "The cell for alt_error__std5 and hud_altitude__std5 comes from these two columns. Their values rise and fall together."),
        ("text", [("r = Sxy / sqrt(Sxx * Szz)", K, 26), ("", K, 12),
                  (f"mean x = {c.mean_x:.4f}   mean z = {c.mean_z:.4f}", K, 19),
                  (f"Sxy = sum((x - mean x)(z - mean z)) = {c.sxy:.2f}", B, 19),
                  (f"Sxx = sum((x - mean x)^2)           = {c.sxx:.2f}", B, 19),
                  (f"Szz = sum((z - mean z)^2)           = {c.szz:.2f}", B, 19), ("", K, 12),
                  (f"sqrt(Sxx * Szz) = {den:.2f}", K, 20),
                  (f"r = {c.sxy:.2f} / {den:.2f} = {c.r:.4f}", G, 24)],
         "3. The formula", "Sums run over all 3279 rows. The result is about 0.9985, which the figure shows as dark red."),
        ("fig", [(477, 132, 521, 176), (211, 398, 255, 442)], "4. Back to the figure",
         "The pair sits in the dark red cell, and its mirror across the diagonal. 08_correlation.py prints every pair with |r| above 0.9."),
        ("text", [("Why check this?", K, 24),
                  ("Two features with r near 1 carry almost the same information.", K, 20),
                  ("The second one adds very little new.", K, 20), ("", K, 12),
                  ("Why pool the rows?", K, 24),
                  ("The question is which columns repeat each other across all development flights.", K, 20)],
         "5. Why we do it", "It shows which features are close to duplicates."),
    ]


# ------------------------------------------------------------------ fig 8
def gif8():
    seeds = csv("fig8_seed_rows.csv")
    fm = csv("fig8_fold_means.csv")
    fo = fm[fm.model == "forest"].reset_index(drop=True)
    srows = [[r.model, r.fold, int(r.seed), f"{r.f1:.3f}"] for r in seeds.itertuples()]
    frows = [[r.model, r.fold, f"{r.f1:.3f}", f"{r.rec_eng:.3f}", f"{r.fa_s_per_min:.3f}"] for r in fo.itertuples()]
    m = fo[["f1", "rec_eng", "fa_s_per_min"]].mean()
    s0 = seeds.f1.mean()
    return [
        ("fig", None, "1. Look at the figure",
         "Forest, CatBoost and MLP on the same 5 development folds, 5 seeds each. Left: F1 per fold. Right: recall against false alarms."),
        ("table", dict(file="model_comparison.csv: forest, fold cv0, 5 seeds", header=["model", "fold", "seed", "f1"], rows=srows,
                       hl={(None, 3): YEL}, legend="one dot in the figure = the mean of these 5 seeds"),
         "2. Where the numbers come from",
         "Each model was trained 5 times with different seeds on every fold. The 5 F1 values of one fold are averaged to one dot."),
        ("table", dict(file="forest, mean over seeds, per fold (one row = one dot)", header=["model", "fold", "f1", "rec_eng", "fa_s_per_min"],
                       rows=frows, hl={(None, 2): YEL, (None, 3): BLU, (None, 4): ORA},
                       legend="yellow: left panel. blue and orange: right panel (y and x)"),
         "3. The dots of one model",
         "Left panel plots f1. Right panel plots false-alarm seconds per minute (x) against engine recall (y)."),
        ("text", [("F1 = 2 * precision * recall / (precision + recall)", K, 20), ("", K, 12),
                  (f"cv0 dot: (0.411+0.411+0.44+0.411+0.411)/5 = {s0:.3f}", K, 19),
                  (f"diamond: mean of 5 folds = {m.f1:.3f}", G, 22),
                  (f"big dot x = mean false alarms = {m.fa_s_per_min:.2f} s/min", B, 22),
                  (f"big dot y = mean engine recall = {m.rec_eng:.2f}", B, 22)],
         "4. The formula", "Diamonds and big dots are plain means over the five fold values."),
        ("fig", [(225, 345, 265, 380), (968, 400, 1018, 445)], "5. Back to the figure",
         f"Forest: diamond at {m.f1:.2f} on the left, big dot at ({m.fa_s_per_min:.1f}, {m.rec_eng:.2f}) on the right."),
        ("text", [("Why 5 seeds?", K, 24),
                  ("One training run can be lucky. The mean smooths that out.", K, 20), ("", K, 12),
                  ("Why show every fold?", K, 24),
                  ("The folds spread much more than the models differ. A single mean would hide it.", K, 20), ("", K, 12),
                  ("Why recall against false alarms?", K, 24),
                  ("More recall costs more false alarms. The plot shows that price.", K, 20)],
         "6. Why we do it", "Compare models on the same folds and the same threshold."),
    ]


def main():
    build("gif_fig1.gif", "fig1_engine_flight.png", "Fig 1: one engine flight, three detectors", gif1())
    build("gif_fig2.gif", "fig2_frozen_channels.png", "Fig 2: frozen control channels", gif2())
    build("gif_fig3.gif", "fig3_recall_by_type.png", "Fig 3: recall by fault type", gif3())
    build("gif_fig4.gif", "fig4_confusion_matrix.png", "Fig 4: confusion matrix", gif4())
    build("gif_fig5.gif", "fig5_alarm_timeline.png", "Fig 5: alarms per flight", gif5())
    build("gif_fig6.gif", "fig6_feature_correlation.png", "Fig 6: which features duplicate each other", gif6())
    build("gif_fig8.gif", "fig8_model_comparison.png", "Fig 8: model comparison", gif8())


if __name__ == "__main__":
    main()
