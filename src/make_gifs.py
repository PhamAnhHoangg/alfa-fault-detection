"""Draws the step-by-step explanation GIFs (no data needed).

Run:  python src/make_gifs.py
Writes figures/gif1_time_window.gif, gif2_pipeline.gif, gif3_split.gif, gif4_script_map.gif

Each GIF is a list of steps. Every step has a caption. Items appear at the step
where their level is reached, and the newest items are drawn with an orange edge.
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

INK, GREY, ORANGE = "#1f2937", "#6b7280", "#f97316"
HEAD = "#1e3a8a"
KIND = {  # fill, edge
    "data": ("#e0f2fe", "#0284c7"),
    "build": ("#bfdbfe", "#2563eb"),
    "hub": ("#e9d5ff", "#7c3aed"),
    "model": ("#bbf7d0", "#16a34a"),
    "report": ("#fde68a", "#d97706"),
    "out": ("#fbcfe8", "#db2777"),
    "plain": ("#f3f4f6", "#4b5563"),
    "test": ("#fecaca", "#dc2626"),
    "dev": ("#bfdbfe", "#2563eb"),
    "val": ("#fde047", "#ca8a04"),
}
STEP_MS, LAST_MS, FADE_N, FADE_MS = 3000, 5000, 6, 50


# ---------------------------------------------------------------- frame base
def base(title, step, n, cap, size, dpi, xlim, ylim):
    fig = plt.figure(figsize=size, dpi=dpi)
    fig.add_artist(Rectangle((0, 0.925), 1, 0.075, transform=fig.transFigure, fc=HEAD, ec="none"))
    fig.text(0.02, 0.962, title, color="white", fontsize=15, fontweight="bold", va="center")
    fig.text(0.69, 0.962, f"Step {step + 1} of {n}", color="white", fontsize=11, ha="right",
             va="center")
    fig.add_artist(Rectangle((0.70, 0.953), 0.28, 0.018, transform=fig.transFigure,
                             fc="#3b5bb5", ec="none"))
    fig.add_artist(Rectangle((0.70, 0.953), 0.28 * (step + 1) / n, 0.018,
                             transform=fig.transFigure, fc="#fbbf24", ec="none"))
    fig.add_artist(FancyBboxPatch((0.02, 0.02), 0.96, 0.135, transform=fig.transFigure,
                                  boxstyle="round,pad=0.002,rounding_size=0.01",
                                  fc="#fffbeb", ec="#f59e0b", lw=1.6))
    fig.add_artist(Rectangle((0.02, 0.02), 0.008, 0.135, transform=fig.transFigure,
                             fc="#f59e0b", ec="none"))
    fig.text(0.5, 0.087, cap, ha="center", va="center", fontsize=13.5, color=INK,
             linespacing=1.55)
    ax = fig.add_axes([0.01, 0.175, 0.98, 0.735])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fade(a, b, n=FADE_N):
    """Fade the diagram; swap caption and step counter sharply halfway."""
    w, h = a.size
    boxes = [(0, int(0.82 * h), w, h), (int(0.55 * w), 0, w, int(0.08 * h))]
    out = []
    for k in range(n):
        f = Image.blend(a, b, (k + 1) / (n + 1))
        src = a if k < n / 2 else b
        for box in boxes:
            f.paste(src.crop(box), box[:2])
        out.append(f)
    return out


def sequence(steps, extra=None):
    """steps: list of PIL images. extra: {i: [frames]} replaces the fade after step i."""
    extra = extra or {}
    frames, durs = [], []
    for i, img in enumerate(steps):
        frames.append(img)
        durs.append(LAST_MS if i == len(steps) - 1 else STEP_MS)
        if i == len(steps) - 1:
            break
        seq = extra.get(i) or fade(img, steps[i + 1])
        for f in seq:
            frames.append(f)
            durs.append(FADE_MS if i not in extra else 60)
    return frames, durs


def save_gif(steps, name, extra=None):
    frames, durs = sequence(steps, extra)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(OUT / name, save_all=True, append_images=pal[1:], duration=durs, loop=0,
                optimize=True)
    print(name, "frames:", len(frames), "size KB:", (OUT / name).stat().st_size // 1024)
    return frames, durs


def save_mp4(frames, durs, name):
    """Write an H.264 video from frames that each stay on screen for durs[i] ms."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lines = []
        for i, (f, d) in enumerate(zip(frames, durs)):
            fp = tmp / f"f{i:04d}.png"
            f.save(fp)
            lines.append(f"file '{fp.as_posix()}'")
            lines.append(f"duration {d / 1000:.3f}")
        lines.append(lines[-2])  # concat demuxer needs the last file repeated
        (tmp / "list.txt").write_text("\n".join(lines))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(tmp / "list.txt"), "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", "20",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        str(OUT / name)], check=True)
    print(name, "size KB:", (OUT / name).stat().st_size // 1024)


# ---------------------------------------------------------------- item drawing
def draw_items(ax, items, step):
    for it in items:
        if it["lv"] > step:
            continue
        new = it["lv"] == step
        t = it["t"]
        if t == "node":
            fc, ec = KIND[it["kind"]]
            x, y, w, h = it["x"], it["y"], it.get("w", 2.0), it.get("h", 0.8)
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=fc, ec=ORANGE if new else ec, lw=3 if new else 1.4,
                                        zorder=3))
            if "title" in it:
                ax.text(x, y + h * 0.26, it["title"], ha="center", va="center",
                        fontsize=it.get("fs", 10.5), fontweight="bold", color=INK, zorder=4)
                ax.text(x, y - h * 0.12, it["body"], ha="center", va="center",
                        fontsize=it.get("fs2", 9), color=INK, zorder=4, linespacing=1.4)
            else:
                ax.text(x, y, it["text"], ha="center", va="center", fontsize=it.get("fs", 9),
                        color=INK, zorder=4, linespacing=1.3)
        elif t == "link":
            ax.add_patch(FancyArrowPatch(it["p"], it["q"], arrowstyle="-|>", mutation_scale=13,
                                         color=ORANGE if new else "#9ca3af",
                                         lw=2.2 if new else 1.3,
                                         ls="--" if it.get("dash") else "-", zorder=2))
        elif t == "text":
            ax.text(it["x"], it["y"], it["s"], fontsize=it.get("fs", 10),
                    color=it.get("color", INK), fontweight="bold" if it.get("bold") else "normal",
                    ha=it.get("ha", "left"), va="center", zorder=4)


def node(lv, x, y, kind, text=None, title=None, body=None, **kw):
    d = dict(t="node", lv=lv, x=x, y=y, kind=kind, **kw)
    if title is not None:
        d.update(title=title, body=body)
    else:
        d["text"] = text
    return d


def link(lv, p, q, dash=False):
    return dict(t="link", lv=lv, p=p, q=q, dash=dash)


def text(lv, x, y, s, **kw):
    return dict(t="text", lv=lv, x=x, y=y, s=s, **kw)


def render_scene(title, items, caps, size, dpi, xlim, ylim, name):
    n = len(caps)
    imgs = []
    for i in range(n):
        fig, ax = base(title, i, n, caps[i], size, dpi, xlim, ylim)
        draw_items(ax, items, i)
        imgs.append(to_image(fig))
    save_gif(imgs, name)


# ================================================================== GIF 1
FAULT = 116.3
SECS = list(range(108, 121))
TW = [
    dict(cap="One box is one second of the flight.\nRow t covers the time from t to t + 1 seconds."),
    dict(cap="The fault starts at 116.3 s.\nThat moment is inside second 116 (from 116 to 117).",
         fault=True, hi=116),
    dict(cap="The label rule: label = 1 when  t + 1 > fault_sec.\nIn words: the second ends after the fault started.",
         fault=True, rule=True),
    dict(cap="t = 115:  116 > 116.3 is false, so the label is 0.",
         fault=True, rule=True, labels=range(108, 116), hi=115),
    dict(cap="t = 116:  117 > 116.3 is true, so the label is 1.\nThis is the first 1, and it is the second that contains the fault start.",
         fault=True, rule=True, labels=range(108, 117), hi=116),
    dict(cap="Every later second is also 1.", fault=True, rule=True, labels=SECS),
    dict(cap="Features look backward.\nmean5 for row 116 is the average of seconds 112 to 116.",
         fault=True, rule=True, labels=SECS, row=116),
    dict(cap="Row 117 gets its own window: seconds 113 to 117.\nThe window slides one second at a time.",
         fault=True, rule=True, labels=SECS, row=117),
    dict(cap="Back to row 116. Seconds after t are never used,\nso a row cannot see the future.",
         fault=True, rule=True, labels=SECS, row=116, future=True),
    dict(cap="Remember: the first label-1 second contains the fault start.\nIt is not \"one second ahead\". Features use only seconds up to t.",
         fault=True, rule=True, labels=SECS, row=116, future=True),
]


def draw_tw(i, win_x=None, row_hi=None):
    st = dict(TW[i])
    if row_hi is not None:
        st["row"] = row_hi
    fig, ax = base("Time window and label", i, len(TW), st["cap"], (11, 6.2), 100,
                   (106.2, 121.5), (-1.3, 4.2))
    labels, hi, row = set(st.get("labels", [])), st.get("hi"), st.get("row")
    for t in SECS:
        lab = 1 if t + 1 > FAULT else 0
        if t in labels:
            fc, ec = ("#fecaca", "#dc2626") if lab else ("#bbf7d0", "#16a34a")
        else:
            fc, ec = "#e0f2fe", "#0284c7"
        lw = 1.5
        if t == hi:
            ec, lw = ORANGE, 3.5
        if row is not None and t == row:
            ec, lw = "#7c3aed", 3.5
        ax.add_patch(Rectangle((t, 0), 1, 1, fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(t + 0.5, 0.5, f"t={t}", ha="center", va="center", fontsize=9, zorder=3)
        if t in labels:
            ax.text(t + 0.5, -0.5, str(lab), ha="center", va="center", fontsize=14,
                    fontweight="bold", color="#dc2626" if lab else "#16a34a")
    ax.text(106.9, 0.5, "row", ha="right", va="center", fontsize=10, color=GREY)
    if labels:
        ax.text(106.9, -0.5, "label", ha="right", va="center", fontsize=10, color=GREY)
    if st.get("fault"):
        ax.plot([FAULT, FAULT], [-0.95, 3.2], color="#dc2626", lw=2.2, ls="--", zorder=4)
        ax.text(FAULT + 0.1, 3.3, "fault starts at 116.3 s", color="#dc2626", fontsize=11,
                fontweight="bold")
    if st.get("rule"):
        ax.text(107, 3.5, "label = 1 when  t + 1 > fault_sec", fontsize=11, color="#7c3aed",
                fontweight="bold", va="center")
    if row is not None:
        x0 = row - 4 if win_x is None else win_x
        ax.add_patch(Rectangle((x0, 1.2), 5, 0.6, fc="#ddd6fe", ec="#7c3aed", lw=1.8, zorder=2))
        ax.text(x0 + 2.5, 1.5, f"mean5 window: seconds {round(x0)} to {round(x0) + 4}",
                ha="center", va="center", fontsize=9.5, color="#5b21b6", zorder=3)
    if st.get("future") and row is not None:
        ax.add_patch(Rectangle((row + 1, 0), 121 - (row + 1), 1, fc="#d1d5db", ec="none",
                               alpha=0.8, hatch="///", zorder=5))
        ax.text((row + 1 + 121) / 2, 1.5, "never used by row 116", ha="center", fontsize=10.5,
                color=GREY, fontweight="bold")
    return to_image(fig)


def gif1():
    steps = [draw_tw(i) for i in range(len(TW))]
    slide = [draw_tw(7, win_x=112 + k / 9, row_hi=116 if k / 9 < 0.5 else 117)
             for k in range(1, 9)]
    save_gif(steps, "gif1_time_window.gif", extra={6: slide})


# ================================================================== GIF 2
PIPE_ITEMS = [
    node(0, 1.3, 5.3, "plain", title="Raw flights", body="ALFA flight files\ncarbonZ_*\n(CMU dataset)",
         w=2.2, h=1.8),
    node(1, 4.0, 5.3, "build", title="Per-second table", body="01_build_table.py\ntable.csv\n(mean, label)",
         w=2.2, h=1.8),
    link(1, (2.4, 5.3), (2.9, 5.3)),
    node(2, 6.7, 5.3, "hub", title="Features", body="mean5, std5, delta10\n21 used by models\n(30 built)",
         w=2.2, h=1.8),
    link(2, (5.1, 5.3), (5.6, 5.3)),
    node(3, 9.4, 5.3, "dev", title="Split", body="02_make_split.py\nsplit.csv\ndev 37, test 9",
         w=2.2, h=1.8),
    link(3, (7.8, 5.3), (8.3, 5.3)),
    node(4, 12.1, 5.3, "model", title="Detectors", body="A rule, F forest,\nD2 command rule\n(scripts 04 to 09)",
         w=2.2, h=1.8),
    link(4, (10.5, 5.3), (11.0, 5.3)),
    text(5, 0.2, 3.5, "Figures and the script that draws each", fs=12, bold=True),
    node(5, 1.3, 2.4, "out", title="fig1, fig2", body="05_figures.py", w=2.2, h=1.1, fs2=8.5),
    node(5, 3.9, 2.4, "out", title="fig3, fig4, fig5", body="07_health_figures.py", w=2.4, h=1.1, fs2=8.5),
    node(5, 6.7, 2.4, "out", title="fig6, fig7", body="08_correlation.py", w=2.4, h=1.1, fs2=8.5),
    node(5, 9.4, 2.4, "out", title="fig8", body="10_comparison_figure.py", w=2.4, h=1.1, fs2=8.5),
    node(5, 12.1, 2.4, "out", title="fig9", body="15_calibration.py", w=2.2, h=1.1, fs2=8.5),
    text(6, 0.2, 0.7, "Every feature is built only from seconds up to the current one. "
         "The test flights are scored once, at the end.", fs=11, color="#7c3aed", bold=True),
]
PIPE_CAPS = [
    "Everything starts from the raw ALFA flight files (carbonZ_*), one folder per flight.",
    "01_build_table.py turns each flight into one row per second and adds the label.\nOutput: table.csv.",
    "From the table we build rolling features: mean5, std5 and delta10 for 10 signals.\n30 are built, 21 are used by the models.",
    "02_make_split.py gives whole sessions either to the dev part or to the locked test part.\nDev has 37 flights, test has 9.",
    "Three detectors read the features: A (a simple rule), F (a random forest)\nand D2 (a rule on the frozen command channels).",
    "Each figure is drawn by one script.\nfig9, the calibration plot from 15_calibration.py, is not on GitHub yet.",
    "The rule of the whole pipeline:\nfeatures look backward only, and the test flights are scored once.",
]


def gif2():
    render_scene("From raw flight files to the figures", PIPE_ITEMS, PIPE_CAPS, (13, 7.2), 100,
                 (0, 13.4), (0, 6.6), "gif2_pipeline.gif")


# ================================================================== GIF 3
COUNTS = [("cv0", 3), ("cv1", 6), ("cv2", 5), ("cv3", 6), ("cv4", 4), ("test", 6)]
SPLIT_CAPS = [
    "The data has 30 sessions of flights.\nSessions are never cut: all flights of a session stay together.",
    "02_make_split.py hands the sessions to six parts:\nfive dev folds (cv0 to cv4) and one test part.",
    "The test part (6 sessions, 9 flights) is locked.\nWe do not tune anything on it.",
    "Round 1: train on cv1 to cv4, validate on cv0.",
    "Round 2: train on cv0, cv2, cv3 and cv4, validate on cv1.",
    "Round 3: validate on cv2, train on the other dev folds.",
    "Round 4: validate on cv3.",
    "Round 5: validate on cv4. Now every dev fold has been validated once.",
    "Why by session: no session is ever in both training and validation.\nThe test part is scored once, at the very end.",
]


def draw_split(i, step=None, n=None, cap=None):
    fig, ax = base("How the data is split", i if step is None else step, n or len(SPLIT_CAPS),
                   cap or SPLIT_CAPS[i], (12, 6.4), 100, (0, 13), (-0.2, 6))
    scale = 11.8 / 30
    pos, x = {}, 0.1
    for p, c in COUNTS:
        pos[p] = (x, c * scale)
        x += c * scale
    ax.text(0.1, 5.5, "30 sessions", fontsize=12, fontweight="bold", color=INK)
    for p, c in COUNTS:
        x0, w = pos[p]
        if i == 0:
            kind, label = "plain", f"{c}"
        else:
            kind = "test" if p == "test" else "dev"
            label = f"{p}\n{c}"
        new = (i == 1 and p != "test") or (i == 2 and p == "test")
        fc, ec = KIND[kind]
        ax.add_patch(FancyBboxPatch((x0, 4.3), w - 0.06, 0.8,
                                    boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc,
                                    ec=ORANGE if new else ec, lw=3 if new else 1.4))
        ax.text(x0 + w / 2 - 0.03, 4.7, label, ha="center", va="center", fontsize=9)
    if i >= 3:
        ax.text(0.1, 3.7, "Cross-validation on the dev part", fontsize=11, fontweight="bold",
                color=INK)
    for r in range(5):
        if i < 3 + r:
            continue
        y = 3.0 - r * 0.62
        ax.text(0.1, y + 0.22, f"round {r + 1}", fontsize=9.5, va="center", color=GREY)
        for p, c in COUNTS:
            x0, w = pos[p]
            if p == "test":
                kind, txt = "plain", "locked"
            elif p == f"cv{r}":
                kind, txt = "val", "validate"
            else:
                kind, txt = "dev", "train"
            fc, ec = KIND[kind]
            newrow = i == 3 + r
            ax.add_patch(Rectangle((x0 + 1.0, y), w - 0.06, 0.45, fc=fc,
                                   ec=ORANGE if newrow and kind == "val" else ec,
                                   lw=2.6 if newrow and kind == "val" else 0.9))
            ax.text(x0 + 1.0 + w / 2 - 0.03, y + 0.22, txt, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 14)
    return to_image(fig)


def gif3():
    save_gif([draw_split(i) for i in range(len(SPLIT_CAPS))], "gif3_split.gif")


# ================================================================== GIF 4
XS = [1.2, 3.5, 5.8, 8.1, 10.4, 12.7, 15.0, 17.3, 19.6]
MN = ["04_final_test.py", "05_figures.py\n(also raw rc-out)", "06_oof_predictions.py",
      "08_correlation.py", "09_model_comparison.py", "16_permutation_\nimportance.py",
      "17_learning_curve.py", "18_train_final.py", "19_sample_request.py"]
MF = ["test scores\n(printed once)", "fig1, fig2", "results/\noof_predictions.csv", "fig6, fig7",
      "results/oof_models.csv\nmodel_comparison.csv", "permutation_\nimportance.csv",
      "learning_curve.csv", "models/\nforest.joblib", "sample_request.json"]
REP = [(5.0, "07_health_\nfigures.py", 2, "fig3, fig4, fig5", "out"),
       (6.95, "12_conformal_\nthreshold.py", 2, "conformal_\nthreshold.csv", "out"),
       (8.9, "10_comparison_\nfigure.py", 4, "fig8", "out"),
       (10.7, "13_conformal_\nmodels.py", 4, "conformal_\nmodels.csv", "out"),
       (12.5, "14_dummy_\nbaseline.py", 4, "printed\ncomparison", "out"),
       (14.3, "15_calibration.py", 4, "fig9 (not on\nGitHub yet)", "out"),
       (17.3, "api.py", 7, "predictions\nover HTTP", "out")]

SM = [
    node(0, 9.0, 11.3, "data", text="raw flight files (carbonZ_*)", w=3.2, h=0.7),
    node(0, 9.0, 10.4, "build", text="01_build_table.py"),
    node(0, 9.0, 9.5, "data", text="table.csv"),
    link(0, (9.0, 11.0), (9.0, 10.75)), link(0, (9.0, 10.05), (9.0, 9.85)),
    node(1, 4.5, 8.3, "build", text="02_make_split.py"),
    node(1, 4.5, 7.4, "data", text="split.csv"),
    link(1, (8.4, 9.2), (5.1, 8.65)), link(1, (4.5, 7.95), (4.5, 7.75)),
    node(2, 13.8, 8.3, "build", text="03_command_features.py\n(also reads raw rc-out)", w=2.8, h=0.8),
    node(2, 13.8, 7.4, "data", text="table2.csv"),
    link(2, (9.6, 9.2), (13.2, 8.65)), link(2, (5.5, 7.5), (12.4, 8.2)),
    link(2, (13.8, 7.9), (13.8, 7.75)),
    node(3, 10.75, 6.3, "hub", w=20.3, h=0.75, fs=10,
         text="evalkit.py: load() joins table.csv and split.csv, plus the shared rule, persist and "
              "evaluate code (config.py gives every script the data paths)"),
    link(3, (9.0, 9.15), (9.0, 6.7)), link(3, (4.5, 7.05), (4.5, 6.7)),
    link(3, (13.8, 7.05), (13.8, 6.7), dash=True),
    text(3, 15.4, 7.4, "04 and 05 switch to table2.csv", fs=9, color=GREY),
]
for x, n, f in zip(XS, MN, MF):
    SM.append(node(4, x, 5.2, "model", text=n, w=2.15, h=0.8))
    SM.append(link(4, (x, 5.95), (x, 5.6)))
    SM.append(node(5, x, 3.85, "data", text=f, w=2.15, h=0.8, fs=8.5))
    SM.append(link(5, (x, 4.8), (x, 4.25)))
for x, n, src, o, k in REP:
    SM.append(node(6, x, 2.3, "report", text=n, w=1.7, h=0.8))
    SM.append(link(6, (XS[src], 3.45), (x, 2.7)))
    SM.append(node(7, x, 0.9, k, text=o, w=1.7, h=0.8, fs=8))
    SM.append(link(7, (x, 1.9), (x, 1.3)))
SM.append(text(8, 0.3, 0.1, "tests/: test_evalkit, test_evaluate and test_data check evalkit.py and the data; "
               "test_api checks api.py.", fs=10, color="#7c3aed", bold=True))
SM_CAPS = [
    "01_build_table.py reads the raw flight files and writes table.csv, one row per second.",
    "02_make_split.py reads table.csv and writes split.csv (which flight goes to which part).",
    "03_command_features.py adds the command-channel features and writes table2.csv.",
    "evalkit.py is the shared module. Its load() joins table.csv and split.csv.\nconfig.py gives every script the data paths.",
    "These scripts all load their data through evalkit.py.\n04 and 05 switch it to table2.csv.",
    "Each model script writes its own result file, or its figures.\n06 writes the out-of-fold predictions and 09 the model results.",
    "The report scripts read those result files:\n07 and 12 read 06's file, 10, 13, 14 and 15 read 09's files, api.py loads forest.joblib.",
    "They write the remaining figures, tables and the API output.\nfig9 from 15_calibration.py is not on GitHub yet.",
    "Run order is the number in the file name.\nScripts 07 to 15 need the result files of 06 and 09 first.",
]


def gif4():
    render_scene("How the Python files connect", SM, SM_CAPS, (18, 10.6), 80, (0, 21.5),
                 (-0.3, 12.0), "gif4_script_map.gif")


if __name__ == "__main__":
    gif1()
    gif2()
    gif3()
    gif4()
