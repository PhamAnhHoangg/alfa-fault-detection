"""Draws the explanation diagrams (no data needed, facts come from the scripts).

Run:  python src/make_diagrams.py
Writes figures/diagram1_time_window.png ... diagram5_detectors.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

INK, GREY, LIGHT = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, RED, GREEN, AMBER = "#2563eb", "#dc2626", "#16a34a", "#d97706"
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fs=9, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color)


def arrow(ax, p, q, color=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3))


# ---------------------------------------------------------------- picture 1
def time_window():
    fault_sec, t_row = 116.166, 116
    secs = list(range(108, 121))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(107.5, 121.5)
    ax.set_ylim(-3.2, 4.2)
    ax.axis("off")

    # one box per second (second t covers t to t+1)
    for t in secs:
        label = 1 if t + 1 > fault_sec else 0
        fc = "#fee2e2" if label else "#f3f4f6"
        ax.add_patch(Rectangle((t, 0), 1, 1, fc=fc, ec=INK, lw=1))
        ax.text(t + 0.5, 0.5, f"t={t}", ha="center", va="center", fontsize=9)
        ax.text(t + 0.5, -0.45, str(label), ha="center", va="center", fontsize=12,
                fontweight="bold", color=RED if label else GREY)
    ax.text(107.4, -0.45, "label", ha="right", va="center", fontsize=10, color=INK)
    ax.text(107.4, 0.5, "row", ha="right", va="center", fontsize=10, color=INK)

    # 5-second window for the row t=116
    ax.add_patch(Rectangle((t_row - 4, 1.25), 5, 0.55, fc="#dbeafe", ec=BLUE, lw=1.5))
    ax.text(t_row - 1.5, 1.52, "mean5 window: seconds 112 to 116", ha="center",
            va="center", fontsize=9.5, color=BLUE)
    ax.add_patch(Rectangle((t_row, 0), 1, 1, fc="none", ec=BLUE, lw=3))
    ax.annotate("this row", xy=(t_row + 0.5, 1.0), xytext=(t_row + 2.6, 2.6), fontsize=10,
                color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))

    # fault start
    ax.plot([fault_sec, fault_sec], [-0.9, 3.6], color=RED, lw=2, ls="--")
    ax.text(fault_sec + 0.1, 3.75, "fault starts at 116.17 s", color=RED, fontsize=10.5,
            fontweight="bold", ha="left")

    # rule box
    ax.text(108, 3.55, "label = 1 when  t + 1 > fault_sec", fontsize=10.5, color=INK,
            fontweight="bold")
    ax.text(108, 3.1, "t=115: 116 > 116.17 is false, so 0\nt=116: 117 > 116.17 is true, so 1",
            fontsize=9.5, color=INK, va="top")

    # looks backward only
    ax.text(t_row + 2.9, -1.2, "later seconds are never used by the row at t=116",
            fontsize=9.5, color=GREY, ha="center")
    ax.add_patch(Rectangle((t_row + 1, -1.05), 4.5, 0.0001, fc="none", ec="none"))

    ax.text(114, -2.15,
            "The first label-1 second (t=116) contains the fault start. It is not\n"
            "\"one second ahead\". Features look backward only: a row sees seconds up to and\n"
            "including its own, never later ones.",
            fontsize=9.5, color=INK, ha="center", va="top")
    ax.set_title("One second, its 5-second window, and the label flip", fontsize=13,
                 fontweight="bold", color=INK, loc="left")
    fig.savefig(OUT / "diagram1_time_window.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- picture 2
def pipeline():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    cols = [
        (0.2, "Raw flights", "ALFA flight files\ncarbonZ_*\n(CMU dataset)", "#f3f4f6"),
        (2.9, "Per-second table", "01_build_table.py\ntable.csv\n(mean, label)", "#dbeafe"),
        (5.6, "Features", "mean5, std5, delta10\n21 used by models\n(30 built)", "#dbeafe"),
        (8.3, "Split", "02_make_split.py\nsplit.csv\ndev 37, test 9", "#fef3c7"),
        (11.0, "Detectors", "A rule, F forest,\nD2 command rule\n(scripts 04 to 09)", "#dcfce7"),
    ]
    for x, title, body, fc in cols:
        box(ax, x, 3.6, 2.0, 2.0, "", fc=fc)
        ax.text(x + 1.0, 5.2, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(x + 1.0, 4.3, body, ha="center", va="center", fontsize=9)
    for i in range(len(cols) - 1):
        arrow(ax, (cols[i][0] + 2.0, 4.6), (cols[i + 1][0], 4.6))

    ax.text(0.2, 2.9, "Figures and the script that draws each", fontsize=11.5, fontweight="bold")
    outs = [
        (0.2, "fig1, fig2", "05_figures.py"),
        (2.2, "fig3, fig4, fig5", "07_health_figures.py"),
        (4.9, "fig6, fig7", "08_correlation.py"),
        (7.6, "fig8", "10_comparison_figure.py"),
        (10.2, "fig9", "15_calibration.py"),
    ]
    widths = [1.8, 2.5, 2.5, 2.4, 2.4]
    for (x, name, src), w in zip(outs, widths):
        box(ax, x, 1.5, w, 1.0, "", fc="#ffffff")
        ax.text(x + w / 2, 2.15, name, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, 1.78, src, ha="center", va="center", fontsize=8.5, color=GREY)

    ax.text(0.2, 0.9, "Rule of the whole pipeline: every feature is built only from seconds up to "
            "the current one,\nand the test flights are scored once at the end.",
            fontsize=10, color=INK, va="top")
    ax.set_title("From raw flight files to the figures",
                 fontsize=13, fontweight="bold", loc="left")
    fig.savefig(OUT / "diagram2_pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- picture 3
def split():
    # taken from ASSIGN in 02_make_split.py: sessions per part
    counts = {"cv0": 3, "cv1": 6, "cv2": 5, "cv3": 6, "cv4": 4, "test": 6}
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    ax.text(0.1, 5.2, "30 sessions, split by session (never inside a session)",
            fontsize=12, fontweight="bold")
    parts = list(counts)
    x = 0.1
    scale = 11.8 / 30
    pos = {}
    for p in parts:
        w = counts[p] * scale
        fc = "#fecaca" if p == "test" else "#dbeafe"
        ax.add_patch(Rectangle((x, 4.1), w - 0.04, 0.7, fc=fc, ec=INK, lw=1))
        ax.text(x + w / 2, 4.45, f"{p}\n{counts[p]}", ha="center", va="center", fontsize=9)
        pos[p] = (x, w)
        x += w

    ax.text(0.1, 3.6, "Cross-validation on the dev part: one fold held out at a time",
            fontsize=11, fontweight="bold")
    for i, held in enumerate(["cv0", "cv1", "cv2", "cv3", "cv4"]):
        y = 2.9 - i * 0.55
        ax.text(0.1, y + 0.2, f"round {i + 1}", fontsize=9, va="center")
        for p in parts:
            x0, w = pos[p]
            if p == "test":
                fc, txt = "#f3f4f6", "locked"
            elif p == held:
                fc, txt = "#fde68a", "validate"
            else:
                fc, txt = "#bfdbfe", "train"
            ax.add_patch(Rectangle((x0 + 0.9, y), w - 0.04, 0.4, fc=fc, ec=INK, lw=0.8))
            ax.text(x0 + 0.9 + w / 2, y + 0.2, txt, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 13)
    ax.text(0.1, -0.25,
            "Whole sessions stay in one part, so no session is in both training and validation.\nThe test flights (9 flights) are scored once, at the very end.",
            fontsize=9.5, color=INK, va="top")
    ax.set_ylim(-0.9, 5.6)
    ax.set_title("How the data is split", fontsize=13, fontweight="bold", loc="left", y=1.02)
    fig.savefig(OUT / "diagram3_split.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- picture 4
def script_map():
    fig, ax = plt.subplots(figsize=(21.5, 13))
    ax.set_xlim(0, 21.5)
    ax.set_ylim(-0.4, 13.2)
    ax.axis("off")
    C = {"data": "#f3f4f6", "build": "#dbeafe", "model": "#dcfce7", "report": "#fef3c7",
         "hub": "#ede9fe", "out": "#ffffff"}

    def node(x, y, text, kind, w=2.0, h=0.7, fs=8):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=C[kind], ec=INK, lw=1.1))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def link(p, q, color=GREY, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                     color=color, lw=1.1, ls=ls))

    ax.text(0.2, 12.9, "How the Python files connect: files in grey, scripts in colour",
            fontsize=14, fontweight="bold")
    # legend
    for i, (k, lab) in enumerate([("data", "data or result file"), ("build", "builds the data"),
                                  ("hub", "shared module"), ("model", "reads the data"),
                                  ("report", "reads a result file")]):
        node(1.6 + i * 2.7, 12.3, lab, k, w=2.4, h=0.5, fs=8)

    node(9.0, 11.3, "raw flight files (carbonZ_*)", "data", w=3.2)
    node(9.0, 10.4, "01_build_table.py", "build")
    node(9.0, 9.5, "table.csv", "data")
    link((9.0, 11.0), (9.0, 10.75)); link((9.0, 10.05), (9.0, 9.85))

    node(4.5, 8.3, "02_make_split.py", "build")
    node(4.5, 7.4, "split.csv", "data")
    link((8.4, 9.2), (5.1, 8.65)); link((4.5, 7.95), (4.5, 7.75))

    node(13.8, 8.3, "03_command_features.py\n(also reads raw rc-out)", "build", w=2.8, h=0.8)
    node(13.8, 7.4, "table2.csv", "data")
    link((9.6, 9.2), (13.2, 8.65)); link((5.5, 7.5), (12.4, 8.2)); link((13.8, 7.9), (13.8, 7.75))

    # evalkit hub
    ax.add_patch(FancyBboxPatch((0.6, 5.95), 20.3, 0.75,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=C["hub"], ec=INK, lw=1.2))
    ax.text(10.75, 6.32, "evalkit.py: load() joins table.csv and split.csv, plus the shared rule, "
            "persist and evaluate code (config.py gives every script the data paths)",
            ha="center", va="center", fontsize=9.5)
    link((9.0, 9.15), (9.0, 6.72)); link((4.5, 7.05), (4.5, 6.72))
    link((13.8, 7.05), (13.8, 6.72), ls="--")
    ax.text(14.0, 6.9, "04 and 05 switch to table2.csv", fontsize=8, color=GREY)

    xs = [1.2, 3.5, 5.8, 8.1, 10.4, 12.7, 15.0, 17.3, 19.6]
    names = ["04_final_test.py", "05_figures.py\n(also raw rc-out)", "06_oof_predictions.py",
             "08_correlation.py", "09_model_comparison.py", "16_permutation_\nimportance.py",
             "17_learning_curve.py", "18_train_final.py", "19_sample_request.py"]
    for x, n in zip(xs, names):
        node(x, 5.2, n, "model", w=2.15, h=0.8)
        link((x, 5.95), (x, 5.6))

    files = {0: "test scores\n(printed once)", 1: "fig1, fig2", 2: "results/\noof_predictions.csv",
             3: "fig6, fig7", 4: "results/oof_models.csv\nmodel_comparison.csv",
             5: "permutation_\nimportance.csv", 6: "learning_curve.csv",
             7: "models/\nforest.joblib", 8: "sample_request.json"}
    for i, t in files.items():
        node(xs[i], 3.85, t, "data", w=2.15, h=0.8, fs=7.5)
        link((xs[i], 4.8), (xs[i], 4.25))

    rep = [(5.0, "07_health_\nfigures.py", 2), (6.95, "12_conformal_\nthreshold.py", 2),
           (8.9, "10_comparison_\nfigure.py", 4), (10.7, "13_conformal_\nmodels.py", 4),
           (12.5, "14_dummy_\nbaseline.py", 4), (14.3, "15_calibration.py", 4),
           (17.3, "api.py", 7)]
    for x, n, src in rep:
        node(x, 2.3, n, "report", w=1.7, h=0.8)
        link((xs[src], 3.45), (x, 2.7))

    outs = [(5.0, "fig3, fig4, fig5"), (6.95, "conformal_\nthreshold.csv"), (8.9, "fig8"),
            (10.7, "conformal_\nmodels.csv"), (12.5, "printed\ncomparison"),
            (14.3, "fig9"), (17.3, "predictions over\nHTTP")]
    for x, t in outs:
        node(x, 0.9, t, "out", w=1.7, h=0.8, fs=7.5)
        link((x, 1.9), (x, 1.3))

    ax.text(0.3, 0.2, "tests/: test_evalkit, test_evaluate and test_data check evalkit.py and the "
            "data, test_api checks api.py.", fontsize=9, color=INK)
    ax.text(0.3, -0.2, "Run order is the number in the file name. Scripts 07 to 15 read result "
            "files that 06 and 09 must have written first.", fontsize=9, color=INK)
    fig.savefig(OUT / "diagram4_script_map.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- picture 5
def detectors():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    cols = [
        (0.3, "A  sensor rule", "#dbeafe", "21 sensor features",
         "one feature, one threshold\nfitted on dev rows:\nvel_lin_z__mean5 <= -0.833", "3 seconds in a row", "test F1 0.46"),
        (4.75, "F  sensor forest", "#dbeafe", "21 sensor features",
         "random forest, 300 trees,\nclass-balanced\nfault if p >= 0.5", "3 seconds in a row", "test F1 0.50"),
        (9.2, "D2  command side", "#fef3c7", "rc-out channels 1 to 5",
         "channel 1, 3, 4 or 5 frozen 3 s\nOR channel 2 (throttle)\nbelow a level fitted on dev", "throttle part: 3 in a row", "test F1 0.87, a ceiling"),
    ]
    for x, title, fc, reads, rule, persist, score in cols:
        ax.text(x + 1.95, 6.05, title, ha="center", va="center", fontsize=12, fontweight="bold")
        box(ax, x, 4.75, 3.9, 0.8, "reads: " + reads, fc=fc, fs=9.5)
        box(ax, x, 2.95, 3.9, 1.3, rule, fc="#ffffff", fs=9.5)
        box(ax, x, 1.75, 3.9, 0.7, persist, fc="#ffffff", fs=9.5)
        box(ax, x, 0.75, 3.9, 0.6, score, fc="#f3f4f6", fs=9.5, bold=True)
        for y0, y1 in ((4.75, 4.25), (2.95, 2.45), (1.75, 1.35)):
            arrow(ax, (x + 1.95, y0), (x + 1.95, y1))
    ax.text(0.3, 0.3, "D2 reads the recorded servo commands, which already contain the injected fault, "
            "so its score shows how easy the task gets, not a detector for a real aircraft.",
            fontsize=9.5, color=INK)
    ax.set_title("The three detectors and what each one reads", fontsize=13, fontweight="bold", loc="left")
    fig.savefig(OUT / "diagram5_detectors.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    time_window()
    pipeline()
    split()
    script_map()
    detectors()
    print("saved 5 diagrams in", OUT)
