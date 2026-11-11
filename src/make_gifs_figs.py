"""Step-by-step GIFs for the result figures (fig1 to fig6 and fig8).

Run:  python src/make_gifs_figs.py
Reads  figures/fig*.png  (already committed)
Writes figures/gif_fig1.gif ... gif_fig6.gif, gif_fig8.gif

Each GIF shows one finished figure. Every step dims everything except one
region (orange box) and explains it in a caption. 3 seconds per step, with a
short cross-fade between steps. No data is read, only the PNG files.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIG = Path(__file__).resolve().parent.parent / "figures"
W, H = 1280, 620
HEADER_H, BAR_H = 56, 10
AREA = (20, HEADER_H + 16, 800, H - BAR_H - 16)       # where the figure goes
PANEL_X = 826                                          # right text panel
STEP_MS, LAST_MS, FADE_N, FADE_MS = 3000, 5000, 6, 50
ORANGE = "#f97316"
COLORS = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#db2777", "#0284c7", "#dc2626"]
FD = "/usr/share/fonts/truetype/dejavu/"


def font(size, bold=False):
    return ImageFont.truetype(FD + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


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


def frame(png, title, steps, i):
    """One frame: figure with the box of step i, header, caption, progress bar."""
    n = len(steps)
    box, head, text = steps[i]
    color = COLORS[i % len(COLORS)]
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # header band
    d.rectangle((0, 0, W, HEADER_H), fill="#1e3a8a")
    d.text((20, 14), title, font=font(24, True), fill="white")
    tag = f"Step {i + 1} of {n}"
    d.text((W - 20 - d.textlength(tag, font=font(22, True)), 16), tag, font=font(22, True), fill="#fde047")

    # figure, scaled to fit the area
    ax0, ay0, ax1, ay1 = AREA
    s = min((ax1 - ax0) / png.width, (ay1 - ay0) / png.height)
    pw, ph = int(png.width * s), int(png.height * s)
    px, py = ax0 + (ax1 - ax0 - pw) // 2, ay0 + (ay1 - ay0 - ph) // 2
    fig = png.resize((pw, ph), Image.LANCZOS).convert("RGB")
    if box is not None:                                # dim everything outside the box
        dim = Image.blend(fig, Image.new("RGB", fig.size, "white"), 0.72)
        bx = [int(v * s) for v in box]
        out = dim.copy()
        out.paste(fig.crop(bx), (bx[0], bx[1]))
        fig = out
    img.paste(fig, (px, py))
    d.rectangle((px - 1, py - 1, px + pw, py + ph), outline="#cbd5e1", width=1)
    if box is not None:
        bx = [int(v * s) for v in box]
        d.rectangle((px + bx[0], py + bx[1], px + bx[2], py + bx[3]), outline=ORANGE, width=4)

    # caption panel
    pw2 = W - PANEL_X - 20
    d.rounded_rectangle((PANEL_X, HEADER_H + 16, PANEL_X + pw2, H - BAR_H - 16), radius=14,
                        fill="#f8fafc", outline=color, width=4)
    d.rounded_rectangle((PANEL_X, HEADER_H + 16, PANEL_X + pw2, HEADER_H + 62), radius=14, fill=color)
    d.text((PANEL_X + 16, HEADER_H + 26), head, font=font(21, True), fill="white")
    y = HEADER_H + 80
    fnt = font(19)
    for line in wrap(d, text, fnt, pw2 - 32):
        d.text((PANEL_X + 16, y), line, font=fnt, fill="#1f2937")
        y += 28

    # progress bar
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


# --------------------------------------------------------------------------- GIFs
FIG1 = [
    (None, "The whole figure",
     "One test engine flight (2018-09-11-11-56-30). The x axis is seconds relative to the fault start, "
     "so 0 is the moment the fault begins. Three panels share it."),
    ((232, 52, 1328, 375), "Panel 1: vertical velocity",
     "The 5 second mean of vertical velocity (vel_lin_z__mean5). Before the fault it wanders around zero."),
    ((232, 222, 1328, 240), "The dashed threshold",
     "Detector A fires when this signal is at or below the dashed line (about -0.833, fitted on development data) "
     "for 3 seconds in a row."),
    ((1085, 40, 1115, 960), "The red line",
     "The fault starts here. Everything to the right of it is a fault second."),
    ((232, 403, 1328, 723), "Panel 2: throttle output",
     "The throttle command in microseconds. After the fault it drops to about 1100 and stays flat. "
     "This is what the command side sees."),
    ((232, 753, 1328, 962), "Panel 3: alarms",
     "One row per detector. A tick is a second where that detector is in alarm."),
    ((1130, 780, 1285, 815), "D2 fires first",
     "D2 (command side) starts a few seconds after the fault, soon after the throttle falls."),
    ((1190, 840, 1285, 875), "F comes later",
     "F, the forest on sensor features, starts later than D2."),
    ((1150, 900, 1285, 935), "A is the slowest",
     "A, the sensor rule, has one early tick, then stays silent until the vertical velocity has fallen far enough."),
]

FIG2 = [
    (None, "The whole figure",
     "Test flight 2018-09-11-17-27-13 with two faults: rudder_zero and a left aileron failure. "
     "Raw rc-out channels 3, 4 and 5 from 20 s before to 30 s after the fault start."),
    ((140, 50, 1328, 282), "Channel 3: rudder",
     "The inferred rudder command (PWM in microseconds) moves freely before the fault."),
    ((651, 130, 1280, 170), "Rudder frozen",
     "After the red line the rudder command is a flat line at 1500. A frozen command is the signal D2 looks for."),
    ((140, 316, 1328, 548), "Channel 4: left aileron",
     "The inferred left aileron command. Same picture: free before the fault."),
    ((651, 400, 1280, 430), "Left aileron frozen",
     "Also flat at 1500 after the fault starts."),
    ((140, 581, 1328, 814), "Channel 5: right aileron",
     "The right aileron was not failed, so it keeps moving after the fault. "
     "The contrast with channels 3 and 4 is the point of the figure."),
]

FIG3 = [
    (None, "The whole figure",
     "Random forest, development folds: recall by fault type. Recall is the share of fault seconds that were detected."),
    ((665, 65, 1015, 130), "Two bars per type",
     "Light bar: the raw per-second prediction. Dark bar: the alarm, which needs 3 seconds in a row."),
    ((140, 150, 315, 545), "Engine",
     "301 fault seconds. Raw recall 0.75, alarm recall 0.61. The best detected type by far."),
    ((365, 370, 540, 545), "Elevator",
     "Only 13 fault seconds. Raw 0.31, alarm 0.15. Too few seconds to say much."),
    ((590, 490, 765, 545), "Aileron",
     "150 fault seconds. Raw 0.07, alarm 0.04. The sensor features barely react."),
    ((815, 450, 990, 545), "Rudder",
     "44 fault seconds. Raw 0.18, alarm 0.02. The alarm needs 3 in a row, which almost never happens here."),
]

FIG4 = [
    (None, "The whole figure",
     "Confusion matrix of the alarm. Each row is an actual class and sums to 1, so a cell is a share of that class."),
    ((150, 55, 393, 298), "Normal, no alarm: 0.99",
     "99 percent of normal seconds stay quiet."),
    ((393, 55, 636, 298), "Normal, alarm: 0.01",
     "1 percent of normal seconds raise a false alarm."),
    ((150, 298, 393, 541), "Fault, no alarm: 0.62",
     "62 percent of fault seconds are missed."),
    ((393, 298, 636, 541), "Fault, alarm: 0.38",
     "38 percent of fault seconds are caught. This is the recall over all fault types together."),
]

FIG5 = [
    (None, "The whole figure",
     "One grey bar per development flight, out-of-fold predictions. The black line is the fault start. "
     "Red ticks are alarm seconds."),
    ((1070, 80, 1096, 1290), "The black line",
     "Time zero, the fault start. Ticks to the right of it are alarms after the fault."),
    ((20, 95, 1240, 860), "Engine flights",
     "Almost every engine flight has a run of red ticks a few seconds after the fault."),
    ((1045, 185, 1070, 212), "Alarms before the fault",
     "Engine 3 shows two ticks just before the line. These are false alarms."),
    ((20, 870, 1290, 1255), "Elevator, aileron, rudder",
     "Mostly no ticks at all. These faults are missed by the sensor features."),
    ((1030, 955, 1050, 980), "A lone false alarm",
     "Aileron 2 has one tick before the fault and none after."),
]

FIG6 = [
    (None, "The whole figure",
     "Pearson correlation between every pair of the 21 features, all development rows pooled. "
     "Red means they rise together, blue means one rises as the other falls."),
    ((167, 88, 1097, 1017), "The matrix",
     "Rows and columns are the same 21 features, so the matrix mirrors itself across the diagonal."),
    ((167, 88, 211, 132), "The diagonal",
     "Every feature has r = 1 with itself. That is the red diagonal."),
    ((477, 132, 521, 176), "A near-duplicate pair",
     "alt_error__std5 and hud_altitude__std5 look almost the same colour as the diagonal, so they carry almost "
     "the same information. The script prints every pair with |r| above 0.9."),
    ((521, 176, 566, 220), "A strong negative pair",
     "alt_error__delta10 against hud_altitude__delta10 is dark blue: when one goes up the other goes down."),
    ((1153, 28, 1250, 1076), "The colour scale",
     "From -1 (blue) through 0 (grey) to +1 (red)."),
]

FIG8 = [
    (None, "The whole figure",
     "Three models on the same 5 development folds, each trained with 5 seeds. "
     "Threshold 0.5 and 3 seconds in a row for every model."),
    ((100, 52, 806, 602), "Left: F1 per fold",
     "One dot is the mean of 5 seeds for one held-out fold. Grey lines join the same fold across models."),
    ((225, 345, 265, 380), "Forest mean: 0.51",
     "The black diamond is the mean of the five folds. Forest is about 0.51."),
    ((475, 320, 520, 358), "CatBoost mean: 0.54", "CatBoost is about 0.54."),
    ((725, 325, 772, 368), "MLP mean: 0.53",
     "MLP is about 0.53. The three means are close, and the folds spread much more than the models differ."),
    ((918, 52, 1640, 602), "Right: recall vs false alarms",
     "x is false-alarm seconds per minute of normal flight, y is engine-fault recall. Big dot = mean, small dots = folds."),
    ((1280, 235, 1325, 280), "Higher recall, more false alarms",
     "The MLP has the highest recall but also the most false alarms. Forest has the fewest of both."),
]


def main():
    build("gif_fig1.gif", "fig1_engine_flight.png", "Fig 1: one engine flight, three detectors", FIG1)
    build("gif_fig2.gif", "fig2_frozen_channels.png", "Fig 2: frozen control channels", FIG2)
    build("gif_fig3.gif", "fig3_recall_by_type.png", "Fig 3: recall by fault type", FIG3)
    build("gif_fig4.gif", "fig4_confusion_matrix.png", "Fig 4: confusion matrix", FIG4)
    build("gif_fig5.gif", "fig5_alarm_timeline.png", "Fig 5: alarms per flight", FIG5)
    build("gif_fig6.gif", "fig6_feature_correlation.png", "Fig 6: which features duplicate each other", FIG6)
    build("gif_fig8.gif", "fig8_model_comparison.png", "Fig 8: model comparison", FIG8)


if __name__ == "__main__":
    main()
