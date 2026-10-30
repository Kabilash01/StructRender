"""Figures for the conference paper (docs/paper). Every plotted number is read from a committed
report or log, as in docs/lab_report/build/make_figures.py; the charts are redrawn here at IEEE
column width (3.5 in) with one axis each. The two diagrams are drawn by the lab-report script
(saved here as vector PDF as well), and the two inference figures are the lab report's PNGs.

Run from the repository root:  python docs/paper/build/make_paper_figures.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/paper/figures"
LAB_FIGS = ROOT / "docs/lab_report/figures"

# dataviz reference palette, light mode; blue/orange validated as a categorical pair
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b2"
BLUE, BLUE_L, ORANGE = "#2a78d6", "#86b6ef", "#eb6834"
COL_W = 3.45  # inches; IEEEtran \columnwidth is 3.5 in


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.titlelocation": "left", "axes.titleweight": "bold",
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.linewidth": 0.6, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _j(p):
    return json.loads((ROOT / p).read_text())


def _save(fig, name, pdf=True):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    if pdf:
        fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", OUT / stem)


def _label(ax, v, y, text, dx=4, color=INK):
    ax.annotate(text, (v, y), xytext=(dx, 0), textcoords="offset points", va="center", fontsize=7, color=color)


# --------------------------------------------------------------------------- figures
def fig_teaser():
    """The same live frame three ways, cut from the demo posters (rows 64-640 hold the 1024x576 panels;
    the native halves of the two posters are pixel-identical)."""
    oil = np.asarray(Image.open(ROOT / "eval/reports/demo/poster_oil.png").convert("RGB"))
    wat = np.asarray(Image.open(ROOT / "eval/reports/demo/poster_watercolour.png").convert("RGB"))
    assert (oil[:, :1024] == wat[:, :1024]).all(), "posters do not show the same native frame"
    panels = [(oil[64:640, :1024], "(a) Native Godot render"), (oil[64:640, 1032:], "(b) StructRender, live: oil"),
              (wat[64:640, 1032:], "(c) StructRender, live: watercolour")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.55))
    for ax, (img, t) in zip(axes, panels):
        ax.imshow(img); ax.set_title(t, fontsize=8, fontweight="normal", pad=3); ax.axis("off")
    fig.subplots_adjust(wspace=0.03, left=0, right=1, top=0.88, bottom=0)
    _save(fig, "teaser.png", pdf=False)


def fig_diagrams():
    """Architecture and network diagrams, drawn by the lab-report script, saved here as PDF + PNG."""
    spec = importlib.util.spec_from_file_location("lab_figs", ROOT / "docs/lab_report/build/make_figures.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    mf._save = _save
    mf.fig_architecture()
    mf.fig_network()
    _style()  # the lab-report module set its own rcParams on import


def fig_lab_rasters():
    """Rendered with the shipped checkpoint on a sealed test frame by the lab-report script."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("inputs.png", "qualitative.png"):
        shutil.copyfile(LAB_FIGS / f, OUT / f)
        print("copied", OUT / f)


def fig_training():
    h = _j("eval/reports/game_b_two_style_train.json")["history"]
    ep = [r["epoch"] for r in h]
    vl = [(r["epoch"], r["val_l1"]) for r in h if "val_l1" in r]
    best = min(vl, key=lambda r: r[1])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(COL_W, 3.1), sharex=True)
    a1.plot(ep, [r["train_loss"] for r in h], color=BLUE, lw=1.4)
    a1.set_title("(a) Training objective, both styles")
    a2.plot(ep, [r["train_temporal"] for r in h], color=BLUE, lw=1.4)
    a2.set_title("(b) Temporal term of the objective")
    a3.plot([r[0] for r in vl], [r[1] for r in vl], "o-", color=BLUE, ms=3, lw=1.2, mec="white", mew=0.6)
    a3.axvline(best[0], color=INK2, ls=":", lw=0.9)
    a3.annotate(f"best: epoch {best[0]}, {best[1]:.6f}", (best[0], best[1]), xytext=(-6, 14),
                textcoords="offset points", ha="right", fontsize=7, color=INK)
    a3.set_title("(c) Validation L1 (5,936 frames)")
    a3.set_xlabel("epoch")
    for a in (a1, a2, a3):
        a.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=0.6)
    _save(fig, "training.png")


def fig_ladder():
    r = _j("eval/reports/m9_ladder.json")["rungs"]
    names = {"A": "A  U-Net", "B": "B  A + temporal", "C": "C  StructFormer", "D": "D  C + temporal",
             "E": "E  C + entity masks", "N": "N  raw-ID control"}
    keys = [k for k in ("A", "B", "C", "D", "E", "N") if k in r][::-1]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.0))
    for ax, metric, title, fmt in ((axes[0], "psnr_db", "(a) PSNR (dB), higher is better", "{:.2f}"),
                                   (axes[1], "sharpness", "(b) Sharpness vs teacher, gate 0.90", "{:.3f}")):
        for y, k in enumerate(keys):
            v = r[k][metric]
            ax.plot([v], [y], "o", ms=6, color=BLUE if k == "A" else MUTED, mec="white", mew=1, zorder=3)
            _label(ax, v, y, fmt.format(v), dx=5, color=INK if k == "A" else INK2)
        if metric == "sharpness":
            ax.axvline(0.90, color=INK2, ls=(0, (4, 3)), lw=0.9)
        ax.set_title(title)
        ax.set_yticks(range(len(keys)), [names[k] for k in keys])
        ax.grid(axis="y", visible=False)
        lo, hi = min(r[k][metric] for k in keys), max(r[k][metric] for k in keys)
        pad = (hi - lo) * 0.3
        ax.set_xlim(lo - pad, max(hi, 0.90 if metric == "sharpness" else hi) + pad * 2.0)
    fig.tight_layout(h_pad=0.8)
    _save(fig, "ladder.png")


def fig_flicker():
    rows = [("Oil", _j("eval/reports/game_b_two_style_test_oil.json")),
            ("Watercolour", _j("eval/reports/game_b_two_style_test_watercolor.json"))]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    w = 0.34
    for i, (lab, d) in enumerate(rows):
        t, m = d["temporal_error"]["target"]["mean"], d["temporal_error"]["model"]["mean"]
        for dx, v, c in ((-w / 2 - 0.01, t, MUTED), (w / 2 + 0.01, m, BLUE)):
            ax.bar(i + dx, v, width=w, color=c, edgecolor="white", linewidth=1)
            ax.annotate(f"{v:.4f}", (i + dx, v), xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=7, color=INK)
    nat = rows[0][1]["temporal_error"]["native"]["mean"]
    ratios = [d["temporal_error"]["model"]["mean"] / d["temporal_error"]["target"]["mean"] for _, d in rows]
    ax.set_xticks([0, 1], [f"{r[0]}\nmodel {q:.2f}x the teacher" for r, q in zip(rows, ratios)])
    ax.set_ylabel("temporal error")
    ax.set_ylim(0, max(r[1]["temporal_error"]["target"]["mean"] for r in rows) * 1.3)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["teacher targets", "two-style model"], loc="upper right", ncol=2,
              bbox_to_anchor=(1.0, 1.12))
    ax.text(0.0, -0.36, f"native engine: {nat:.5f} (lower is steadier)", transform=ax.transAxes, fontsize=7,
            color=INK2)
    _save(fig, "flicker.png")


def fig_history():
    pairs = [("Oil", "oil", "oil_impasto"), ("Watercolour", "watercolor", "watercolor_bold_s055")]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.2))
    for ax, metric, title in ((axes[0], "psnr_db", "(a) PSNR (dB), higher is better"),
                              (axes[1], "lpips", "(b) LPIPS, lower is better")):
        for y, (lab, a, b) in enumerate(pairs):
            w = _j(f"eval/reports/game_b_two_style_test_{a}.json")["image_quality"][metric]
            n = _j(f"eval/reports/game_b_two_style_test_{b}_nohistory.json")["image_quality"][metric]
            ax.plot([n, w], [y, y], color=GRID, lw=2.5, zorder=1)
            ax.plot([n], [y], "o", ms=6, color=BLUE_L, mec="white", mew=1, zorder=3)
            ax.plot([w], [y], "o", ms=6, color=BLUE, mec="white", mew=1, zorder=3)
            fmt = "{:.2f}" if metric == "psnr_db" else "{:.4f}"
            for v, c in ((n, INK2), (w, INK)):
                ax.annotate(fmt.format(v), (v, y), xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=7, color=c)
        ax.set_title(title)
        ax.set_yticks([0, 1], [p[0] for p in pairs])
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.2, y=0.55)
    axes[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=5, color=BLUE_L),
                            plt.Line2D([], [], marker="o", ls="", ms=5, color=BLUE)],
                   labels=["no history (cold start)", "with history (as deployed)"], loc="upper center",
                   bbox_to_anchor=(0.45, -0.28), ncol=2)
    fig.tight_layout(h_pad=0.6)
    _save(fig, "history.png")


def _fps(log):
    t = (ROOT / log).read_text()
    return {s: float(re.search(rf"LIVE_GAME {s}: .*\(([\d.]+) fps\)", t).group(1)) for s in ("oil", "watercolour")}


def _bench(s):
    return float(re.search(r"median ([\d.]+) ms", s).group(1))


def fig_live():
    configs = [("No history", _fps("eval/logs/live_game_check.log")),
               ("History, FP32 strict", _fps("eval/logs/m10_final/fp32_strict_live.log")),
               ("History, FP16 (shipped)", _fps("eval/logs/m10_final/stability_fp16.log"))]
    eng = _j("eval/reports/m10_live_temporal_engines.json")["styles"]["oil_impasto"]
    fp16 = _j("eval/reports/m10_fp16_live.json")["styles"]["oil_impasto"]
    lat = [("FP32 with TF32", _bench(eng["fp32_tf32"]["bench"])), ("FP32 strict", _bench(fp16["bench_fp32_strict"])),
           ("FP16 (shipped)", _bench(fp16["bench_fp16"]))]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.0), gridspec_kw={"height_ratios": [1.45, 1]})
    h = 0.36
    for y, (lab, f) in enumerate(configs):
        for dy, s, c in ((-h / 2, "oil", BLUE), (h / 2, "watercolour", ORANGE)):
            a1.barh(y + dy, f[s], height=h * 0.92, color=c, edgecolor="white", linewidth=0.8,
                    label=s if y == 0 else None)
            _label(a1, f[s], y + dy, f"{f[s]:.1f}", dx=3)
    a1.axvline(30, color=INK2, ls=(0, (4, 3)), lw=0.9)
    a1.text(30, 1.0, " target 30 fps", transform=a1.get_xaxis_transform(), fontsize=7, color=INK2, va="bottom")
    a1.set_yticks(range(len(configs)), [c[0] for c in configs]); a1.invert_yaxis()
    a1.set_xlim(0, 60); a1.set_xlabel("frames per second, whole live loop")
    a1.grid(axis="y", visible=False)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2)
    a1.set_title("(a) Live game, 512 x 288", pad=10)
    for y, (lab, v) in enumerate(lat):
        a2.barh(y, v, height=0.55, color=BLUE if "shipped" in lab else MUTED, edgecolor="white", linewidth=0.8)
        _label(a2, v, y, f"{v:.2f} ms", dx=3)
    a2.axvline(20, color=INK2, ls=(0, (4, 3)), lw=0.9)
    a2.text(20, 1.0, "budget 20 ms ", transform=a2.get_xaxis_transform(), fontsize=7, color=INK2, va="bottom",
            ha="right")
    a2.set_yticks(range(len(lat)), [l[0] for l in lat]); a2.invert_yaxis()
    a2.set_xlim(0, 22); a2.set_xlabel("TensorRT time per live step (ms)")
    a2.grid(axis="y", visible=False)
    a2.set_title("(b) Network alone", pad=10)
    fig.tight_layout(h_pad=0.8)
    _save(fig, "live.png")


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    _style()
    for fn in (fig_teaser, fig_diagrams, fig_lab_rasters, fig_training, fig_ladder, fig_flicker, fig_history,
               fig_live):
        if not only or fn.__name__ in only:
            fn()
