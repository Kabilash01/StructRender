"""Figures for docs/REPORT.md — every number read from a committed report or log.

Nothing here is typed in by hand: each value comes from the file named beside
it, so a figure cannot drift from the evidence it summarises.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"

# reference palette (dataviz skill, light mode); blue/orange validated as a pair
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b2"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#86b6ef", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _j(p):
    return json.loads((ROOT / p).read_text())


def _fps(log):
    t = (ROOT / log).read_text()
    return {s: float(re.search(rf"LIVE_GAME {s}: .*\(([\d.]+) fps\)", t).group(1)) for s in ("oil", "watercolour")}


def _bench(s):
    return float(re.search(r"median ([\d.]+) ms", s).group(1))


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_ladder():
    """Dot plots (no zero baseline needed): the M9 ladder on the sealed test split."""
    r = _j("eval/reports/m9_ladder.json")["rungs"]
    names = {"A": "A  U-Net", "B": "B  U-Net + temporal", "C": "C  StructFormer",
             "D": "D  StructFormer + temporal", "E": "E  + entity masks", "N": "N  raw-ID control"}
    keys = [k for k in ("A", "B", "C", "D", "E", "N") if k in r][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, metric, title, fmt in ((axes[0], "psnr_db", "PSNR (dB) — higher is better", "{:.2f}"),
                                   (axes[1], "sharpness", "Sharpness vs teacher — gate 0.90", "{:.3f}")):
        for y, k in enumerate(keys):
            v = r[k][metric]
            c = BLUE if k == "A" else MUTED
            ax.plot([v], [y], "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=3)
            ax.annotate(fmt.format(v), (v, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK if k == "A" else INK2)
        if metric == "sharpness":
            ax.axvline(0.90, color=INK2, ls=(0, (4, 3)), lw=1.2)
            ax.text(0.90, 1.01, " gate", transform=ax.get_xaxis_transform(), fontsize=9.5, color=INK2,
                    va="bottom")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        lo, hi = min(r[k][metric] for k in keys), max(r[k][metric] for k in keys)
        pad = (hi - lo) * 0.35
        ax.set_xlim(lo - pad, max(hi, 0.90 if metric == "sharpness" else hi) + pad * 1.6)
    axes[0].set_yticks(range(len(keys)), [names[k] for k in keys])
    fig.suptitle("Ablation ladder, sealed test split (3,960 frames): the plain U-Net (A) leads on both",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "fig1_ladder.png")


def fig_flicker():
    """Grouped bars from zero: temporal error on moving pixels, game test sessions."""
    rows = [("Oil", _j("eval/reports/game_b_two_style_test_oil.json")),
            ("Watercolour", _j("eval/reports/game_b_two_style_test_watercolor.json"))]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.34
    for i, (lab, d) in enumerate(rows):
        t, m = d["temporal_error"]["target"]["mean"], d["temporal_error"]["model"]["mean"]
        for dx, v, c in ((-w / 2 - 0.01, t, ORANGE), (w / 2 + 0.01, m, BLUE)):
            ax.bar(i + dx, v, width=w, color=c, edgecolor=SURFACE, linewidth=2)
            ax.annotate(f"{v:.4f}", (i + dx, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, color=INK)
        ax.annotate(f"{m / t:.2f}x the teacher", (i + w / 2, m), xytext=(0, 20), textcoords="offset points",
                    ha="center", fontsize=10, color=INK2)
    nat = rows[0][1]["temporal_error"]["native"]["mean"]
    ax.set_xticks([0, 1], [r[0] for r in rows])
    ax.set_ylabel("temporal error (lower = steadier)")
    ax.set_ylim(0, max(r[1]["temporal_error"]["target"]["mean"] for r in rows) * 1.3)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["teacher images", "two-style Model B"], frameon=False, loc="upper right")
    ax.set_title(f"Flicker on the game's test sessions (native engine: {nat:.4f})")
    _save(fig, "fig2_flicker.png")


def fig_history():
    """Dumbbells: the same model with and without its one-frame history."""
    pairs = [("Oil", "oil", "oil_impasto"), ("Watercolour", "watercolor", "watercolor_bold_s055")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), sharey=True)
    for ax, metric, title, better in ((axes[0], "psnr_db", "PSNR (dB) — higher is better", "hi"),
                                      (axes[1], "lpips", "LPIPS — lower is better", "lo")):
        for y, (lab, a, b) in enumerate(pairs):
            w = _j(f"eval/reports/game_b_two_style_test_{a}.json")["image_quality"][metric]
            n = _j(f"eval/reports/game_b_two_style_test_{b}_nohistory.json")["image_quality"][metric]
            ax.plot([n, w], [y, y], color=GRID, lw=3, zorder=1)
            ax.plot([n], [y], "o", ms=10, color=BLUE_LIGHT, mec=SURFACE, mew=2, zorder=3)
            ax.plot([w], [y], "o", ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=3)
            fmt = "{:.2f}" if metric == "psnr_db" else "{:.4f}"
            for v, c in ((n, INK2), (w, INK)):
                ax.annotate(fmt.format(v), (v, y), xytext=(0, 11), textcoords="offset points",
                            ha="center", fontsize=10, color=c)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25, y=0.6)
    axes[0].set_yticks([0, 1], [p[0] for p in pairs])
    axes[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=9, color=BLUE_LIGHT),
                            plt.Line2D([], [], marker="o", ls="", ms=9, color=BLUE)],
                   labels=["no history (cold start every frame)", "with history (as trained, as live now)"],
                   frameon=False, loc="lower center", bbox_to_anchor=(-0.1, -0.42), ncol=2)
    fig.suptitle("What the frame-to-frame memory buys, same frames, same model",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.04)
    _save(fig, "fig3_history.png")


def fig_live():
    """Two panels, one measure each: live fps, and standalone engine latency."""
    configs = [("No history (Decision 96)", _fps("eval/logs/live_game_check.log")),
               ("With history, FP32 strict", _fps("eval/logs/m10_final/fp32_strict_live.log")),
               ("With history, FP16 (ships)", _fps("eval/logs/m10_final/stability_fp16.log"))]
    eng = _j("eval/reports/m10_live_temporal_engines.json")["styles"]["oil_impasto"]
    fp16 = _j("eval/reports/m10_fp16_live.json")["styles"]["oil_impasto"]
    lat = [("FP32, TF32 on", _bench(eng["fp32_tf32"]["bench"])),
           ("FP32 strict", _bench(fp16["bench_fp32_strict"])),
           ("FP16 (ships)", _bench(fp16["bench_fp16"]))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})
    h = 0.36
    for y, (lab, f) in enumerate(configs):
        for dy, s, c in ((-h / 2 - 0.01, "oil", BLUE), (h / 2 + 0.01, "watercolour", BLUE_LIGHT)):
            a1.barh(y + dy, f[s], height=h, color=c, edgecolor=SURFACE, linewidth=2)
            a1.annotate(f"{f[s]:.1f}", (f[s], y + dy), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9.5, color=INK)
    a1.axvline(30, color=INK2, ls=(0, (4, 3)), lw=1.2)
    a1.text(30, 1.01, " target ~30 fps", transform=a1.get_xaxis_transform(), fontsize=9.5, color=INK2,
            va="bottom")
    a1.set_yticks(range(len(configs)), [c[0] for c in configs]); a1.invert_yaxis()
    a1.set_xlim(0, 60); a1.set_xlabel("frames per second, whole live loop (higher is better)")
    a1.grid(axis="y", visible=False)
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=BLUE_LIGHT)],
              labels=["oil", "watercolour"], frameon=False, loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
    a1.set_title("Live game, 512x288, RTX 5060 Laptop", pad=16)
    for y, (lab, v) in enumerate(lat):
        a2.barh(y, v, height=0.55, color=BLUE if "ships" in lab else MUTED, edgecolor=SURFACE, linewidth=2)
        a2.annotate(f"{v:.2f} ms", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=9.5, color=INK)
    a2.axvline(20, color=INK2, ls=(0, (4, 3)), lw=1.2)
    a2.text(20, 1.01, "budget ~20 ms ", transform=a2.get_xaxis_transform(), fontsize=9.5, color=INK2,
            va="bottom", ha="right")
    a2.set_yticks(range(len(lat)), [l[0] for l in lat]); a2.invert_yaxis()
    a2.set_xlim(0, 22); a2.set_xlabel("TensorRT time per live step, ms (lower is better)")
    a2.grid(axis="y", visible=False)
    a2.set_title("Engine alone", pad=16)
    _save(fig, "fig4_live.png")


if __name__ == "__main__":
    fig_ladder(); fig_flicker(); fig_history(); fig_live()
