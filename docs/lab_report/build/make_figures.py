"""Figures for the lab report (docs/lab_report). Every plotted number is read from a
committed report or log; the qualitative and input figures are rendered from a sealed
test frame with the shipped checkpoint, through the same forward path as
scripts/eval_game.py (inference only, no training).

Run from the repository root:  python docs/lab_report/build/make_figures.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/lab_report/figures"

INK, INK2, GRID, MUTED = "#111111", "#4a4a4a", "#dedede", "#b5b5b5"
BLUE, BLUE_L, ORANGE, GREEN, PURPLE = "#1f5fa8", "#8fb6e3", "#d9622b", "#2e7d4f", "#6a4c93"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "axes.edgecolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "savefig.facecolor": "white", "figure.facecolor": "white",
    "legend.frameon": False,
})


def _j(p):
    return json.loads((ROOT / p).read_text())


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------- diagrams
def _box(ax, x, y, w, h, title, lines=(), fc="#eef3fa", ec=BLUE, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=fs + 0.6, fontweight="bold", color=INK)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.30 - i * 0.165, s, ha="center", va="top", fontsize=fs, color=INK2)


def _arrow(ax, a, b, color=INK2, style="-|>", ls="-", rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def _poly(ax, pts, color=INK2, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=color, lw=1.0, ls=ls)


def fig_architecture():
    W, H = 7.4, 5.55
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(0.02, H - 0.02, "A. Offline: data and supervision", fontsize=9.5, fontweight="bold", va="top")
    bw, bh, gx = 1.66, 0.98, 0.20
    xs = [0.02 + i * (bw + gx) for i in range(4)]
    y1 = H - 1.30
    off = [("Godot game", ["main.tscn: player, coins,", "enemies; physics, logic", "and entity IDs (truth)"]),
           ("Structural exporter", ["depth, normals, albedo,", "motion, entity-ID map,", "camera matrices"]),
           ("Dataset recorder", ["24 sessions x 990 frames", ".npz buffers + PNG,", "manifest, schema m2"]),
           ("M2 validator", ["schema, channels,", "sequence semantics;", "VALIDATED marker"])]
    for x, (t, l) in zip(xs, off):
        _box(ax, x, y1, bw, bh, t, l)
    for i in range(3):
        _arrow(ax, (xs[i] + bw, y1 + bh / 2), (xs[i + 1], y1 + bh / 2))
    y2 = y1 - 1.38
    off2 = [("Offline teacher", ["SD 1.5 + ControlNet-depth", "img2img, per style", "(oil 0.50, watercolour 0.55)"]),
            ("Alignment gates", ["boundary-F, spurious edges,", "silhouette, stylisation;", "per-frame exclusion (11)"]),
            ("Training", ["StyleTemporalUNet", "L1 + 0.5 LPIPS + 0.5 L_temp", "AdamW, cosine, 60 epochs"]),
            ("Export", ["LiveStep graph -> ONNX", "-> TensorRT FP16 engine", "(one per style)"])]
    for x, (t, l) in zip(xs, off2):
        _box(ax, x, y2, bw, bh, t, l, fc="#fdf1ea", ec=ORANGE)
    for i in range(3):
        _arrow(ax, (xs[i] + bw, y2 + bh / 2), (xs[i + 1], y2 + bh / 2))
    ym = (y1 + y2 + bh) / 2
    _poly(ax, [(xs[3] + bw / 2, y1), (xs[3] + bw / 2, ym), (xs[0] + bw / 2, ym)])
    _arrow(ax, (xs[0] + bw / 2, ym), (xs[0] + bw / 2, y2 + bh))
    ax.text(xs[1] + bw / 2, ym + 0.04, "validated sessions: native RGB + exported depth -> styled targets",
            ha="left", va="bottom", fontsize=7.4, color=INK2, style="italic")

    ax.text(0.02, y2 - 0.20, "B. Runtime: one iteration per game frame (C++ / GDExtension)", fontsize=9.5,
            fontweight="bold", va="top")
    y3 = y2 - 1.55
    rt = [("Godot tick", ["game advances;", "Godot owns all state"]),
          ("Live capture", ["4 passes: depth, normal,", "albedo, entity ID", "+ camera, transforms"]),
          ("C++ runtime", ["decode buffers; motion", "field; spawned-entity", "(no-history) mask"]),
          ("TensorRT LiveStep", ["in-graph confidence mask,", "warp + fuse history,", "decode with style FiLM"])]
    for x, (t, l) in zip(xs, rt):
        _box(ax, x, y3, bw, bh, t, l, fc="#edf6f0", ec=GREEN)
    for i in range(3):
        _arrow(ax, (xs[i] + bw, y3 + bh / 2), (xs[i + 1], y3 + bh / 2))
    _arrow(ax, (xs[3] + bw / 2, y2), (xs[3] + bw / 2, y3 + bh), ls="--", color=ORANGE)
    ax.text(xs[3] + bw / 2 + 0.06, y3 + bh + 0.16, "engine file", fontsize=7.2, color=ORANGE, va="center")
    y4 = y3 - 1.25
    _box(ax, xs[3], y4, bw, 0.70, "Display", ["styled RGB, 512 x 288"], fc="#edf6f0", ec=GREEN)
    _box(ax, xs[0], y4, bw, 0.70, "Fallback", ["any failure: show native", "frame; game never waits"],
         fc="#f4f4f4", ec=MUTED)
    _arrow(ax, (xs[3] + bw * 0.9, y3), (xs[3] + bw * 0.9, y4 + 0.70))
    # feature feedback: LiveStep output features -> next frame's LiveStep input
    _poly(ax, [(xs[3] + bw * 0.5, y3), (xs[3] + bw * 0.5, y3 - 0.17), (xs[2] + bw * 0.72, y3 - 0.17)], color=GREEN)
    _arrow(ax, (xs[2] + bw * 0.72, y3 - 0.17), (xs[2] + bw * 0.72, y3), color=GREEN)
    ax.text((xs[2] + bw * 0.72 + xs[3] + bw * 0.5) / 2, y3 - 0.24, "feat_t becomes next frame's feat_prev",
            fontsize=7.0, color=GREEN, ha="center", va="top")
    _poly(ax, [(xs[2] + bw * 0.15, y3), (xs[2] + bw * 0.15, y4 + 0.35), (xs[0] + bw, y4 + 0.35)], color=MUTED, ls="--")
    _arrow(ax, (xs[0] + bw + 0.15, y4 + 0.35), (xs[0] + bw, y4 + 0.35), color=MUTED)
    ax.text(xs[1] + 0.1, y4 + 0.40, "on error", fontsize=7.1, color=INK2, va="bottom")
    _save(fig, "architecture.png")


def fig_network():
    W, H = 7.4, 4.9
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(0.15, H - 0.08, "Input x_t: 7 ch, 288x512 (inverse depth, normals, albedo)", fontsize=7.6,
            fontweight="bold", va="top")
    ax.text(W - 0.15, H - 0.08, "Output: RGB, 3 ch, 288x512 (1x1 conv + sigmoid)", fontsize=7.6,
            fontweight="bold", va="top", ha="right")
    enc = [("32 ch", "288x512"), ("64 ch", "144x256"), ("128 ch", "72x128"), ("256 ch", "36x64")]
    bw, bh = 1.0, 0.50
    ys = [H - 1.20 - i * 0.66 for i in range(4)]
    for i, (c, r) in enumerate(enc):
        x, y = 0.15 + i * 0.62, ys[i]
        xd = W - 0.15 - bw - i * 0.62
        _box(ax, x, y, bw, bh, f"Enc {i + 1}", [f"{c}, {r}"], fs=6.7)
        _box(ax, xd, y, bw, bh, f"Dec {i + 1} + FiLM", [f"{c}, {r}"], fc="#fdf1ea", ec=ORANGE, fs=6.7)
        _arrow(ax, (x + bw, y + bh / 2), (xd, y + bh / 2), ls=":", color=MUTED)
        if i < 3:
            _arrow(ax, (x + 0.5, y), (x + 0.5 + 0.12, ys[i + 1] + bh), color=INK2)
            _arrow(ax, (xd + 0.5 - 0.12, ys[i + 1] + bh), (xd + 0.5, y), color=INK2)
    ax.text(W / 2, ys[0] + bh / 2 + 0.05, "skip connections (concatenate)", ha="center", va="bottom", fontsize=7.2,
            color=INK2, style="italic")
    ax.text(0.15 + 0.5 + 0.03, H - 0.42, "", fontsize=1)
    _arrow(ax, (0.65, H - 0.30), (0.65, ys[0] + bh), color=INK2)
    _arrow(ax, (W - 0.65, ys[0] + bh), (W - 0.65, H - 0.30), color=INK2)
    yb = ys[3] - 0.80
    _box(ax, W / 2 - 0.70, yb, 1.40, 0.50, "Bottleneck", ["512 ch, 18x32"], fc="#eef3fa", ec=BLUE, fs=7.0)
    _arrow(ax, (0.15 + 3 * 0.62 + 0.5, ys[3]), (W / 2 - 0.70, yb + 0.25))
    _arrow(ax, (W / 2 + 0.70, yb + 0.25), (W - 0.15 - bw - 3 * 0.62 + 0.5, ys[3]))
    yt = 0.04
    _box(ax, 2.20, yt, 3.00, 0.78, "Temporal fusion (engine-guided)",
         ["warped = W(feat_prev, motion) x M_t", "feat' = feat + M_t x Conv1x1([feat, warped])",
          "(Conv1x1 zero-initialised)"], fc="#edf6f0", ec=GREEN, fs=6.7)
    _arrow(ax, (W / 2, yt + 0.78), (W / 2, yb), color=GREEN)
    _box(ax, 0.10, yt, 1.95, 0.78, "History inputs", ["feat_prev: 512 x 18 x 32", "motion: 2 ch (UV, backward)",
                                                      "confidence mask M_t"], fc="#f4f4f4", ec=MUTED, fs=6.7)
    _arrow(ax, (2.05, yt + 0.41), (2.20, yt + 0.41), color=GREEN)
    _box(ax, 5.35, yt, 1.95, 0.78, "Style table", ["2 styles x 16-d vector", "Linear -> (gamma, beta)",
                                                    "for each decoder level"], fc="#f2edf7", ec=PURPLE, fs=6.7)
    _arrow(ax, (6.32, yt + 0.78), (6.32, ys[3]), color=PURPLE, ls="--")
    _save(fig, "network.png")


# --------------------------------------------------------------------------- real frames
def _frames():
    import torch
    from data_pipeline.teacher.structural import StructuralSession
    from eval.metrics.temporal import engine_confidence
    from renderer.temporal.pairs import FramePair
    from renderer.training.dataset import build_input, target_dir_for
    from renderer.training.train_baseline import load_checkpoint

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = StructuralSession(ROOT / "data_pipeline/recorder/ds_1790538506_f1f9c484")   # sealed test session
    # pick the frame (stride 10) with the most entity pixels, so the player/enemies are visible
    best, f = -1, None
    for fid in s.frame_ids()[10::10]:
        n = int((s.load_structure(fid).entity_id > 0).sum())
        if n > best and s.motion_valid(fid):
            best, f = n, fid
    m = load_checkpoint(ROOT / "checkpoints/game_b_two_style/best.pt", dev).eval()
    st = s.load_structure(f)
    x = torch.from_numpy(build_input(st))[None].to(dev)
    with torch.no_grad():
        pf, _ = m.encode(torch.from_numpy(build_input(s.load_structure(f - 1)))[None].to(dev))
        cf = engine_confidence(s, FramePair(prev_id=f - 1, curr_id=f, tick_delta=1))
        outs, tgts = {}, {}
        for style in m.style_ids:
            y = m(x, pf, cf["motion"].to(dev), cf["mask"].to(dev),
                  style_idx=torch.full((1,), m.style_ids.index(style), device=dev))
            outs[style] = np.transpose(y[0].float().clamp(0, 1).cpu().numpy(), (1, 2, 0))
            tdir = target_dir_for(s.session_id, style)
            rel = {json.loads(l)["frame_id"]: json.loads(l)["target_rgb_path"]
                   for l in (tdir / "manifest.jsonl").read_text().splitlines() if l.strip()}
            from PIL import Image
            tgts[style] = np.asarray(Image.open(tdir / rel[f]).convert("RGB"))
    return s, f, st, cf, outs, tgts


def fig_inputs_and_quality():
    s, f, st, cf, outs, tgts = _frames()
    from renderer.training.dataset import inverse_depth
    depth = st.depth[..., 0] if st.depth.ndim == 3 else st.depth
    nrm = st.normal.astype(np.float32)
    ids = st.entity_id[..., 0] if st.entity_id.ndim == 3 else st.entity_id
    mot = cf["motion"][0].numpy() if hasattr(cf["motion"], "numpy") else np.asarray(cf["motion"][0])
    mag = np.sqrt((mot ** 2).sum(0)) * 512
    mask = cf["mask"][0, 0].numpy()
    panels = [("(a) Native Godot RGB", s.load_native_rgb(f), None),
              ("(b) Inverse depth 1/(1+d)", inverse_depth(depth), "gray"),
              ("(c) Surface normals", np.clip(nrm * 0.5 + 0.5, 0, 1), None),
              ("(d) Albedo", np.clip(st.albedo.astype(np.float32), 0, 1), None),
              ("(e) Entity-ID map (address only)", np.ma.masked_where(ids == 0, ids % 10), "tab10"),
              ("(f) Motion magnitude (px)", mag, "magma"),
              ("(g) Confidence mask M_t", mask, "viridis"),
              ("(h) Teacher target (oil)", tgts["oil_impasto"], None)]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 2.45))
    for a, (t, im, cm) in zip(axes.flat, panels):
        if np.ma.isMaskedArray(im):                     # background (id 0) shown black
            a.imshow(np.zeros(im.shape), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.imshow(im, cmap=cm, interpolation="nearest")
        a.set_title(t, fontsize=7.6, pad=2.5); a.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.18)
    _save(fig, "inputs.png")

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.0))
    nat = s.load_native_rgb(f)
    for r, (style, lab) in enumerate((("oil_impasto", "Oil"), ("watercolor_bold_s055", "Watercolour"))):
        for c, (t, im) in enumerate((("Native render (input structure)", nat),
                                     (f"Teacher target ({lab.lower()})", tgts[style]),
                                     (f"Two-style Model B ({lab.lower()})", outs[style]))):
            a = axes[r, c]; a.imshow(im); a.axis("off")
            a.set_title(t, fontsize=8, pad=2.5)
    fig.subplots_adjust(wspace=0.03, hspace=0.16)
    _save(fig, "qualitative.png")
    (OUT / "qualitative_frame.json").write_text(json.dumps({"session": s.session_id, "frame": int(f)}) + "\n")


# --------------------------------------------------------------------------- results charts
def fig_training():
    t = _j("eval/reports/game_b_two_style_train.json")
    h = t["history"]
    ep = [r["epoch"] for r in h]
    vl = [(r["epoch"], r["val_l1"], r["val_psnr_db"]) for r in h if "val_l1" in r]
    best = min(vl, key=lambda r: r[1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.6))
    a1.plot(ep, [r["train_loss"] for r in h], color=BLUE, lw=1.4, label="training objective")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("Training objective (both styles)", loc="left")
    a1b = a1.twinx(); a1b.plot(ep, [r["train_temporal"] for r in h], color=ORANGE, lw=1.1, ls="--")
    a1b.set_ylabel("temporal term L_temp", color=ORANGE); a1b.tick_params(axis="y", colors=ORANGE)
    a1b.spines["right"].set_visible(True); a1b.grid(False)
    a2.plot([r[0] for r in vl], [r[1] for r in vl], "o-", color=BLUE, ms=3, lw=1.2)
    a2.axvline(best[0], color=INK2, ls=":", lw=1)
    a2.annotate(f"best epoch {best[0]}\nval L1 {best[1]:.6f}", (best[0], best[1]), xytext=(-78, 24),
                textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    a2.set_xlabel("epoch"); a2.set_ylabel("validation L1"); a2.set_title("Validation L1 (5,936 frames)", loc="left")
    a2b = a2.twinx(); a2b.plot([r[0] for r in vl], [r[2] for r in vl], color=GREEN, lw=1.0, ls="--")
    a2b.set_ylabel("validation PSNR (dB)", color=GREEN); a2b.tick_params(axis="y", colors=GREEN)
    a2b.spines["right"].set_visible(True); a2b.grid(False)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "training.png")


def _fps(log):
    t = (ROOT / log).read_text()
    return {s: float(re.search(rf"LIVE_GAME {s}: .*\(([\d.]+) fps\)", t).group(1)) for s in ("oil", "watercolour")}


def _bench(s):
    return float(re.search(r"median ([\d.]+) ms", s).group(1))


def fig_live():
    configs = [("No history (D96)", _fps("eval/logs/live_game_check.log")),
               ("History, FP32 strict", _fps("eval/logs/m10_final/fp32_strict_live.log")),
               ("History, FP16 (shipped)", _fps("eval/logs/m10_final/stability_fp16.log"))]
    eng = _j("eval/reports/m10_live_temporal_engines.json")["styles"]["oil_impasto"]
    fp16 = _j("eval/reports/m10_fp16_live.json")["styles"]["oil_impasto"]
    lat = [("FP32 with TF32", _bench(eng["fp32_tf32"]["bench"])), ("FP32 strict", _bench(fp16["bench_fp32_strict"])),
           ("FP16 (shipped)", _bench(fp16["bench_fp16"]))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    h = 0.36
    for y, (lab, f) in enumerate(configs):
        for dy, s, c in ((-h / 2, "oil", BLUE), (h / 2, "watercolour", BLUE_L)):
            a1.barh(y + dy, f[s], height=h * 0.95, color=c, label=s if y == 0 else None)
            a1.annotate(f"{f[s]:.1f}", (f[s], y + dy), xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=8)
    a1.axvline(30, color=ORANGE, ls="--", lw=1)
    a1.set_yticks(range(len(configs)), [c[0] for c in configs]); a1.invert_yaxis()
    a1.set_xlim(0, 66); a1.set_xlabel("frames per second (whole live loop)")
    a1.grid(axis="y", visible=False); a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=8)
    a1.set_title("(a) Live game, 512x288 (target 30 fps)", loc="left", fontsize=9)
    for y, (lab, v) in enumerate(lat):
        a2.barh(y, v, height=0.55, color=BLUE if "shipped" in lab else MUTED)
        a2.annotate(f"{v:.2f} ms", (v, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    a2.axvline(20, color=ORANGE, ls="--", lw=1)
    a2.set_yticks(range(len(lat)), [l[0] for l in lat]); a2.invert_yaxis()
    a2.set_xlim(0, 23); a2.set_xlabel("TensorRT time per step (ms)")
    a2.grid(axis="y", visible=False); a2.set_title("(b) Engine alone (budget 20 ms)", loc="left", fontsize=9)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "live.png")


THEMES = ["original", "cel", "cyberpunk", "noir", "watercolor", "pixel", "wireframe", "anime", "ps1",
          "vaporwave", "claymation", "woodcut", "steampunk", "dieselpunk", "vector", "gothic", "cosmic"]
THEME_LABEL = {"ps1": "PS1", "watercolor": "Watercolour"}


def fig_screenshot_collages():
    """One collage per scene from docs/screenshots/NN_<theme>_<scene>.png: the original frame
    at 2x2 in the top left, the 16 themes in the remaining cells of a 4x5 grid."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    tw, th, gap = 480, 270, 8
    bold = "/usr/share/fonts/liberation/LiberationSerif-Bold.ttf"
    cells = [(2, 0), (3, 0), (2, 1), (3, 1)] + [(c, r) for r in (2, 3, 4) for c in range(4)]
    for scene in ("street", "swing"):
        canvas = Image.new("RGB", (4 * tw + 3 * gap, 5 * th + 4 * gap), "white")
        for i, theme in enumerate(THEMES):
            src = Image.open(ROOT / f"docs/screenshots/{i:02d}_{theme}_{scene}.png").convert("RGB")
            size, (c, r) = ((2 * tw + gap, 2 * th + gap), (0, 0)) if i == 0 else ((tw, th), cells[i - 1])
            tile = ImageOps.fit(src, size, Image.LANCZOS).convert("RGBA")
            font = ImageFont.truetype(bold, 46 if i == 0 else 34)
            text = THEME_LABEL.get(theme, theme.capitalize())
            overlay = Image.new("RGBA", size, (0, 0, 0, 0))
            d = ImageDraw.Draw(overlay)
            x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
            pad = 8
            box = (0, size[1] - (y1 - y0) - 2 * pad - 4, x1 - x0 + 2 * pad + 6, size[1])
            d.rectangle(box, fill=(0, 0, 0, 170))
            d.text((box[0] + pad + 3 - x0, box[1] + pad - y0), text, font=font, fill="white")
            canvas.paste(Image.alpha_composite(tile, overlay).convert("RGB"), (c * (tw + gap), r * (th + gap)))
        OUT.mkdir(parents=True, exist_ok=True)
        canvas.save(OUT / f"collage_{scene}.jpg", quality=92, subsampling=0)
        print("wrote", OUT / f"collage_{scene}.jpg")


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    for fn in (fig_architecture, fig_network, fig_inputs_and_quality, fig_training, fig_live,
               fig_screenshot_collages):
        if not only or fn.__name__ in only:
            fn()
