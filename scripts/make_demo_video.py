"""Compose the demo frames from game_project/tests/runtime/demo_record.gd into one
side-by-side video: native Godot render | live StructRender (two-style B).

Frames are used exactly as saved (upscaled 2x for viewing); nothing is retouched.
Output: eval/reports/demo/structrender_demo.mp4 (60 fps = real game time) and a
poster frame per style.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from renderer.training.train_baseline import ROOT

STYLE = {0: ("OIL", (214, 158, 46)), 1: ("WATERCOLOUR", (86, 156, 214))}
FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
BOLD = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
W, H, TOP, BOT, GAP = 1024, 576, 64, 44, 8


def frame(d: Path, i: int, f_big, f_small) -> Image.Image:
    nat = Image.open(d / f"native_{i:04d}.png").convert("RGB").resize((W, H), Image.LANCZOS)
    sty = Image.open(d / f"styled_{i:04d}.png").convert("RGB").resize((W, H), Image.LANCZOS)
    s = int((d / f"style_{i:04d}.txt").read_text())
    name, col = STYLE[s]
    c = Image.new("RGB", (2 * W + GAP, TOP + H + BOT), (16, 18, 24))
    c.paste(nat, (0, TOP)); c.paste(sty, (W + GAP, TOP))
    g = ImageDraw.Draw(c)
    g.text((20, 16), "Native Godot render", font=f_big, fill=(220, 222, 228))
    g.text((W + GAP + 20, 16), "StructRender, live:", font=f_big, fill=(220, 222, 228))
    x = W + GAP + 20 + g.textlength("StructRender, live: ", font=f_big)
    g.text((x, 16), name, font=f_big, fill=col)
    g.text((20, TOP + H + 10),
           "Same game, same physics, same objects - only the look changes.  Two-style Model B, TensorRT FP16, "
           "frame history fed back.  Every frame as the live renderer produced it (lockstep recording, 60 fps game time).",
           font=f_small, fill=(150, 154, 164))
    return c


def main(src="artifacts/demo_frames") -> int:
    d = ROOT / src
    n = len(list(d.glob("styled_*.png")))
    out = ROOT / "eval/reports/demo"; out.mkdir(parents=True, exist_ok=True)
    f_big, f_small = ImageFont.truetype(BOLD, 30), ImageFont.truetype(FONT, 17)
    first = frame(d, 0, f_big, f_small)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{first.width}x{first.height}", "-r", "60", "-i", "-",
                          "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", str(out / "structrender_demo.mp4")], stdin=subprocess.PIPE)
    posters = {}
    for i in range(n):
        im = first if i == 0 else frame(d, i, f_big, f_small)
        p.stdin.write(im.tobytes())
        s = int((d / f"style_{i:04d}.txt").read_text())
        if s not in posters and i % 300 == 150:
            posters[s] = im; im.save(out / f"poster_{STYLE[s][0].lower()}.png")
    p.stdin.close(); p.wait()
    print("frames", n, "->", out / "structrender_demo.mp4", "exit", p.returncode)
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
