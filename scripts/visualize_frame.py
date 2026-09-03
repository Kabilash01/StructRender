#!/usr/bin/env python3
"""Preview the Milestone 1 structural buffers as PNGs. Pure stdlib (no numpy /
no Pillow) so it runs anywhere, headless.

Usage:
    python3 scripts/visualize_frame.py <dump_dir> [--frame N] [--out <dir>]

<dump_dir> is a directory written by StructuralExporter.dump_frame(), containing
    frame_<N>_depth.npy      float32 [H,W,1]
    frame_<N>_normal.npy     float32 [H,W,3]   raw [-1,1]
    frame_<N>_albedo.npy     float32 [H,W,3]   linear
    frame_<N>_motion.npy     float32 [H,W,2]   backward screen-UV, Y-down
    frame_<N>_entity_id.npy  int32   [H,W]

See data_pipeline/schemas/frame_structure.md for the exact channel semantics.
"""
from __future__ import annotations

import argparse
import ast
import math
import os
import struct
import sys
import zlib


def read_npy(path):
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path}: not a .npy file")
        major, _minor = f.read(2)
        hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        header = ast.literal_eval(f.read(hlen).decode("latin1"))
        descr, shape = header["descr"], header["shape"]
        data = f.read()
    if descr in ("<f4", "|f4", "=f4"):
        fmt, size = "f", 4
    elif descr in ("<i4", "|i4", "=i4"):
        fmt, size = "i", 4
    else:
        raise ValueError(f"{path}: unsupported dtype {descr}")
    n = len(data) // size
    flat = list(struct.unpack("<" + fmt * n, data[: n * size]))
    return flat, shape


def write_png(path, w, h, rgb):
    """rgb: flat list of 3*w*h ints in 0..255."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = bytearray()
    for y in range(h):
        raw.append(0)
        row = y * w * 3
        raw.extend(bytes(rgb[row: row + w * 3]))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def _clamp8(v):
    return 0 if v < 0 else (255 if v > 255 else int(v))


def viz_depth(flat, w, h):
    finite = [v for v in flat if math.isfinite(v)]
    lo = min(finite) if finite else 0.0
    hi = max(finite) if finite else 1.0
    span = (hi - lo) or 1.0
    out = []
    for v in flat:
        if not math.isfinite(v):
            out += [255, 0, 255]  # sentinel -> magenta
        else:
            g = _clamp8((1.0 - (v - lo) / span) * 255)  # near = bright
            out += [g, g, g]
    return out


def viz_normal(flat, w, h):
    out = []
    for i in range(w * h):
        nx, ny, nz = flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]
        out += [_clamp8((nx * 0.5 + 0.5) * 255),
                _clamp8((ny * 0.5 + 0.5) * 255),
                _clamp8((nz * 0.5 + 0.5) * 255)]
    return out


def viz_albedo(flat, w, h):
    out = []
    for i in range(w * h):
        for c in range(3):
            lin = max(0.0, flat[i * 3 + c])
            out.append(_clamp8((lin ** (1.0 / 2.2)) * 255))
    return out


def _hsv(hue, sat, val):
    i = int(hue * 6) % 6
    f = hue * 6 - int(hue * 6)
    p, q, t = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    r, g, b = [(val, t, p), (q, val, p), (p, val, t),
               (p, q, val), (t, p, val), (val, p, q)][i]
    return [_clamp8(r * 255), _clamp8(g * 255), _clamp8(b * 255)]


def viz_motion(flat, w, h, scale=40.0):
    out = []
    for i in range(w * h):
        mx, my = flat[i * 2], flat[i * 2 + 1]
        mag = min(1.0, math.hypot(mx, my) * scale)
        ang = (math.atan2(my, mx) / (2 * math.pi)) % 1.0
        out += _hsv(ang, 1.0, mag)
    return out


def viz_id(flat, w, h):
    out = []
    for v in flat:
        if v == 0:
            out += [20, 20, 24]
        else:
            out += _hsv(((v * 2654435761) % 997) / 997.0, 0.65, 1.0)
    return out


VIZ = {
    "depth": (viz_depth, 1),
    "normal": (viz_normal, 3),
    "albedo": (viz_albedo, 3),
    "motion": (viz_motion, 2),
    "entity_id": (viz_id, 1),
}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(args.dump_dir, "png")
    os.makedirs(out_dir, exist_ok=True)

    frames = set()
    for name in os.listdir(args.dump_dir):
        if name.startswith("frame_") and name.endswith(".npy"):
            try:
                frames.add(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    if args.frame is not None:
        frames = {args.frame}
    if not frames:
        print(f"no frame_*.npy in {args.dump_dir}", file=sys.stderr)
        return 1

    for fr in sorted(frames):
        for chan, (fn, _nc) in VIZ.items():
            src = os.path.join(args.dump_dir, f"frame_{fr}_{chan}.npy")
            if not os.path.exists(src):
                continue
            flat, shape = read_npy(src)
            hgt, wid = shape[0], shape[1]
            rgb = fn(flat, wid, hgt)
            dst = os.path.join(out_dir, f"frame_{fr}_{chan}.png")
            write_png(dst, wid, hgt, rgb)
            print(f"wrote {dst}  ({wid}x{hgt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
