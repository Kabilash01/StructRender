"""Alignment metrics — edge/gradient domain only.

Why not SSIM/LPIPS against the native RGB: a teacher target is *supposed* to look
different (KB §3C). Pixel-domain similarity to the native frame would punish
exactly the stylisation we asked for. What must survive is **structure**, so all
three metrics compare the target's edge map against edge maps derived exactly
from the engine's own structural buffers (KB §12 *Teacher Structural
Distortion*).

Two structural maps are derived, and they are not the same thing:

``reference`` (R)
    depth-discontinuity edges ∪ ``entity_id`` boundaries. The layout that MUST
    survive. Recall and per-entity silhouette support are measured against it.

``support`` (S)
    R ∪ normal discontinuities ∪ albedo discontinuities. A superset: an edge in
    the target that lands on a material boundary or a surface crease is
    *explained* by the scene even though it is not a silhouette. Precision and
    the spurious-edge ratio are measured against S, because scoring a painterly
    stroke along a texture boundary as "hallucinated geometry" would make the
    §12 guard fire on legitimate stylisation. Both maps come only from the M2
    structural buffers — never from the native RGB, and never from the target.

Asymmetry that matters (calibrated, see KB Decision 32): recall is measured on
the *permissive* target edge set (did the layout survive at all — in a
low-contrast scene a legitimate boundary can be rendered softly), precision on
the *strong* target edge set (are the confident edges explained by the scene).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapter import FrameStructure
from .imageops import dilate, gradient_magnitude, luma, nms_thin, shift, valid_shift_mask

_NBRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass(frozen=True)
class Thresholds:
    """Every number the alignment verdict depends on, in one place.

    Calibrated over the full 990-frame validated seed session with the real
    teacher (SD 1.5 + ControlNet-depth, ``gouache`` at strength 0.45 /
    controlnet 1.3) and pinned in **KB Decision 32**. Observed, 0 frames
    failing — min / p05 / median / max:

        boundary_f              0.709 / 0.878 / 0.980 / 1.000   (gate >= 0.60)
        spurious_edge_ratio     0.000 / 0.000 / 0.000 / 0.337   (gate <= 0.35)
        min_silhouette_support  0.661 / 0.818 / 0.993 / 1.000   (gate >= 0.60)
        stylisation_hist        0.232 / 0.310 / 0.509 / 0.955   (gate >= 0.15)

    Every threshold was fixed before this data was seen, so the pass is genuine
    rather than fitted. **Gate 2's margin is thin** — the worst frame is 0.337
    against a 0.35 limit. If a busier scene crosses it, re-examine whether those
    edges are genuinely unsupported (the reference/support split above exists
    because scoring painterly strokes along texture boundaries as hallucinated
    geometry produced false positives); do not simply raise the number.

    Gate 4 is not optional: an untouched native frame scores boundary_f 0.976
    with spurious 0.000, so gates 1-3 alone are all satisfied by a teacher that
    does nothing.

    **Translation detection floor.** Measured on this scene: 3 px for a diagonal
    shift, but **8 px for an x-only shift** — at 5 px x-only the worst frame
    still scores F=0.717 / spurious=0.294 / silhouette=0.630, inside every gate.
    A systematic sub-5 px horizontal misalignment therefore passes all four gates
    and would reach training as supervision. See ``test_alignment_rejects.py``,
    which pins both floors.

    **Partial circularity.** The teacher is conditioned on the exported depth
    buffer and gate 1's recall is measured against a reference map derived from
    that same buffer. The ``support`` map adds normal and albedo discontinuities,
    which are not conditioning inputs, so this is partly mitigated — but these
    metrics cannot detect a distortion that ControlNet-depth is itself blind to.
    """

    # --- gates ---
    boundary_f: float = 0.60            # gate 1: F-measure vs the reference edge map
    spurious_ratio: float = 0.35        # gate 2: strong target edges with no structural support
    silhouette_support: float = 0.60    # gate 3: per-entity silhouette coverage
    min_entity_px: int = 200            # gate 3 only applies to entities this large
    min_stylisation: float = 0.15       # gate 4: colour-histogram L1 distance from the native frame

    # --- how the maps are built ---
    tolerance_px: int = 2               # boundary matching tolerance
    edge_threshold: float = 0.008       # target edge (post-NMS Sobel, [0,1] luma units)
    strong_edge_threshold: float = 0.04 # "strong" target edge for the hallucination gate
    depth_rel_jump: float = 0.05        # relative depth step counted as a discontinuity
    normal_angle_deg: float = 30.0      # normal crease angle
    albedo_jump: float = 0.05           # linear-albedo step counted as a material boundary

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# structural (engine ground-truth) edge maps
# --------------------------------------------------------------------------- #
def reference_edges(structure: FrameStructure, th: Thresholds) -> np.ndarray:
    """Depth-discontinuity edges ∪ entity-ID boundaries, exact from the buffers.

    Thin by construction: a silhouette pixel is assigned to the *nearer* surface
    (the occluder owns its boundary), and entity boundaries are marked on the
    entity side. Image-border neighbours are not compared — a mesh running off
    the edge of the frame is not a boundary in the image.
    """
    depth = structure.depth
    eid = structure.entity_id
    shape = depth.shape
    fin = np.isfinite(depth) & (depth > 0.0)
    d = np.where(fin, depth, 1.0).astype(np.float64)

    edges = np.zeros(shape, dtype=bool)
    for dy, dx in _NBRS:
        valid = valid_shift_mask(shape, dy, dx)
        nfin = shift(fin, dy, dx)
        nd = shift(d, dy, dx)

        # foreground surface against background: this pixel is the occluder
        edges |= valid & fin & ~nfin

        both = valid & fin & nfin
        denom = np.minimum(d, nd)
        rel = np.abs(d - nd) / np.maximum(denom, 1e-6)
        edges |= both & (rel > th.depth_rel_jump) & (d < nd)

        # entity boundary, marked on the entity side (KB Decision 9: the id is an
        # address; only *inequality* is ever used, never its magnitude)
        neid = shift(eid, dy, dx)
        edges |= valid & (eid != 0) & (neid != eid)

    return edges


def support_edges(structure: FrameStructure, th: Thresholds) -> np.ndarray:
    """Reference edges ∪ normal creases ∪ albedo/material boundaries."""
    depth = structure.depth
    shape = depth.shape
    fin = np.isfinite(depth) & (depth > 0.0)

    normal = structure.normal.astype(np.float64)
    nlen = np.linalg.norm(normal, axis=-1, keepdims=True)
    unit = normal / np.maximum(nlen, 1e-6)
    albedo = structure.albedo.astype(np.float64)
    cos_thresh = float(np.cos(np.deg2rad(th.normal_angle_deg)))

    extra = np.zeros(shape, dtype=bool)
    for dy, dx in _NBRS:
        valid = valid_shift_mask(shape, dy, dx)
        both = valid & fin & shift(fin, dy, dx)

        dot = (unit * shift(unit, dy, dx)).sum(axis=-1)
        extra |= both & (dot < cos_thresh)

        da = np.abs(albedo - shift(albedo, dy, dx)).max(axis=-1)
        extra |= both & (da > th.albedo_jump)

    return reference_edges(structure, th) | extra


# --------------------------------------------------------------------------- #
# target edge map
# --------------------------------------------------------------------------- #
def target_edges(target_rgb: np.ndarray, th: Thresholds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(thin_magnitude, E_all, E_strong)`` for a uint8 RGB target."""
    gray = luma(target_rgb.astype(np.float64) / 255.0)
    mag, gx, gy = gradient_magnitude(gray)
    thin = nms_thin(mag, gx, gy)
    return thin, thin >= th.edge_threshold, thin >= th.strong_edge_threshold


# --------------------------------------------------------------------------- #
# per-frame metrics
# --------------------------------------------------------------------------- #
def stylisation_distance(target_rgb: np.ndarray, native_rgb: np.ndarray) -> tuple[float, float]:
    """How far the target actually moved from the native frame.

    ``(mean absolute pixel delta in 0-255, colour-histogram L1 distance in 0-2)``.

    Gates 1-3 only ask *did the structure survive*, and the perfect way to
    satisfy all three is to change nothing at all: measured on the seed session,
    the untouched native frame scores F=0.976 with a spurious ratio of 0.000 —
    better than any real target. Without a floor on stylisation, "turn the
    teacher down until it passes" is the optimum, and the milestone's exit
    criterion would be met by a teacher that does nothing.

    The histogram term carries the gate because the scene is dark: a real
    restyle there shifts the palette a long way while moving mean absolute pixel
    value only a few levels out of 255. The pixel term is reported alongside for
    diagnostics.
    """
    d = float(np.abs(target_rgb.astype(np.float64) - native_rgb.astype(np.float64)).mean())
    hs = []
    for img in (target_rgb, native_rgb):
        h = np.concatenate(
            [np.histogram(img[..., c], bins=32, range=(0, 255))[0] for c in range(3)]
        ).astype(np.float64)
        hs.append(h / max(h.sum(), 1.0))
    return d, float(np.abs(hs[0] - hs[1]).sum())


def frame_metrics(
    structure: FrameStructure,
    target_rgb: np.ndarray,
    th: Thresholds,
    native_rgb: np.ndarray | None = None,
) -> dict:
    """All four gate quantities for one frame, plus the counts behind them.

    ``native_rgb`` is optional only so diagnostic callers can skip it; the
    validator always supplies it, and without it gate 4 cannot be evaluated.
    """
    R = reference_edges(structure, th)
    S = support_edges(structure, th)
    _thin, E_all, E_strong = target_edges(target_rgb, th)

    tol = th.tolerance_px
    R_dil = dilate(R, tol)
    S_dil = dilate(S, tol)
    E_dil = dilate(E_all, tol)

    n_R = int(R.sum())
    n_E = int(E_all.sum())
    n_Es = int(E_strong.sum())

    recall = float((R & E_dil).sum()) / n_R if n_R else float("nan")

    # gate 2 — the §12 hallucinated-geometry guard, on the CONFIDENT edges only
    supported_strong = float((E_strong & S_dil).sum()) / n_Es if n_Es else 1.0
    spurious = 1.0 - supported_strong

    # Gate 1's precision term is deliberately the *strong*-edge precision, i.e.
    # 1 - the gate-2 ratio. Measuring precision on the permissive edge set
    # instead scores how much low-amplitude brushwork a painterly teacher added
    # — stylisation, not structural distortion — and on the measured seed data it
    # rejects the real teacher while accepting a 3 px misaligned one. Reported
    # separately as `edge_precision_all` for diagnostics only. See KB Decision 32.
    precision = supported_strong if n_Es else float("nan")
    precision_all = float((E_all & S_dil).sum()) / n_E if n_E else float("nan")

    if not n_R:
        # nothing to preserve (no visible geometry at all) — undefined, not a fail
        f_measure = float("nan")
    elif not n_E or np.isnan(precision) or (recall + precision) <= 0.0:
        # a target that draws nothing has zero recall; this must FAIL, not be
        # excused as "undefined" — a collapsed teacher is the easy way to score
        # perfect precision
        f_measure = 0.0
    else:
        f_measure = 2.0 * precision * recall / (precision + recall)

    # gate 3 — per-entity silhouette support
    eid = structure.entity_id
    per_entity: list[dict] = []
    for raw_id in np.unique(eid):
        i = int(raw_id)
        if i == 0:
            continue
        mask = eid == i
        px = int(mask.sum())
        if px < th.min_entity_px:
            continue
        sil = np.zeros(mask.shape, dtype=bool)
        for dy, dx in _NBRS:
            valid = valid_shift_mask(mask.shape, dy, dx)
            sil |= valid & mask & (shift(eid, dy, dx) != i)
        n_sil = int(sil.sum())
        if n_sil == 0:
            continue
        per_entity.append({
            "instance_id": i,
            "pixel_count": px,
            "silhouette_px": n_sil,
            "support": float((sil & E_dil).sum()) / n_sil,
        })

    # gate 4 — did the teacher actually restyle anything?
    if native_rgb is None:
        styl_pix = styl_hist = float("nan")
    else:
        styl_pix, styl_hist = stylisation_distance(target_rgb, native_rgb)

    supports = [e["support"] for e in per_entity]
    return {
        "frame_id": structure.frame_id,
        "stylisation_pix": styl_pix,
        "stylisation_hist": styl_hist,
        "reference_edge_px": n_R,
        "target_edge_px": n_E,
        "strong_edge_px": n_Es,
        "boundary_recall": recall,
        "boundary_precision": precision,
        "edge_precision_all": precision_all,
        "boundary_f": f_measure,
        "spurious_edge_ratio": spurious,
        "entities_measured": len(per_entity),
        "min_silhouette_support": min(supports) if supports else float("nan"),
        "per_entity": per_entity,
    }


def gate_failures(m: dict, th: Thresholds) -> list[tuple[str, str]]:
    """``[(code, message)]`` for each gate this frame fails. NaN (undefined —
    e.g. a frame with no visible geometry) never fails a gate."""
    out: list[tuple[str, str]] = []
    f = m["boundary_f"]
    if not np.isnan(f) and f < th.boundary_f:
        out.append((
            "align.boundary_f",
            f"frame {m['frame_id']}: boundary F={f:.3f} < {th.boundary_f} "
            f"(recall={m['boundary_recall']:.3f}, precision={m['boundary_precision']:.3f}) — "
            "the target no longer reproduces the engine's scene layout",
        ))
    st = m.get("stylisation_hist", float("nan"))
    if not np.isnan(st) and st < th.min_stylisation:
        out.append((
            "align.not_stylised",
            f"frame {m['frame_id']}: stylisation distance {st:.3f} < {th.min_stylisation} "
            f"(mean |Δpixel|={m.get('stylisation_pix', float('nan')):.2f}) — the target is "
            "barely distinguishable from the native frame. Gates 1-3 are all satisfied by "
            "changing nothing, so a target that does not restyle is not usable supervision",
        ))
    s = m["spurious_edge_ratio"]
    if s > th.spurious_ratio:
        out.append((
            "align.spurious_edges",
            f"frame {m['frame_id']}: spurious-edge ratio {s:.3f} > {th.spurious_ratio} "
            f"({m['strong_edge_px']} strong target edges, that fraction with no structural "
            "support within tolerance) — hallucinated geometry (KB §12)",
        ))
    for e in m["per_entity"]:
        if e["support"] < th.silhouette_support:
            out.append((
                "align.entity_silhouette",
                f"frame {m['frame_id']}: entity {e['instance_id']} "
                f"({e['pixel_count']} px) silhouette support {e['support']:.3f} < "
                f"{th.silhouette_support} — the entity's outline is not present in the target",
            ))
    return out


def distribution(values: list[float]) -> dict:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p01": float(np.percentile(a, 1)),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }
