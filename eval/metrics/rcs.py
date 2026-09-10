"""Reappearance Consistency Score — KB §11.

    RCS_i(dt) = cos( E(I_i^t), E(I_i^{t+dt}) )

for entity ``i`` last seen at ``t`` and seen again after ``dt`` outside view.
This is the project's **headline** metric (KB line 2308: "the headline result is
RCS versus time-outside-view ... the image-quality ladder is the secondary
figure"), and until now it had never been computed.

Three choices this module makes, none of which §11 fixes, all recorded in KB
Decision 53:

1. ``E`` is a **masked deep-feature descriptor**: AlexNet conv features (the same
   trunk LPIPS uses, already a project dependency), average-pooled over the
   entity's own pixels only, each layer unit-normalised, then concatenated. It
   describes how the entity *looks*, not where it is, and ignores every pixel
   that is not the entity.
2. Only the entity's pixels contribute. A crop would drag in background that
   changes across the gap for reasons that have nothing to do with identity.
3. There is **no silent fallback**. If the feature trunk is unavailable the
   caller gets an error, never a home-made substitute reported under the same
   name (the rule Decision 35 set for LPIPS).

A score is only meaningful next to its references, so ``evaluate_source`` is run
over the native engine frames and the teacher targets too:

* **native** — the engine renders deterministically, so this is the ceiling that
  pose change alone allows. It is not 1.0: the object is seen from a different
  angle after the gap.
* **target** — the teacher's own consistency, and the real bar. A model trained
  on these targets cannot be more consistent than its supervision without
  memory, which is precisely what EPAM is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


class AppearanceEncoder:
    """``E`` from §11's formula. AlexNet conv features, masked-average-pooled."""

    def __init__(self, device: torch.device | str = "cpu"):
        try:
            from torchvision.models import AlexNet_Weights, alexnet
        except ImportError as e:  # ImportError ONLY — a broader catch would hide a real fault
            raise RuntimeError(
                "RCS needs torchvision's AlexNet trunk (the LPIPS backbone). There is no "
                "fallback descriptor on purpose: a home-made number published as RCS is "
                "worse than no number (KB Decision 35)."
            ) from e
        self.device = torch.device(device)
        net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features.eval().to(self.device)
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        # after each ReLU that follows a conv — LPIPS's tap points
        self.taps = (1, 4, 7, 9, 11)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """``rgb`` (H,W,3) in [0,1]; ``mask`` (H,W) bool. Returns a unit vector."""
        if mask.sum() == 0:
            raise ValueError("empty entity mask — cannot describe an entity with no pixels")
        # Zero the background BEFORE the trunk, not just when pooling. Conv
        # features have receptive fields, so masked pooling alone still lets
        # background near the silhouette reach the descriptor: measured, replacing
        # the entire background moved the score to 0.979 while inverting the
        # entity itself moved it to 0.971 — comparable magnitudes, i.e. the
        # descriptor was reading the scene nearly as much as the entity. With the
        # input pre-masked the same background swap scores exactly 1.0.
        # KB Decision 53.
        rgb = np.where(mask[..., None], rgb, 0.0).astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1)[None]
        x = ((x.to(self.device) - self.mean) / self.std)
        m = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None].to(self.device)

        parts, h = [], x
        for i, layer in enumerate(self.net):
            h = layer(h)
            if i in self.taps:
                mm = F.interpolate(m, size=h.shape[-2:], mode="area")
                denom = mm.sum().clamp_min(1e-6)
                pooled = (h * mm).sum(dim=(2, 3)) / denom
                parts.append(F.normalize(pooled, dim=1))
        return F.normalize(torch.cat(parts, dim=1), dim=1)[0]


def rcs(before: torch.Tensor, after: torch.Tensor) -> float:
    """cos(E_before, E_after) — both already unit vectors."""
    return float(torch.dot(before, after).clamp(-1.0, 1.0))


@dataclass
class SequenceRCS:
    sequence_id: str
    mechanism: str
    unseen_seconds: float
    instance_id: int
    last_seen_frame: int
    first_seen_again_frame: int
    px_before: int
    px_after: int
    score: float
    n_pairs: int = 1
    pair_sd: float | None = None
    frames_before: list | None = None
    frames_after: list | None = None


def entity_mask_at(session, frame_id: int, instance_id: int) -> np.ndarray:
    """The entity's own pixels. `entity_id` is an ADDRESS here (Decision 9) —
    used to select pixels, never fed to a network as a value."""
    return session.load_structure(frame_id).entity_id == instance_id


def reappearance_sequences(session) -> list[dict]:
    return [s for s in session.header["sequences"] if s["sequence_kind"] == "reappearance"]


def _visible_run(session, iid: int, anchor: int, direction: int, want: int,
                 min_entity_px: int) -> list[int]:
    """Up to ``want`` frames walking outward from ``anchor``, keeping only frames
    where the entity is actually visible with enough pixels.

    ``direction`` is -1 to walk backwards from the last-seen frame, +1 to walk
    forwards from the first-seen-again frame.
    """
    frames, fid = [], anchor
    while len(frames) < want and 0 <= fid < session.frame_count:
        if (session.load_structure(fid).entity_id == iid).sum() >= min_entity_px:
            frames.append(fid)
        else:
            break  # contiguity matters: stop at the gap, don't jump over it
        fid += direction
    return frames


def evaluate_source(session, image_for_frame, encoder: AppearanceEncoder,
                    min_entity_px: int = 200, window: int = 3) -> list[SequenceRCS]:
    """RCS for every reappearance sequence, for one source of images.

    ``image_for_frame(frame_id) -> (H,W,3) float in [0,1]`` is the only thing that
    varies between native frames, teacher targets and a model's predictions, so
    every source is scored through an identical path.

    The mask always comes from the ENGINE's entity map, never from the rendered
    image: the question is whether the same engine entity looks the same, and a
    mask derived from the render would move with the render's own errors.
    """
    out: list[SequenceRCS] = []
    for s in reappearance_sequences(session):
        iid = int(s["target_instance_id"])
        t0, t1 = int(s["last_seen_frame"]), int(s["first_seen_again_frame"])
        # A single frame pair makes one unlucky pose move a whole data point, and
        # with only ten sequences that is most of the signal. Average over all
        # before x after pairs in a short contiguous window each side, which also
        # yields a within-sequence spread (KB Decision 54).
        before = _visible_run(session, iid, t0, -1, window, min_entity_px)
        after = _visible_run(session, iid, t1, +1, window, min_entity_px)
        if not before or not after:
            continue  # too few pixels to describe; recorded by omission, not imputed
        embs_b = [encoder(image_for_frame(f), entity_mask_at(session, f, iid)) for f in before]
        embs_a = [encoder(image_for_frame(f), entity_mask_at(session, f, iid)) for f in after]
        pairs = [rcs(b, a) for b in embs_b for a in embs_a]
        arr = np.asarray(pairs, dtype=np.float64)
        out.append(SequenceRCS(
            sequence_id=s["sequence_id"], mechanism=s["mechanism"],
            unseen_seconds=float(s["unseen_seconds"]), instance_id=iid,
            last_seen_frame=t0, first_seen_again_frame=t1,
            px_before=int(entity_mask_at(session, before[0], iid).sum()),
            px_after=int(entity_mask_at(session, after[0], iid).sum()),
            score=float(arr.mean()), n_pairs=int(arr.size),
            pair_sd=float(arr.std(ddof=1)) if arr.size > 1 else None,
            frames_before=before, frames_after=after))
    return out


def summarise(rows: list[SequenceRCS]) -> dict:
    """Mean/sd overall, per unseen duration and per mechanism — §11's reporting."""
    def stats(vs):
        a = np.asarray(vs, dtype=np.float64)
        return {"n": int(a.size), "mean": float(a.mean()) if a.size else float("nan"),
                "sd": float(a.std(ddof=1)) if a.size > 1 else None}

    by_dt: dict[str, list] = {}
    by_mech: dict[str, list] = {}
    for r in rows:
        by_dt.setdefault(f"{r.unseen_seconds:g}s", []).append(r.score)
        by_mech.setdefault(r.mechanism, []).append(r.score)
    return {
        "overall": stats([r.score for r in rows]),
        "by_unseen_seconds": {k: stats(v) for k, v in sorted(by_dt.items(), key=lambda kv: float(kv[0][:-1]))},
        "by_mechanism": {k: stats(v) for k, v in sorted(by_mech.items())},
        "per_sequence": [r.__dict__ for r in rows],
    }
