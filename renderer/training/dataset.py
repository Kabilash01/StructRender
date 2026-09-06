"""Structural session + teacher targets -> training tensors (M4).

Reuses :class:`data_pipeline.teacher.structural.StructuralSession` as the single
reader for the immutable m2.0 structural session; nothing here writes into the
session or the target set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.teacher.adapter import FrameStructure
from data_pipeline.teacher.structural import StructuralSession

IN_CHANNELS = 7
CHANNEL_NAMES = (
    "depth_inv",
    "normal_x",
    "normal_y",
    "normal_z",
    "albedo_r",
    "albedo_g",
    "albedo_b",
)


# --------------------------------------------------------------------------- #
# channel construction
# --------------------------------------------------------------------------- #
def inverse_depth(depth: np.ndarray) -> np.ndarray:
    """``1 / (1 + d)`` — bounded [0,1], monotone in nearness, background -> 0.

    ``depth`` is raw linear eye-space Z in metres with ``+inf`` for background
    (frame_structure.md §2.1). ``1/(1+inf) == 0`` exactly, so the background
    sentinel maps to the bottom of the range with no magic constant and no
    scene-dependent far-plane normalisation.
    """
    return (1.0 / (1.0 + depth.astype(np.float32))).astype(np.float32)


def build_input(structure: FrameStructure) -> np.ndarray:
    """``FrameStructure`` -> ``float32 [7, H, W]``.

    ``structure.entity_id`` and the session's ``motion`` buffer are deliberately
    not read here (KB §4 CNN Stem Explicit Boundaries; Decisions 8 and 9).
    """
    depth = structure.depth
    if depth.ndim == 3:
        depth = depth[..., 0]
    chw = np.concatenate(
        [
            inverse_depth(depth)[None, ...],
            np.transpose(structure.normal.astype(np.float32), (2, 0, 1)),
            np.transpose(structure.albedo.astype(np.float32), (2, 0, 1)),
        ],
        axis=0,
    )
    return np.ascontiguousarray(chw, dtype=np.float32)


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    train_ids: list[int]
    val_ids: list[int]
    val_sequences: list[str]
    guard_frames: int
    dropped_guard_ids: list[int]

    def as_report(self) -> dict:
        return {
            "policy": "contiguous_sequence_holdout",
            "val_sequences": self.val_sequences,
            "guard_frames": self.guard_frames,
            "n_train": len(self.train_ids),
            "n_val": len(self.val_ids),
            "n_dropped_guard": len(self.dropped_guard_ids),
            "val_frame_ranges": _ranges(self.val_ids),
            "train_frame_ranges": _ranges(self.train_ids),
        }


def _ranges(ids: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for i in sorted(ids):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def make_split(
    session: StructuralSession, val_sequences: list[str], guard_frames: int
) -> Split:
    """Hold out whole contiguous sequences, with a temporal guard band.

    A random frame split leaks: at 60 fps with smooth drift, a val frame's
    neighbours are near-duplicates of it. Holding out whole ``sequences[]``
    spans makes train/val temporally disjoint; ``guard_frames`` additionally
    drops train frames adjacent to a held-out span so they are not *contiguous*
    either.
    """
    by_id = {s["sequence_id"]: s for s in session.header["sequences"]}
    missing = [s for s in val_sequences if s not in by_id]
    if missing:
        raise KeyError(f"sequences not in session {session.session_id}: {missing}")

    all_ids = set(session.frame_ids())
    val: set[int] = set()
    guard: set[int] = set()
    for name in val_sequences:
        lo, hi = by_id[name]["frame_range"]
        val |= {i for i in range(lo, hi + 1) if i in all_ids}
        guard |= {i for i in range(lo - guard_frames, hi + guard_frames + 1) if i in all_ids}
    guard -= val
    train = sorted(all_ids - val - guard)
    return Split(
        train_ids=train,
        val_ids=sorted(val),
        val_sequences=list(val_sequences),
        guard_frames=guard_frames,
        dropped_guard_ids=sorted(guard),
    )


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def discover_session(dataset_root: str | os.PathLike | None = None) -> Path:
    """Newest ``ds_*`` under the recorder root. Raises if there is none."""
    root = Path(
        dataset_root
        or os.environ.get("STRUCTRENDER_DATASET_ROOT")
        or Path(__file__).resolve().parents[2] / "data_pipeline" / "recorder"
    )
    candidates = sorted(p for p in root.glob("ds_*") if (p / "manifest.json").is_file())
    if not candidates:
        raise FileNotFoundError(f"no recorded session under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def target_dir_for(session_id: str, style_id: str, target_root: str | os.PathLike | None = None) -> Path:
    root = Path(
        target_root
        or os.environ.get("STRUCTRENDER_TARGET_ROOT")
        or Path(__file__).resolve().parents[2] / "data_pipeline" / "teacher_targets"
    )
    return root / style_id / session_id


class StyledFrameDataset(Dataset):
    """One item = (structural input ``[7,H,W]``, teacher target ``[3,H,W]``, frame_id)."""

    def __init__(self, session: StructuralSession, target_dir: str | os.PathLike, frame_ids: list[int]):
        self.session = session
        self.target_dir = Path(target_dir)
        self.frame_ids = list(frame_ids)
        lines = [
            json.loads(l)
            for l in (self.target_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        self._target_rel = {int(r["frame_id"]): r["target_rgb_path"] for r in lines}
        missing = [i for i in self.frame_ids if i not in self._target_rel]
        if missing:
            raise KeyError(f"{len(missing)} frames have no teacher target (first: {missing[:5]})")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def load_target(self, frame_id: int) -> np.ndarray:
        from PIL import Image

        with Image.open(self.target_dir / self._target_rel[frame_id]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def __getitem__(self, index: int):
        frame_id = self.frame_ids[index]
        x = build_input(self.session.load_structure(frame_id))
        y = np.transpose(self.load_target(frame_id).astype(np.float32) / 255.0, (2, 0, 1))
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(y)), frame_id
