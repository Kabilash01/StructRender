"""Frames plus their engine entity masks — Model E's training input (M7/M9).

`renderer.training.dataset.build_input` deliberately omits `structure.entity_id`
(Decision 9: IDs are addresses, never numeric channels), so Model E's masks have
to arrive as their own tensor rather than as an eighth input channel. That is the
whole reason this dataset exists.

Masks are built by the SAME `select_entities` the runtime uses, so what the model
trains on is what it will be served: descending pixel_count, ties broken by
(instance_id, generation), everything past K_max dropped deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from renderer.entity.keys import keys_for_frame
from renderer.entity.selection import build_entity_masks, pixel_counts, select_entities
from renderer.training.dataset import StyledFrameDataset


class EntityFrameDataset(Dataset):
    """Wraps `StyledFrameDataset`, adding `[K_max,H,W]` engine entity masks."""

    def __init__(self, base: StyledFrameDataset, k_max: int = 8, min_px: int = 64) -> None:
        self.base = base
        self.session = base.session
        self.k_max = k_max
        self.min_px = min_px
        self._ents: dict[int, list] = {}
        root = Path(self.session.dir)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                self._ents[int(r["frame_id"])] = r.get("entities", [])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y, fid = self.base[i]
        fid = int(fid)
        idmap = self.session.load_structure(fid).entity_id
        keys = keys_for_frame(self._ents.get(fid, []))
        counts = pixel_counts(idmap, min_px=self.min_px)
        sel = select_entities(counts, keys, self.k_max)
        masks = build_entity_masks(idmap, sel, self.k_max)
        return {"structure": x, "target": y, "entity_masks": torch.from_numpy(masks),
                "frame_id": fid}


def build_entity_splits(style_id: str, split: dict, k_max: int = 8):
    """Session-level train/val/test, each wrapped with its engine entity masks.

    Delegates the split itself to `multi_session.build_splits` rather than
    re-deriving it, so Model E trains on exactly the sessions Models A and C did
    — Decision 35: the ladder must measure the architecture, not the split.
    """
    from torch.utils.data import ConcatDataset

    from renderer.training.multi_session import build_splits

    def wrap(ds):
        parts = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
        return ConcatDataset([EntityFrameDataset(p, k_max) for p in parts])

    tr, va, te = build_splits(style_id, split)
    return wrap(tr), wrap(va), wrap(te)
