"""Model N — the naive raw-ID CONTROL (KB §11 ladder row N; Decision 9's "Allowed Control").

This module deliberately does what every production model is forbidden to do: it
feeds the engine's entity ID to the network as a NUMBER — an eighth input channel,
`entity_id / ID_SCALE`. It exists only to answer one question: is exposing IDs
equivalent to using them structurally (Model E's masked conditioning)?

It lives outside `renderer/training/dataset.py` on purpose, so `build_input` — the
production input builder — keeps its guarantee that IDs never enter as values
(asserted by `test_entity_id_is_not_an_input_channel`). Nothing here may be
imported by a production model.

ID_SCALE is a fixed constant, not a per-frame max: a naive control normalises the
way a naive implementation would, and a per-frame max would silently turn the
channel into a rank. In this dataset only IDs 0 (background), 1 and 2 occur, so the
channel is close to a two-entity label — which makes N a STRONG version of the
naive control here, and the comparison is read with that stated.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from renderer.training.dataset import StyledFrameDataset

ID_SCALE = 8.0


class RawIDFrameDataset(Dataset):
    """StyledFrameDataset with `entity_id / ID_SCALE` appended as channel 8."""

    def __init__(self, base: StyledFrameDataset) -> None:
        self.base = base
        self.session = base.session
        self.frame_ids = base.frame_ids

    def __len__(self) -> int:
        return len(self.base)

    def load_target(self, fid):
        return self.base.load_target(fid)

    def __getitem__(self, i: int):
        x, y, fid = self.base[i]
        ids = self.session.load_structure(int(fid)).entity_id.astype(np.float32) / ID_SCALE
        return torch.cat([x, torch.from_numpy(ids)[None]], dim=0), y, fid


def build_raw_id_splits(style_id: str, split: dict):
    from renderer.training.multi_session import build_splits

    def wrap(ds):
        parts = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
        return ConcatDataset([RawIDFrameDataset(p) for p in parts])

    tr, va, te = build_splits(style_id, split)
    return wrap(tr), wrap(va), wrap(te)
