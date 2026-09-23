"""Short clips for EPAM training: history frames feed the memory, the last frame
is rendered and scored. Every slot is keyed by the engine's EntityKey, never by
slot index (Decision 76)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_pipeline.teacher.structural import StructuralSession
from renderer.entity.keys import keys_for_frame
from renderer.entity.selection import build_entity_masks, pixel_counts, select_entities
from renderer.epam.model import key_code
from renderer.training.dataset import build_input, target_dir_for


class EPAMClipDataset(Dataset):
    def __init__(self, root: Path, sessions: list[str], styles: list[str], *,
                 clip_len: int = 4, strides=(1, 2, 4), every: int = 1,
                 train: bool = True, k_max: int = 8, seed: int = 0) -> None:
        self.styles, self.T, self.strides = list(styles), clip_len, tuple(strides)
        self.train, self.k_max, self.seed = train, k_max, seed
        self.sess, self.items = [], []
        for si, sid in enumerate(sessions):
            s = StructuralSession(root / "data_pipeline" / "recorder" / sid)
            ents = {}
            for line in (Path(s.dir) / "manifest.jsonl").read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    ents[int(r["frame_id"])] = r.get("entities", [])
            tgt = {}
            for st in self.styles:
                d = target_dir_for(s.session_id, st)
                if not (d / "ALIGNED").is_file():
                    raise FileNotFoundError(f"{sid}/{st}: no ALIGNED marker — refusing unvalidated targets")
                tgt[st] = (d, {int(json.loads(l)["frame_id"]): json.loads(l)["target_rgb_path"]
                               for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()})
            self.sess.append((s, ents, tgt))
            ids = sorted(s.frame_ids())
            have = set(ids)
            span = (self.T - 1) * max(self.strides)
            for f in ids[span::every]:
                if all(f - j * st in have for st in self.strides for j in range(self.T)):
                    self.items.append((si, f))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        si, end = self.items[i]
        s, ents, tgt = self.sess[si]
        if self.train:
            rng = random.Random()
            stride, style = rng.choice(self.strides), rng.randrange(len(self.styles))
        else:
            # deterministic, and both styles scored equally often
            stride, style = self.strides[0], i % len(self.styles)
        frames = [end - j * stride for j in reversed(range(self.T))]
        xs, ms, ks, pxs = [], [], [], []
        for f in frames:
            stc = s.load_structure(f)
            keys = keys_for_frame(ents.get(f, []))
            counts = pixel_counts(stc.entity_id)
            sel = select_entities(counts, keys, self.k_max)
            xs.append(build_input(stc))
            ms.append(build_entity_masks(stc.entity_id, sel, self.k_max))
            kc = np.zeros(self.k_max, dtype=np.int64); pc = np.zeros(self.k_max, dtype=np.int64)
            for slot, iid in enumerate(sel):
                kc[slot] = key_code(keys[iid].instance_id, keys[iid].generation)
                pc[slot] = counts[iid]
            ks.append(kc); pxs.append(pc)
        d, rel = tgt[self.styles[style]]
        y = np.asarray(Image.open(d / rel[end]).convert("RGB"), dtype=np.float32) / 255.0
        return {"structure": torch.from_numpy(np.stack(xs)),
                "masks": torch.from_numpy(np.stack(ms)),
                "keys": torch.from_numpy(np.stack(ks)),
                "px": torch.from_numpy(np.stack(pxs)),
                "target": torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))),
                "style": torch.tensor(style), "frame_id": end, "stride": stride}
