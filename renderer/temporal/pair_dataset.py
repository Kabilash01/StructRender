"""Consecutive-frame samples for Model D — KB §10 Milestone 6.

Model C's dataset serves one frame at a time because Model C is frame-independent.
Model D needs history, so a sample here is a **run of three consecutive frames**
``(t-2, t-1, t)`` drawn entirely from one split.

**Why three and not two.** The temporal loss compares ``\\hat I_t`` against
``W(\\hat I_{t-1}, V_t)``, so the step needs the model's prediction for frame
``t-1``. That prediction must be produced the *same way the M6 gate will produce
it* — with frame ``t-1``'s own history — or the loss would train the model to
make its history-informed output match its history-free output, i.e. to ignore
history. Rendering ``t-1`` with history needs ``t-2``. The marginal cost is one
extra encoder pass; the decoder and RGB head run twice either way.

Every triple is built by **chaining two `consecutive_pairs` results**, not by
re-deriving adjacency here, so the guarantees that function already carries are
inherited rather than restated: both ends inside the given frame set, ``motion_valid``
on the newer frame, and a computed (never assumed) tick delta. Chaining two pairs
that share a frame therefore puts all three frames inside the split, so a triple
can neither straddle the 60-frame guard band nor reach out of the training split.

Each item carries, at **image** resolution:

* three structural inputs — the same 7 channels as Models A and C
  (``build_input``: no entity IDs, no motion; Decisions 8 and 9);
* the motion field and the §4 confidence mask for **both** steps, produced by
  ``eval.metrics.temporal.engine_confidence`` — the *same function* the M6 gate
  uses, so the loss and the gate cannot drift apart on which pixels are
  trustworthy;
* the teacher target for frame ``t`` only. Reconstruction is supervised once per
  frame per epoch, exactly as it is for Model C; supervising both ends would
  double each frame's exposure relative to the other rungs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from eval.metrics.temporal import engine_confidence
from renderer.temporal.pairs import FramePair, consecutive_pairs
from renderer.training.dataset import StyledFrameDataset, build_input


def _confidence(session, prev_id: int, curr_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(motion, M_t)`` for one step, from `engine_confidence` — the GATE's mask.

    The tick delta is computed from the records rather than assumed: it is unused
    by `engine_confidence` today, but a `FramePair` that misstates it is a lie
    waiting to be read by the next caller (KB Decision 29: recompute derived
    fields, never trust the asserted value).
    """
    dt = int(session.record(curr_id)["tick"]) - int(session.record(prev_id)["tick"])
    c = engine_confidence(session, FramePair(prev_id, curr_id, dt))
    return c["motion"][0], c["mask"][0]


@dataclass(frozen=True)
class FrameTriple:
    """``(t-2, t-1, t)``, all inside one split, both steps one tick apart."""

    prev2_id: int
    prev_id: int
    curr_id: int


def consecutive_triples(session, frame_ids, tick_delta: int = 1) -> list[FrameTriple]:
    """Chain :func:`renderer.temporal.pairs.consecutive_pairs` into runs of three.

    ``tick_delta`` filters on *engine ticks*, not frame ids: this session drops to
    a reduced rate during the long reappearance gaps, so consecutive frame ids can
    be 8 ticks apart and carry ~8x the displacement. Training is 100 % ``dt == 1``
    and the M6 gate is a ``dt == 1`` set, so mixing an 8-tick step into a triple
    would train on a displacement regime the gate never measures.
    """
    pairs = [p for p in consecutive_pairs(session, frame_ids) if p.tick_delta == tick_delta]
    by_curr = {p.curr_id: p for p in pairs}
    return [
        FrameTriple(prev2_id=by_curr[p.prev_id].prev_id, prev_id=p.prev_id, curr_id=p.curr_id)
        for p in sorted(pairs, key=lambda q: q.curr_id)
        if p.prev_id in by_curr
    ]


class TemporalTripleDataset(Dataset):
    """One item = three consecutive frames, their motion, their masks, one target."""

    def __init__(self, session, target_dir, frame_ids: list[int], tick_delta: int = 1):
        self.session = session
        self.triples = consecutive_triples(session, frame_ids, tick_delta)
        if not self.triples:
            raise ValueError(
                f"no dt=={tick_delta} triples inside a {len(frame_ids)}-frame set; "
                "a temporal rung cannot be trained on it"
            )
        # Targets are only ever read for `curr_id`, but the frame set is declared
        # in full so a missing target fails at construction, not at epoch 3.
        self._targets = StyledFrameDataset(session, target_dir, sorted(frame_ids))

    def __len__(self) -> int:
        return len(self.triples)

    @property
    def frame_ids(self) -> list[int]:
        """The frames this dataset actually *scores* — one per triple."""
        return [t.curr_id for t in self.triples]

    def _structure(self, frame_id: int) -> torch.Tensor:
        return torch.from_numpy(build_input(self.session.load_structure(frame_id)))

    def _step(self, prev_id: int, curr_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _confidence(self.session, prev_id, curr_id)

    def __getitem__(self, index: int) -> dict:
        t = self.triples[index]
        motion_prev, mask_prev = self._step(t.prev2_id, t.prev_id)
        motion_curr, mask_curr = self._step(t.prev_id, t.curr_id)
        y = np.transpose(
            self._targets.load_target(t.curr_id).astype(np.float32) / 255.0, (2, 0, 1)
        )
        return {
            "structure_prev2": self._structure(t.prev2_id),
            "structure_prev": self._structure(t.prev_id),
            "structure_curr": self._structure(t.curr_id),
            "motion_prev": motion_prev,
            "mask_prev": mask_prev,
            "motion_curr": motion_curr,
            "mask_curr": mask_curr,
            "target_curr": torch.from_numpy(np.ascontiguousarray(y)),
            "frame_id": t.curr_id,
        }


class TemporalFrameDataset(Dataset):
    """One item per frame, with the immediately previous frame as its history.

    This is the **evaluation / selection** dataset. It exists separately from
    :class:`TemporalTripleDataset` because scoring frame ``f`` needs only ``f-1``
    (a pair), while *training* additionally needs ``f-2`` to render the temporal
    loss's reference frame the way the gate will render it.

    The distinction that matters for the §11 ladder: this dataset has **exactly
    one item per frame id given to it**, so Model D's selection set is the same
    166 validation frames Models A and C were selected on, under the same rule
    (lowest ``val_l1``). A triple-based validation set would have silently
    selected Model D on a subset.

    **The validation split mixes tick deltas and this dataset does not hide it.**
    120 of its 164 consecutive pairs are 8 engine ticks apart (the recorder drops
    to a reduced rate during the long reappearance gaps), carrying roughly 8x the
    displacement of anything in training, which is 100 % ``dt == 1``. Those frames
    are still served with their real predecessor, because that is what the model
    would face; the consequence is that Model D's selection metric includes an
    out-of-distribution displacement regime that Model C's did not have to handle.
    KB Decision 55 records the same split property for the temporal metric.

    A frame with no usable predecessor — the first frame of the set, or one whose
    ``motion_valid`` is false — is served with a **zero mask and zero motion**,
    and with its own structure standing in for the previous frame. That is not a
    special case in the model: the fusion is gated by ``M_t``, so a zero mask
    makes the temporal contribution exactly zero and the model falls back to
    Model C's frame-independent path by construction rather than by a branch.
    """

    def __init__(self, session, target_dir, frame_ids: list[int]):
        self.session = session
        self.frame_ids = sorted(int(f) for f in frame_ids)
        self._targets = StyledFrameDataset(session, target_dir, self.frame_ids)
        have = set(self.frame_ids)
        self._prev = {
            f: (f - 1 if (f - 1) in have and session.motion_valid(f) else None)
            for f in self.frame_ids
        }

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> dict:
        fid = self.frame_ids[index]
        prev = self._prev[fid]
        structure = torch.from_numpy(build_input(self.session.load_structure(fid)))
        if prev is None:
            structure_prev = structure
            motion = torch.zeros(2, *structure.shape[-2:])
            mask = torch.zeros(1, *structure.shape[-2:])
        else:
            structure_prev = torch.from_numpy(
                build_input(self.session.load_structure(prev))
            )
            motion, mask = _confidence(self.session, prev, fid)
        y = np.transpose(self._targets.load_target(fid).astype(np.float32) / 255.0, (2, 0, 1))
        return {
            "structure_prev": structure_prev,
            "structure_curr": structure,
            "motion_curr": motion,
            "mask_curr": mask,
            "target_curr": torch.from_numpy(np.ascontiguousarray(y)),
            "frame_id": fid,
            "has_history": int(prev is not None),
        }
