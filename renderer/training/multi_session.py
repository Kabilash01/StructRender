"""Multi-session datasets and the session-level split — KB Decision 64.

Everything before this trained on ONE recorded session, so `make_split` held out
contiguous *sequences* within it and needed a 60-frame guard band to keep train
and val non-adjacent (KB Decision 35). With 15 independent recordings that is no
longer the right instrument:

* whole **sessions** can be held out, so train/val/test share no frames, no
  camera path and no entity trajectory — not merely no indices;
* the guard band becomes unnecessary, because sessions are separate recordings
  rather than adjacent frames of one continuous drift;
* and for the first time a **test** split exists that nothing trains on and
  nothing selects on. Every number in this project before now was a
  best-of-N-epochs minimum on the frames used to pick the checkpoint.

The split itself is frozen in `configs/data/session_split_v1.json`, chosen by
position in the sorted session-id list *before any metric was computed*, so the
held-out sessions cannot have been picked to flatter a result.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import ConcatDataset, Dataset

from data_pipeline.teacher.structural import StructuralSession
from renderer.training.dataset import StyledFrameDataset, target_dir_for
from renderer.training.train_baseline import ROOT


def load_session_split(path: str | Path = "configs/data/session_split_v1.json") -> dict:
    return json.loads((Path(ROOT) / path).read_text(encoding="utf-8"))


def _one(session_id: str, style_id: str, dataset_root: str | None, target_root: str | None):
    root = Path(dataset_root or Path(ROOT) / "data_pipeline" / "recorder")
    session = StructuralSession(root / session_id)
    if not session.validated:
        raise RuntimeError(f"{session_id} has no VALIDATED marker")
    tdir = target_dir_for(session.session_id, style_id, target_root)
    if not (Path(tdir) / "ALIGNED").is_file():
        raise RuntimeError(f"{session_id} targets have no ALIGNED marker")
    return StyledFrameDataset(session, tdir, session.frame_ids())


def build_splits(style_id: str, split: dict | None = None,
                 dataset_root: str | None = None, target_root: str | None = None):
    """``(train, val, test)`` as concatenated per-session datasets.

    Every session is checked for BOTH markers: a session that never passed M2
    validation or M3 alignment cannot enter any split, rather than being silently
    trained on.
    """
    split = split or load_session_split()
    out = []
    for key in ("train_sessions", "val_sessions", "test_sessions"):
        parts = [_one(s, style_id, dataset_root, target_root) for s in split[key]]
        out.append(ConcatDataset(parts))
    train, val, test = out

    overlap = (set(split["train_sessions"]) & set(split["val_sessions"])
               | set(split["train_sessions"]) & set(split["test_sessions"])
               | set(split["val_sessions"]) & set(split["test_sessions"]))
    if overlap:
        raise RuntimeError(f"splits overlap on {sorted(overlap)} — the held-out set is not held out")
    return train, val, test


def split_report(split: dict | None = None) -> dict:
    split = split or load_session_split()
    return {
        "policy": split["policy"],
        "train_sessions": split["train_sessions"],
        "val_sessions": split["val_sessions"],
        "test_sessions": split["test_sessions"],
        "n_train_frames": split["n_train_frames"],
        "n_val_frames": split["n_val_frames"],
        "n_test_frames": split["n_test_frames"],
        "test_note": ("The test split is NEVER used for training or checkpoint selection. It is "
                      "scored once, at the end, and is the project's first independent "
                      "generalisation estimate (KB Decision 64)."),
    }


def build_temporal_splits(style_id: str, split: dict | None = None,
                          dataset_root: str | None = None, target_root: str | None = None,
                          tick_delta: int = 1):
    """``(train_triples, val_frames)`` for Model D across many sessions.

    Triples are built **per session** and only then concatenated: a triple must
    never span a session boundary, because frame ``t-1`` of one recording has no
    motion relationship to frame ``t`` of another. `consecutive_triples` already
    requires all three ends inside the given frame set, so passing one session's
    ids at a time is what enforces it (KB Decision 64).
    """
    from pathlib import Path as _P

    from renderer.temporal.pair_dataset import TemporalFrameDataset, TemporalTripleDataset

    split = split or load_session_split()
    root = _P(dataset_root or _P(ROOT) / "data_pipeline" / "recorder")

    def _sessions(key, cls):
        parts = []
        for sid in split[key]:
            session = StructuralSession(root / sid)
            if not session.validated:
                raise RuntimeError(f"{sid} has no VALIDATED marker")
            tdir = target_dir_for(session.session_id, style_id, target_root)
            if not (_P(tdir) / "ALIGNED").is_file():
                raise RuntimeError(f"{sid} targets have no ALIGNED marker")
            kw = {"tick_delta": tick_delta} if cls is TemporalTripleDataset else {}
            parts.append(cls(session, tdir, session.frame_ids(), **kw))
        return ConcatDataset(parts)

    return _sessions("train_sessions", TemporalTripleDataset), \
        _sessions("val_sessions", TemporalFrameDataset)


class _StyleTagged(Dataset):
    """A per-(session, style) dataset whose items also carry the style's index."""

    def __init__(self, inner, style_idx: int):
        self.inner, self.style_idx = inner, int(style_idx)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> dict:
        return {**self.inner[i], "style_idx": self.style_idx}


def excluded_frames(exclusions: dict, session_id: str) -> set[int]:
    """Frames of one session failing a gate in EITHER style (KB Decision 94).

    The union across styles, so both styles train and are scored on exactly the
    same frames and the style is the only thing that differs between them.
    """
    return {int(f) for f in exclusions["excluded"].get(session_id, {})}


def build_two_style_temporal_splits(styles: list[str], split: dict, exclusions: dict,
                                    dataset_root: str | None = None,
                                    target_root: str | None = None, tick_delta: int = 1):
    """``(train_triples, val_frames)`` over every (session, style), for two-style B.

    Per-frame exclusion (KB Decision 94) replaces the whole-session drop: a
    session enters a style only if the exclusion builder ADMITTED it — ALIGNED,
    or its sole validator errors were per-frame gate failures — and its excluded
    frames are removed before triples and pairs are formed, so no triple and no
    scored pair touches a frame that failed a gate. Gates are unchanged.
    """
    from pathlib import Path as _P

    from renderer.temporal.pair_dataset import TemporalFrameDataset, TemporalTripleDataset

    if list(exclusions["styles"]) != list(styles):
        raise RuntimeError(f"exclusion list covers {exclusions['styles']}, config trains {styles}")
    root = _P(dataset_root or _P(ROOT) / "data_pipeline" / "recorder")

    def _sessions(key, cls):
        parts = []
        for sid in split[key]:
            session = StructuralSession(root / sid)
            if not session.validated:
                raise RuntimeError(f"{sid} has no VALIDATED marker")
            keep = sorted(set(session.frame_ids()) - excluded_frames(exclusions, sid))
            for idx, style in enumerate(styles):
                if sid not in exclusions["admitted"][style]:
                    raise RuntimeError(f"{sid} was not admitted for {style}: "
                                       f"{exclusions['refused'][style].get(sid, 'not scanned')}")
                tdir = target_dir_for(session.session_id, style, target_root)
                kw = {"tick_delta": tick_delta} if cls is TemporalTripleDataset else {}
                parts.append(_StyleTagged(cls(session, tdir, keep, **kw), idx))
        return ConcatDataset(parts)

    return _sessions("train_sessions", TemporalTripleDataset), \
        _sessions("val_sessions", TemporalFrameDataset)
