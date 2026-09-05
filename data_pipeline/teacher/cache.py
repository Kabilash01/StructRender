"""Target cache — KB §3C.5 "Teacher outputs are cached because regeneration is
expensive".

The cache key is exactly the tuple the M3 brief pins:

    (session_id, frame_id, style_id, teacher_version, seed)

``teacher_version`` folds in the style config's content digest (see
``adapter.resolve_teacher_version``), so editing a prompt or a generation
parameter changes the key and the stale target is regenerated rather than
silently reused.

The cache is the sidecar target directory itself: a hit is "the manifest line
for this frame records this key *and* its PNG is on disk". There is no eviction
policy and no LRU — a target is only ever replaced by an explicit ``--force``
regeneration (KB Decision 13's spirit; the M2 validator carries the same rule).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def cache_key(session_id: str, frame_id: int, style_id: str, teacher_version: str, seed: int) -> str:
    payload = {
        "session_id": session_id,
        "frame_id": int(frame_id),
        "style_id": style_id,
        "teacher_version": teacher_version,
        "seed": int(seed),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def content_key(native_rgb, depth, teacher_version: str, seed: int) -> str:
    """Identity of the teacher's *inputs*, independent of which frame they are.

    A recorded session can contain the same rendered moment many times over — a
    reappearance ladder that holds a pose during each out-of-view span records
    byte-identical frames by the hundred. Keying only on ``frame_id`` makes the
    teacher re-render each of them from scratch, which is where the bulk of a
    full run's wall-clock goes.

    This is a pure speed optimisation and NOT an approximation: the backend
    seeds one generator per *style* rather than per frame
    (``sd15_controlnet.py``), so identical conditioning inputs under the same
    ``teacher_version`` and ``seed`` are guaranteed to produce identical output.
    Reusing the earlier render is exactly what regenerating would have written.
    If a backend ever seeds per frame, this equivalence breaks and the reuse
    path must be disabled with it.

    Hashes the native RGB and the depth buffer — the two signals the teacher
    actually conditions on (Decision 31: control is the exported depth buffer).
    """
    h = hashlib.sha256()
    for arr in (native_rgb, depth):
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    h.update(teacher_version.encode("utf-8"))
    h.update(str(int(seed)).encode("utf-8"))
    return h.hexdigest()[:24]


def target_rel_path(frame_id: int) -> str:
    return f"frames/frame_{frame_id:05d}_target.png"


def target_dir(out_root: str | Path, style_id: str, session_id: str) -> Path:
    """``<out_root>/<style_id>/<session_id>/`` — style-keyed from day one
    (KB Decision 31)."""
    return Path(out_root) / style_id / session_id


def read_index(target_dir_path: Path) -> dict[int, dict]:
    """Existing manifest lines by ``frame_id``.

    Tolerates a torn final line: an interrupted run can leave a partially
    written record, and dropping it is the resume path (that frame is simply
    regenerated). Nothing else is repaired.
    """
    p = target_dir_path / "manifest.jsonl"
    if not p.is_file():
        return {}
    index: dict[int, dict] = {}
    lines = p.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn tail from an interrupted run
            raise
        if isinstance(rec, dict) and isinstance(rec.get("frame_id"), int):
            index[rec["frame_id"]] = rec
    return index


def is_hit(target_dir_path: Path, index: dict[int, dict], frame_id: int, key: str) -> bool:
    rec = index.get(frame_id)
    if rec is None or rec.get("cache_key") != key:
        return False
    rel = rec.get("target_rgb_path")
    return isinstance(rel, str) and (target_dir_path / rel).is_file()
