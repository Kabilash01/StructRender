"""Read-only accessor for a recorded m2.0 structural session.

The structural session is **immutable** (KB §5, Decision 31): nothing in this
package opens it for writing, re-records it, or duplicates it per style. Targets
are a sidecar that references it by ``session_id`` + ``frame_id``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .adapter import FrameStructure


class StructuralSessionError(RuntimeError):
    pass


class StructuralSession:
    def __init__(self, session_dir: str | os.PathLike):
        self.dir = Path(session_dir).resolve()
        header_path = self.dir / "manifest.json"
        jsonl_path = self.dir / "manifest.jsonl"
        if not header_path.is_file() or not jsonl_path.is_file():
            raise StructuralSessionError(
                f"{self.dir} is not a recorded session (manifest.json / manifest.jsonl missing)"
            )
        self.header = json.loads(header_path.read_text(encoding="utf-8"))
        self.frames = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._by_id = {f["frame_id"]: f for f in self.frames}

    # -- header shortcuts ----------------------------------------------------
    @property
    def session_id(self) -> str:
        return self.header["session_id"]

    @property
    def schema_version(self) -> str:
        return self.header["schema_version"]

    @property
    def frame_count(self) -> int:
        return int(self.header["frame_count"])

    @property
    def resolution(self) -> list[int]:
        return list(self.header["resolution"])

    @property
    def validated(self) -> bool:
        return (self.dir / "VALIDATED").is_file()

    def frame_ids(self) -> list[int]:
        return [f["frame_id"] for f in self.frames]

    def record(self, frame_id: int) -> dict:
        try:
            return self._by_id[frame_id]
        except KeyError:
            raise StructuralSessionError(f"frame {frame_id} not in {self.session_id}") from None

    # -- buffers -------------------------------------------------------------
    def load_native_rgb(self, frame_id: int) -> np.ndarray:
        from PIL import Image

        rel = self.record(frame_id)["native_rgb_path"]
        with Image.open(self.dir / rel) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def load_motion(self, frame_id: int) -> np.ndarray:
        """``(H, W, 2)`` float32 backward screen-UV motion (KB Decision 25).

        Kept out of :class:`FrameStructure` on purpose: that dataclass is the
        *appearance* input the renderer sees, and KB §4's CNN Stem boundary says
        "motion is not treated as an unexplained appearance channel". Motion has
        an explicit warp operation (Decision 8), so it is loaded by the M6
        temporal path and by nothing else.
        """
        rec = self.record(frame_id)
        rel = rec["structure"]["motion_path"].split("#", 1)[0]
        with np.load(self.dir / rel) as npz:
            return np.asarray(npz["motion"], dtype=np.float32)

    def motion_valid(self, frame_id: int) -> bool:
        """Whole-frame flag: is there a previous frame at all? False on frame 0."""
        return bool(self.record(frame_id)["motion_valid"])

    def load_structure(self, frame_id: int) -> FrameStructure:
        rec = self.record(frame_id)
        rel = rec["structure"]["depth_path"].split("#", 1)[0]
        with np.load(self.dir / rel) as npz:
            depth = np.asarray(npz["depth"], dtype=np.float32)
            normal = np.asarray(npz["normal"], dtype=np.float32)
            albedo = np.asarray(npz["albedo"], dtype=np.float32)
            entity_id = np.asarray(npz["entity_id"], dtype=np.int32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return FrameStructure(
            session_id=self.session_id,
            frame_id=frame_id,
            depth=depth,
            normal=normal,
            albedo=albedo,
            entity_id=entity_id,
        )
