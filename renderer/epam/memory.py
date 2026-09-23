"""EPAM — Engine-Keyed Persistent Appearance Memory. KB §4, Milestone 8.

The protected headline novelty. A per-session store of one compact appearance
embedding per live engine entity, keyed by the engine's `EntityKey`:

    M_i^t = gamma * M_i^{t-1} + (1 - gamma) * F_i^t     only if  C_i^t > tau

Lifecycle, verbatim from §4/§5:

    SPAWN -> first reliable observation -> CREATE -> high-confidence visible
    observation -> UPDATE -> leaves view -> KEEP UNCHANGED -> reappears ->
    LOOKUP SAME MEMORY -> entity destroyed -> DELETE
    style change -> CLEAR          session end -> CLEAR

Boundaries (§4, each enforced here and asserted by a test):
  * keyed by `EntityKey`, which carries `generation` — a recycled instance_id
    is a DIFFERENT key, so it can never read a destroyed entity's memory;
  * no memory after destroy; no cross-session persistence;
  * no stale cross-style memory: every entry records its style, a style change
    clears the store, and a lookup under a different style returns nothing
    even if a clear were somehow missed;
  * no eviction (§4 Capacity Rule: every live, previously seen entity is kept);
  * EPAM never decides whether an entity exists — Godot does. The store only
    reacts to engine events and engine masks.

What memory is FOR in this project is recorded in Decisions 72/73/77: memory
across an occlusion carries no usable residual information here, but committing
to one appearance per entity measurably sharpens (+0.0151, 2.5 sigma, Model E).
The EMA below is that commitment made persistent: an entity's descriptor stops
being re-estimated from scratch each frame and converges toward a stable value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from renderer.entity.keys import EntityKey


@dataclass
class MemoryEntry:
    embedding: torch.Tensor          # [D_memory], detached
    first_seen_frame: int
    last_seen_frame: int
    last_update_frame: int
    confidence: float
    style_id: str


class EPAM:
    """Per-session appearance memory. One instance per session, never shared."""

    def __init__(self, session_id: str, style_id: str, *, gamma: float = 0.9,
                 tau: float = 0.5) -> None:
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0,1), got {gamma}")
        self.session_id = session_id
        self.style_id = style_id
        self.gamma = gamma
        self.tau = tau
        self._m: dict[EntityKey, MemoryEntry] = {}

    # ---- queries -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._m)

    def __contains__(self, key: EntityKey) -> bool:
        return key in self._m

    def lookup(self, key: EntityKey) -> MemoryEntry | None:
        self._check_session(key)
        e = self._m.get(key)
        # Belt and braces for "no stale cross-style memory": set_style clears,
        # but if an entry ever survived a style change it must still not be read.
        if e is None or e.style_id != self.style_id:
            return None
        return e

    # ---- engine-driven updates -----------------------------------------
    def observe(self, key: EntityKey, feature: torch.Tensor, confidence: float,
                frame: int) -> MemoryEntry | None:
        """A visible observation. CREATE on the first reliable one, UPDATE after.

        Below `tau` nothing happens at all — not even `last_seen_frame` moves —
        because a low-confidence observation is exactly the one the §4 update
        rule excludes, and letting it touch the entry would make "unchanged
        while unreliable" false in a way no test of the embedding would see.
        """
        self._check_session(key)
        if confidence <= self.tau:
            return self._m.get(key)
        f = feature.detach().flatten().float()
        e = self._m.get(key)
        if e is None:
            e = MemoryEntry(f.clone(), frame, frame, frame, float(confidence), self.style_id)
            self._m[key] = e
            return e
        if e.embedding.shape != f.shape:
            raise ValueError(f"feature dim {tuple(f.shape)} != memory dim {tuple(e.embedding.shape)}")
        e.embedding = self.gamma * e.embedding + (1.0 - self.gamma) * f
        e.last_seen_frame = frame
        e.last_update_frame = frame
        e.confidence = float(confidence)
        return e

    def destroy(self, key: EntityKey) -> None:
        """Engine destroy event: DELETE. Idempotent — a destroy for an entity that
        was never reliably observed is not an error, it simply had no memory."""
        self._check_session(key)
        self._m.pop(key, None)

    def set_style(self, style_id: str) -> None:
        """Style change: CLEAR (v1 rule). A no-op if the style did not change."""
        if style_id != self.style_id:
            self._m.clear()
            self.style_id = style_id

    def end_session(self) -> None:
        self._m.clear()

    # ---- internals -----------------------------------------------------
    def _check_session(self, key: EntityKey) -> None:
        # No cross-session persistence: a key from another session is a bug in
        # the caller, and silently accepting it is how persistence would leak in.
        if key.session_id != self.session_id:
            raise ValueError(f"key from session {key.session_id!r} used in EPAM for "
                             f"{self.session_id!r}")


def visibility_confidence(pixel_count: int, full_px: int = 1600) -> float:
    """C_i^t from engine visibility: fraction of a 'fully visible' pixel budget,
    clamped to 1. A sliver at a frame edge describes the entity badly, and §4 says
    memory must not update from it.

    `full_px = 1600` is the MEASURED median visible-entity size on ladder-v3 data
    (443 observations: p5 1,025 / p50 1,634 / p95 4,934 px). With tau = 0.5 an
    entity must show >= 800 px to update. The first value tried was 800, chosen
    without measuring; it made the gate inert (every sampled observation cleared
    400 px). Stated plainly: even at 1600 the gate rarely fires in this scene,
    because entities here are large and seldom partially visible — it is a
    guard for edge-of-frame slivers, not an active filter on typical frames.
    """
    return min(1.0, max(0.0, pixel_count / float(full_px)))
