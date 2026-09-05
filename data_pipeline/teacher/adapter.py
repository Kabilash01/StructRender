"""Teacher adapter interface — KB §6 "Replace the Offline Teacher".

    generate(native_rgb, structure, style) -> styled_rgb + metadata

Swapping the teacher means implementing :class:`TeacherBackend` and naming it in
a style config. Nothing else changes: target alignment is preserved (the backend
only ever returns an image the same size as the native frame), teacher metadata
is versioned (``BACKEND_VERSION`` + the style config digest), and only the
targets whose ``teacher_version`` changed are regenerated (the cache key carries
it — see :mod:`data_pipeline.teacher.cache`).

Boundary: a backend receives structural buffers *read-only* and returns pixels.
It never sees or emits entity identity as a numeric feature — ``entity_id`` is
carried here only so a future backend can use it as an address/mask
(KB Decision 9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .styles import Style


@dataclass(frozen=True)
class FrameStructure:
    """The deterministic structural buffers of one recorded frame (read-only).

    Loaded from the immutable m2.0 structural session; see
    ``data_pipeline/schemas/frame_structure.md`` for value semantics
    (``depth`` uses ``+inf`` for background, ``normal`` is world-space [-1,1],
    ``albedo`` is linear).
    """

    session_id: str
    frame_id: int
    depth: np.ndarray        # (H, W)    float32, +inf == background
    normal: np.ndarray       # (H, W, 3) float32
    albedo: np.ndarray       # (H, W, 3) float32
    entity_id: np.ndarray    # (H, W)    int32 — an ADDRESS, never a value channel

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self.depth.shape[0], self.depth.shape[1]


@dataclass
class TeacherOutput:
    styled_rgb: np.ndarray            # (H, W, 3) uint8, sRGB
    metadata: dict = field(default_factory=dict)


class TeacherBackend(ABC):
    """One offline teacher implementation.

    ``BACKEND_VERSION`` is a class constant so that ``teacher_version`` — and
    therefore the cache key — can be resolved without constructing the backend
    (``--dry-run`` must not download or load a model).
    """

    NAME: str = ""
    BACKEND_VERSION: str = ""
    #: False for backends whose output is only reproducible on a fixed machine
    #: (KB Decision 31: SD generation is seed-pinned but not bit-exact across
    #: GPU/driver — which is exactly why targets are cached, not regenerated).
    BIT_EXACT: bool = False
    #: True only when identical inputs are guaranteed to produce identical output
    #: **within one run** — i.e. the backend derives no randomness from the frame
    #: (it seeds per style, not per frame). Content-addressed cache reuse
    #: (KB Decision 33) is sound *only* under this property: it serves one render
    #: for every frame sharing the same inputs, which is correct exactly when
    #: regenerating would have produced the same bytes.
    #:
    #: Defaults to False so a new backend must opt in deliberately. Decision 33
    #: warns that a backend seeding per frame breaks the equivalence and must
    #: disable reuse; this flag is what enforces that instead of a comment.
    CONTENT_ADDRESSABLE: bool = False

    def __init__(self, style: Style):
        self.style = style

    @abstractmethod
    def generate(
        self,
        native_rgb: np.ndarray,
        structure: FrameStructure,
        style: Style,
    ) -> TeacherOutput:
        """Return a styled RGB frame the same (H, W) as ``native_rgb``."""

    # -- shared post-condition every backend must satisfy --------------------
    @staticmethod
    def _check_output(styled: np.ndarray, native_rgb: np.ndarray) -> np.ndarray:
        if styled.dtype != np.uint8:
            raise TypeError(f"teacher output dtype {styled.dtype} != uint8")
        if styled.shape != native_rgb.shape:
            raise ValueError(
                f"teacher output shape {styled.shape} != native RGB {native_rgb.shape} "
                "— target alignment is a hard contract (KB §3C.3)"
            )
        return styled


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def get_backend_class(name: str) -> type[TeacherBackend]:
    """Look up a backend class by name. Imports are lazy so that the
    dependency-free ``deterministic`` backend works on a checkout with no
    torch/diffusers installed."""
    if name == "deterministic":
        from .backends.deterministic import DeterministicBackend

        return DeterministicBackend
    if name == "sd15_controlnet_depth":
        from .backends.sd15_controlnet import SD15ControlNetDepthBackend

        return SD15ControlNetDepthBackend
    raise KeyError(f"unknown teacher backend {name!r}")


def make_backend(style: Style, backend: str | None = None) -> TeacherBackend:
    return get_backend_class(backend or style.backend)(style)


def resolve_teacher_version(style: Style, backend: str | None = None) -> str:
    """``<backend_version>/<style config digest>``.

    The style config digest is part of the teacher version on purpose: the cache
    key tuple is fixed at ``(session_id, frame_id, style_id, teacher_version,
    seed)`` (KB Decision 31 / M3 brief), so an edited prompt or generation
    parameter must invalidate the cache *through* ``teacher_version`` or it would
    silently serve stale targets.
    """
    cls = get_backend_class(backend or style.backend)
    return f"{cls.BACKEND_VERSION}/{style.config_digest[:12]}"
