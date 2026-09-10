"""Engine-Guided Temporal Reprojection — KB §4, Milestone 6."""

from renderer.temporal.confidence import (
    ALPHA,
    BETA,
    ConfidenceMask,
    confidence_mask,
    depth_error,
    hard_valid,
    normal_error,
)
from renderer.temporal.pairs import (
    FramePair,
    consecutive_pairs,
    entities_without_previous_transform,
    load_geometry,
    spawned_pixel_mask,
)
from renderer.temporal.warp import (
    EngineGuidedWarp,
    backward_warp,
    base_uv_grid,
    in_bounds,
    resize_motion,
    sampling_grid,
)

__all__ = [
    "ALPHA",
    "BETA",
    "ConfidenceMask",
    "EngineGuidedWarp",
    "FramePair",
    "backward_warp",
    "base_uv_grid",
    "confidence_mask",
    "consecutive_pairs",
    "depth_error",
    "entities_without_previous_transform",
    "hard_valid",
    "in_bounds",
    "load_geometry",
    "normal_error",
    "resize_motion",
    "sampling_grid",
    "spawned_pixel_mask",
]
