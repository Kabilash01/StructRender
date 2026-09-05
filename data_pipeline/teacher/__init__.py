"""StructRender Milestone 3 — offline teacher target generation + alignment validation.

Contract sources (read these, not this docstring):
  - data_pipeline/schemas/teacher_targets.md   (M3 sidecar-target contract)
  - data_pipeline/schemas/dataset_manifest.md  (the immutable m2.0 structural session)
  - docs/StructRender_KB.md  §3C, §4 (Offline Teacher Target Generator), §6
    ("Replace the Offline Teacher"), §12 (Teacher Structural Distortion),
    Decisions 6, 12, 27, 30, 31.

Boundaries (KB §4):
  - the teacher never runs inside the game loop and is never a runtime dependency;
  - it never owns structure or entity identity — it *consumes* the recorded
    structural buffers and never writes into the structural session;
  - the alignment validator reports only: no repair, no auto-regeneration.
"""

from .adapter import (
    FrameStructure,
    TeacherBackend,
    TeacherOutput,
    get_backend_class,
    make_backend,
    resolve_teacher_version,
)
from .alignment import ALIGNMENT_VALIDATOR_VERSION, AlignmentReport, Thresholds, validate_alignment
from .generate import GENERATOR_VERSION, GenerationReport, generate_targets
from .styles import Style, StyleConfigError, list_styles, load_style

__all__ = [
    "ALIGNMENT_VALIDATOR_VERSION",
    "AlignmentReport",
    "FrameStructure",
    "GENERATOR_VERSION",
    "GenerationReport",
    "Style",
    "StyleConfigError",
    "TeacherBackend",
    "TeacherOutput",
    "Thresholds",
    "generate_targets",
    "get_backend_class",
    "list_styles",
    "load_style",
    "make_backend",
    "resolve_teacher_version",
    "validate_alignment",
]
