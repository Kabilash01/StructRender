"""StructRender Milestone 2 dataset validator.

Reports pass/fail on a recorded dataset session. Never mutates frames or
manifests (KB error philosophy: explicit over silent, no auto-repair). The only
file it may write is the ``VALIDATED`` marker on success.
"""

from .validator import (
    VALIDATOR_VERSION,
    Finding,
    ValidationReport,
    validate_session,
    write_validated_marker,
)

__all__ = [
    "VALIDATOR_VERSION",
    "Finding",
    "ValidationReport",
    "validate_session",
    "write_validated_marker",
]
