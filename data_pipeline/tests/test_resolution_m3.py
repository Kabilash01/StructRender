"""KB Decision 100: the 768x432 m3.0 dataset — resolution pinned per schema major,
and the alignment gates' pixel-unit parameters scaled with resolution."""

from pathlib import Path

import pytest

from data_pipeline.teacher.metrics import REFERENCE_RESOLUTION, Thresholds
from data_pipeline.validation.validator import ValidationReport, _check_header


def test_scaling_is_the_identity_at_the_reference_resolution():
    assert Thresholds().scaled_for(REFERENCE_RESOLUTION) == Thresholds()


def test_pixel_parameters_scale_length_linearly_and_area_quadratically():
    t = Thresholds().scaled_for((768, 432))
    assert (t.tolerance_px, t.min_entity_px) == (3, 450)
    # every unit-free threshold is untouched
    base = Thresholds().as_dict()
    for k, v in t.as_dict().items():
        if k not in ("tolerance_px", "min_entity_px"):
            assert v == base[k], k


def test_a_non_reference_aspect_is_refused():
    with pytest.raises(ValueError):
        Thresholds().scaled_for((768, 480))


def _header_errors(schema, res):
    rep = ValidationReport(session_dir="x")
    _check_header({"schema_version": schema, "resolution": res, "frame_count": 1}, Path("x"), rep)
    return [f for f in rep.findings if f.code == "header.resolution"]


@pytest.mark.parametrize("schema,res,ok", [
    ("m2.0", [512, 288], True), ("m3.0", [768, 432], True),
    ("m2.0", [768, 432], False), ("m3.0", [512, 288], False),
])
def test_resolution_is_pinned_per_schema_major(schema, res, ok):
    assert (not _header_errors(schema, res)) == ok
