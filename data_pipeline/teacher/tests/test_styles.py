"""Deliverable 2 — selected target style(s).

The load-bearing property here is KB Decision 31's: *"adding a second style must
cost a config file and zero code"*, and that must be **tested**, not aspirational.
``test_second_style_by_config_alone`` is that test.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from data_pipeline.teacher.generate import generate_targets
from data_pipeline.teacher.styles import (
    StyleConfigError,
    config_dir,
    list_styles,
    load_style,
    validate_style_body,
)

REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "teacher"


# --------------------------------------------------------------------------- #
# the shipped configs are schema-valid
# --------------------------------------------------------------------------- #
def test_shipped_style_configs_are_schema_valid():
    ids = list_styles(REPO_CONFIGS)
    assert ids, "configs/teacher/ must ship at least the real style and the CI style"
    for sid in ids:
        style = load_style(sid, REPO_CONFIGS)
        assert style.style_id == sid
        assert style.seed >= 0
        assert style.params


def test_the_real_style_pins_model_id_and_revision():
    """KB §6 'Replace the Offline Teacher' step 3: teacher metadata is versioned.
    A floating revision would make cached targets unreproducible."""
    real = [s for s in (load_style(i, REPO_CONFIGS) for i in list_styles(REPO_CONFIGS))
            if s.backend != "deterministic"]
    assert real, "M3 ships one real (non-CI) style"
    for style in real:
        for key in ("model_id", "revision", "controlnet_model_id", "controlnet_revision"):
            assert style.teacher.get(key), f"{style.style_id}: teacher.{key} unpinned"
        assert style.teacher["revision"] != "main", "revision must be a pinned commit, not a branch"
        assert style.teacher["controlnet_revision"] != "main"


# --------------------------------------------------------------------------- #
# schema validation rejects bad configs
# --------------------------------------------------------------------------- #
def _base() -> dict:
    return json.loads((REPO_CONFIGS / "ci_posterize.json").read_text())


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda c: c.__setitem__("schema_version", "t9.9"), "schema_version"),
        (lambda c: c.__setitem__("style_id", "Not Valid"), "style_id"),
        (lambda c: c.__setitem__("backend", "magic_teacher"), "backend"),
        (lambda c: (c.__setitem__("prompt", ""), c.__setitem__("reference_image", None)), "prompt"),
        (lambda c: c.__setitem__("seed", -1), "seed"),
        (lambda c: c.__setitem__("seed", "abc"), "seed"),
        (lambda c: c.__setitem__("params", {}), "params."),
        (lambda c: c.pop("params"), "params"),
    ],
)
def test_invalid_style_config_is_rejected(mutate, needle):
    cfg = _base()
    mutate(cfg)
    errs = validate_style_body(cfg, expect_style_id=cfg.get("style_id"))
    assert errs, "expected a schema error"
    assert any(needle in e for e in errs), errs


def test_style_id_must_match_the_file_name(tmp_path):
    cfg = _base()
    cfg["style_id"] = "something_else"
    (tmp_path / "ci_posterize.json").write_text(json.dumps(cfg))
    with pytest.raises(StyleConfigError, match="config file stem"):
        load_style("ci_posterize", tmp_path)


def test_diffusion_style_without_pinned_model_is_rejected():
    cfg = json.loads((REPO_CONFIGS / "gouache.json").read_text())
    cfg["teacher"].pop("controlnet_revision")
    errs = validate_style_body(cfg, expect_style_id="gouache")
    assert any("controlnet_revision" in e for e in errs), errs


def test_missing_style_config_names_the_available_ones():
    with pytest.raises(StyleConfigError, match="ci_posterize"):
        load_style("no_such_style", REPO_CONFIGS)


def test_config_dir_is_overridable_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STRUCTRENDER_TEACHER_CONFIG_DIR", str(tmp_path))
    assert config_dir() == tmp_path


# --------------------------------------------------------------------------- #
# THE Decision-31 property: a second style costs a config file and no code
# --------------------------------------------------------------------------- #
def test_second_style_by_config_alone(mini_session, tmp_path, monkeypatch):
    """Add ``configs/teacher/<new>.json`` and nothing else; the new style must
    generate into its own style-keyed directory, produce a distinct
    ``teacher_version``, and land in the target manifest's ``style_ids``.

    No import, registry entry, or branch anywhere in the package is touched.
    """
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    shutil.copy2(REPO_CONFIGS / "ci_posterize.json", cfg_dir / "ci_posterize.json")

    second = json.loads((REPO_CONFIGS / "ci_posterize.json").read_text())
    second["style_id"] = "ci_highkey"
    second["prompt"] = "high-key two-tone poster"
    second["params"]["levels"] = 3          # a visibly different look
    second["seed"] = 7
    (cfg_dir / "ci_highkey.json").write_text(json.dumps(second, indent=2))

    monkeypatch.setenv("STRUCTRENDER_TEACHER_CONFIG_DIR", str(cfg_dir))
    assert list_styles() == ["ci_highkey", "ci_posterize"]

    out_root = tmp_path / "targets"
    a = generate_targets(mini_session, load_style("ci_posterize"), out_root=out_root, limit=4)
    b = generate_targets(mini_session, load_style("ci_highkey"), out_root=out_root, limit=4)
    assert a.ok and b.ok
    assert b.generated == 4

    # style-keyed directory layout, side by side over ONE structural session
    assert Path(a.target_dir).parent.name == "ci_posterize"
    assert Path(b.target_dir).parent.name == "ci_highkey"
    assert Path(a.target_dir).name == Path(b.target_dir).name == a.session_id
    assert Path(a.target_dir).parent.parent == Path(b.target_dir).parent.parent == out_root

    # style-keyed cache: different teacher_version, different cache keys, no collision
    assert a.teacher_version != b.teacher_version
    keys_a = {r["cache_key"] for r in _rows(Path(a.target_dir))}
    keys_b = {r["cache_key"] for r in _rows(Path(b.target_dir))}
    assert keys_a.isdisjoint(keys_b)

    # style_ids propagates into the target manifest
    hb = json.loads((Path(b.target_dir) / "manifest.json").read_text())
    assert hb["style_ids"] == ["ci_highkey"]
    assert hb["style_id"] == "ci_highkey"
    assert hb["teacher"]["style_config"] == "ci_highkey.json"

    # and the pixels actually differ
    pa = (Path(a.target_dir) / "frames" / "frame_00000_target.png").read_bytes()
    pb = (Path(b.target_dir) / "frames" / "frame_00000_target.png").read_bytes()
    assert pa != pb

    # the structural session was not duplicated per style (KB Decision 31)
    assert not any(p.name.startswith("ds_") for p in out_root.iterdir())


def _rows(target_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (target_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
