"""Style registry — ``configs/teacher/<style_id>.json``.

KB Decision 31: one real style at M3, but *every* path is style-keyed from day
one. Adding a second style must cost a config file and zero code, so this module
is the only place that knows what a style is, and it discovers styles by
directory listing rather than from a hardcoded list.

A style config pins everything that changes the pixels a teacher produces:
prompt/reference, teacher model id + revision, seed, and generation params. Its
canonical content digest is folded into ``teacher_version`` (see
``adapter.resolve_teacher_version``) so an edited prompt can never be served
from a stale cache entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "teacher"

STYLE_SCHEMA_VERSION = "t1.0"
_STYLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

# Per-backend required keys. Kept here (not in the backends) so that a config can
# be validated without importing torch/diffusers.
_BACKEND_REQUIRED_PARAMS = {
    "deterministic": ("levels", "smooth_iters", "sigma_color", "edge_strength", "edge_threshold"),
    "sd15_controlnet_depth": (
        "num_inference_steps",
        "guidance_scale",
        "strength",
        "controlnet_conditioning_scale",
    ),
}
_BACKEND_REQUIRED_MODEL_KEYS = {
    "deterministic": (),
    "sd15_controlnet_depth": ("model_id", "revision", "controlnet_model_id", "controlnet_revision"),
}


class StyleConfigError(ValueError):
    """A style config is missing/malformed. Raised, never silently defaulted."""


@dataclass(frozen=True)
class Style:
    style_id: str
    backend: str
    prompt: str
    negative_prompt: str
    reference_image: str | None
    teacher: dict
    seed: int
    params: dict
    config_path: str
    config_digest: str          # sha256 of the canonical config body

    @property
    def has_reference(self) -> bool:
        return bool(self.reference_image)


def config_dir() -> Path:
    """Style config directory. ``$STRUCTRENDER_TEACHER_CONFIG_DIR`` overrides —
    the tests use it to prove a new style needs no code change."""
    env = os.environ.get("STRUCTRENDER_TEACHER_CONFIG_DIR")
    return Path(env) if env else DEFAULT_CONFIG_DIR


def list_styles(directory: str | os.PathLike | None = None) -> list[str]:
    d = Path(directory) if directory is not None else config_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_style(style_id: str, directory: str | os.PathLike | None = None) -> Style:
    d = Path(directory) if directory is not None else config_dir()
    path = d / f"{style_id}.json"
    if not path.is_file():
        raise StyleConfigError(
            f"no style config for {style_id!r} at {path} "
            f"(available: {', '.join(list_styles(d)) or 'none'})"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StyleConfigError(f"{path} is not valid JSON: {exc}") from exc
    return parse_style(raw, path)


def parse_style(raw: object, path: Path) -> Style:
    """Schema-validate a style config body and build a :class:`Style`."""
    errs = validate_style_body(raw, expect_style_id=path.stem)
    if errs:
        raise StyleConfigError(f"{path}: " + "; ".join(errs))
    assert isinstance(raw, dict)  # guaranteed by validate_style_body

    body = {k: v for k, v in raw.items() if k != "notes"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return Style(
        style_id=raw["style_id"],
        backend=raw["backend"],
        prompt=raw["prompt"],
        negative_prompt=raw.get("negative_prompt", ""),
        reference_image=raw.get("reference_image") or None,
        teacher=dict(raw.get("teacher") or {}),
        seed=int(raw["seed"]),
        params=dict(raw["params"]),
        config_path=str(path),
        config_digest=digest,
    )


def validate_style_body(raw: object, expect_style_id: str | None = None) -> list[str]:
    """Return a list of human-readable schema problems (empty == valid).

    Hand-rolled rather than jsonschema: the schema is small and M3 should not add
    a dependency for it.
    """
    errs: list[str] = []
    if not isinstance(raw, dict):
        return ["style config is not a JSON object"]

    sv = raw.get("schema_version")
    if sv != STYLE_SCHEMA_VERSION:
        errs.append(f"schema_version {sv!r} != {STYLE_SCHEMA_VERSION!r}")

    sid = raw.get("style_id")
    if not isinstance(sid, str) or not _STYLE_ID_RE.match(sid):
        errs.append(f"style_id {sid!r} is not a lowercase [a-z0-9_] identifier")
    elif expect_style_id is not None and sid != expect_style_id:
        errs.append(f"style_id {sid!r} != config file stem {expect_style_id!r}")

    backend = raw.get("backend")
    if backend not in _BACKEND_REQUIRED_PARAMS:
        errs.append(
            f"backend {backend!r} unknown (known: {', '.join(sorted(_BACKEND_REQUIRED_PARAMS))})"
        )

    prompt = raw.get("prompt")
    ref = raw.get("reference_image")
    if not isinstance(prompt, str):
        errs.append("prompt must be a string (may be empty when reference_image is set)")
    elif not prompt.strip() and not ref:
        errs.append("style needs a prompt or a reference_image (KB §4: 'style prompt or reference')")
    if ref is not None and not isinstance(ref, str):
        errs.append(f"reference_image {ref!r} must be a string path or null")

    if "negative_prompt" in raw and not isinstance(raw["negative_prompt"], str):
        errs.append("negative_prompt must be a string")

    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errs.append(f"seed {seed!r} must be a non-negative int (generation is seed-pinned)")

    params = raw.get("params")
    if not isinstance(params, dict):
        errs.append("params must be an object")
        params = {}

    teacher = raw.get("teacher")
    if teacher is not None and not isinstance(teacher, dict):
        errs.append("teacher must be an object")
        teacher = {}
    teacher = teacher or {}

    if backend in _BACKEND_REQUIRED_PARAMS:
        for key in _BACKEND_REQUIRED_PARAMS[backend]:
            if key not in params:
                errs.append(f"params.{key} required for backend {backend!r}")
        for key in _BACKEND_REQUIRED_MODEL_KEYS[backend]:
            if not isinstance(teacher.get(key), str) or not teacher[key]:
                errs.append(
                    f"teacher.{key} required for backend {backend!r} "
                    "(KB §6 'Replace the Offline Teacher': teacher metadata is versioned)"
                )

    return errs
