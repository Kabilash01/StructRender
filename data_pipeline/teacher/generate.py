"""Sidecar teacher-target generation driver.

Writes ``<out_root>/<style_id>/<session_id>/`` and **nothing else** — the m2.0
structural session is opened read-only (KB Decision 31).

Cache behaviour (KB §3C.5): a frame whose ``(session_id, frame_id, style_id,
teacher_version, seed)`` key already has a manifest line and a PNG on disk is
skipped. A second run over an unchanged session therefore generates zero frames,
and an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .adapter import get_backend_class, make_backend, resolve_teacher_version
from .cache import cache_key, content_key, is_hit, read_index, target_dir, target_rel_path
from .structural import StructuralSession
from .styles import Style

GENERATOR_VERSION = "m3-teacher-1"
TARGET_SCHEMA_VERSION = "t1.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_ROOT = REPO_ROOT / "data_pipeline" / "teacher_targets"


def target_root() -> Path:
    env = os.environ.get("STRUCTRENDER_TARGET_ROOT")
    return Path(env) if env else DEFAULT_TARGET_ROOT


@dataclass
class GenerationReport:
    style_id: str
    session_id: str
    target_dir: str
    teacher_version: str
    planned: int = 0
    generated: int = 0
    cache_hits: int = 0
    content_reused: int = 0
    dry_run: bool = False
    seconds: float = 0.0
    complete: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _git_commit() -> str:
    env = os.environ.get("STRUCTRENDER_GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Atomic PNG write: an interrupted run never leaves a half-written target
    that would later look like a cache hit."""
    from PIL import Image

    tmp = path.with_suffix(".png.tmp")
    Image.fromarray(rgb, mode="RGB").save(tmp, format="PNG", optimize=False)
    os.replace(tmp, path)


def select_frame_ids(session: StructuralSession, *, start: int, stride: int, limit: int | None) -> list[int]:
    ids = [fid for fid in session.frame_ids() if fid >= start]
    if stride > 1:
        ids = ids[::stride]
    if limit is not None:
        ids = ids[:limit]
    return ids


def generate_targets(
    session_dir: str | os.PathLike,
    style: Style,
    *,
    out_root: str | os.PathLike | None = None,
    backend_name: str | None = None,
    start: int = 0,
    stride: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    allow_unvalidated: bool = False,
    progress_every: int = 0,
) -> GenerationReport:
    t0 = time.time()
    session = StructuralSession(session_dir)
    out_root = Path(out_root) if out_root is not None else target_root()
    tdir = target_dir(out_root, style.style_id, session.session_id)
    teacher_version = resolve_teacher_version(style, backend_name)

    report = GenerationReport(
        style_id=style.style_id,
        session_id=session.session_id,
        target_dir=str(tdir),
        teacher_version=teacher_version,
        dry_run=dry_run,
    )

    if not session.validated and not allow_unvalidated:
        report.errors.append(
            f"structural session {session.session_id} has no VALIDATED marker — targets must be "
            "derived from a validated, frozen session (KB §5). Re-run the M2 validator, or pass "
            "--allow-unvalidated for a scratch run."
        )
        return report

    frame_ids = select_frame_ids(session, start=start, stride=stride, limit=limit)
    report.planned = len(frame_ids)
    complete_run = (start == 0 and stride == 1 and limit is None)

    index = read_index(tdir) if tdir.is_dir() else {}
    keys = {
        fid: cache_key(session.session_id, fid, style.style_id, teacher_version, style.seed)
        for fid in frame_ids
    }

    # ---- dry run: walk the whole selection without generating --------------
    if dry_run:
        missing_inputs: list[str] = []
        for fid in frame_ids:
            rec = session.record(fid)
            npz_rel = rec["structure"]["depth_path"].split("#", 1)[0]
            for rel in (rec["native_rgb_path"], npz_rel):
                if not (session.dir / rel).is_file():
                    missing_inputs.append(f"frame {fid}: missing input {rel}")
            if is_hit(tdir, index, fid, keys[fid]) and not force:
                report.cache_hits += 1
        report.errors.extend(missing_inputs[:20])
        report.seconds = time.time() - t0
        report.complete = complete_run and not missing_inputs
        return report

    # ---- real run ----------------------------------------------------------
    (tdir / "frames").mkdir(parents=True, exist_ok=True)
    backend = None
    jsonl_path = tdir / "manifest.jsonl"
    teacher_metadata: dict = {}

    # Content-addressed reuse is only sound when the backend derives no
    # randomness from the frame (KB Decision 33). Read off the CLASS so this
    # resolves without constructing the backend or loading a model.
    content_addressable = bool(
        getattr(get_backend_class(backend_name or style.backend), "CONTENT_ADDRESSABLE", False)
    )

    # content_key -> the frame_id whose PNG holds that render. Primed from the
    # existing manifest so a resumed run reuses what an earlier one produced.
    # NOT primed under --force: force means "regenerate", and seeding from the
    # old index would have it silently reuse the very renders it was asked to
    # replace (and, for a frame that is its own recorded source, copy a file
    # onto itself).
    content_seen: dict[str, int] = {}
    if not force and content_addressable:
        for _fid, _rec in index.items():
            _ck = _rec.get("content_key")
            if isinstance(_ck, str) and _rec.get("content_source_frame") == _fid:
                content_seen.setdefault(_ck, _fid)

    with jsonl_path.open("a", encoding="utf-8") as fh:
        for n, fid in enumerate(frame_ids):
            key = keys[fid]
            if not force and is_hit(tdir, index, fid, key):
                report.cache_hits += 1
                continue

            native = session.load_native_rgb(fid)
            structure = session.load_structure(fid)

            # Content-addressed reuse: a session can repeat the same rendered
            # moment many times (a held pose across a reappearance gap). The
            # backend seeds per style, not per frame, so identical inputs are
            # guaranteed to yield identical output — reusing the earlier render
            # writes exactly what regenerating would have. See cache.content_key.
            ckey = content_key(native, structure.depth, teacher_version, style.seed)
            if not content_addressable:
                twin = None
            else:
                twin = content_seen.get(ckey)

            rel = target_rel_path(fid)

            if twin is not None and twin != fid and (tdir / target_rel_path(twin)).is_file():
                shutil.copyfile(tdir / target_rel_path(twin), tdir / rel)
                report.content_reused += 1
            else:
                if backend is None:
                    backend = make_backend(style, backend_name)
                out = backend.generate(native, structure, style)
                if out.styled_rgb.shape[:2] != native.shape[:2]:
                    report.errors.append(f"frame {fid}: teacher output shape != native RGB")
                    break
                teacher_metadata = out.metadata
                _write_png(tdir / rel, out.styled_rgb)
                content_seen[ckey] = fid

            rec = {
                "schema_version": TARGET_SCHEMA_VERSION,
                "session_id": session.session_id,
                "frame_id": fid,
                "style_id": style.style_id,
                "target_rgb_path": rel,
                "native_rgb_path": session.record(fid)["native_rgb_path"],
                "cache_key": key,
                "content_key": ckey,
                "content_source_frame": twin if twin is not None else fid,
                "teacher_version": teacher_version,
                "seed": style.seed,
                "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            }
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            index[fid] = rec
            report.generated += 1
            if progress_every and (n + 1) % progress_every == 0:
                print(f"  [{n + 1}/{len(frame_ids)}] generated={report.generated} "
                      f"hits={report.cache_hits}", flush=True)

    # canonical ordering: rewrite the append-only log sorted by frame_id
    ordered = [index[fid] for fid in sorted(index)]
    jsonl_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in ordered) + "\n",
                          encoding="utf-8")

    report.complete = complete_run and len(index) == session.frame_count and not report.errors
    _write_header(tdir, session, style, teacher_version, ordered, teacher_metadata, report)
    report.seconds = time.time() - t0
    return report


def _write_header(
    tdir: Path,
    session: StructuralSession,
    style: Style,
    teacher_version: str,
    ordered: list[dict],
    teacher_metadata: dict,
    report: GenerationReport,
) -> None:
    path = tdir / "manifest.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    header = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "style_id": style.style_id,
        "style_ids": [style.style_id],
        "session_id": session.session_id,
        "frame_count": len(ordered),
        "resolution": session.resolution,
        "complete": report.complete,
        "structural_session": {
            "session_id": session.session_id,
            "schema_version": session.schema_version,
            "frame_count": session.frame_count,
            "validated": session.validated,
            "relative_hint": f"data_pipeline/recorder/{session.session_id}",
        },
        "teacher": {
            "teacher_version": teacher_version,
            "backend": style.backend,
            "seed": style.seed,
            "style_config": str(Path(style.config_path).name),
            "style_config_digest": style.config_digest,
            "prompt": style.prompt,
            "negative_prompt": style.negative_prompt,
            "reference_image": style.reference_image,
            "params": style.params,
            "model": style.teacher,
            "metadata": teacher_metadata or previous.get("teacher", {}).get("metadata", {}),
        },
        "provenance": {
            "generator_version": GENERATOR_VERSION,
            "git_commit": _git_commit(),
            "kb_decisions_in_force": ["6", "9", "12", "27", "30", "31"],
            "created_unix": previous.get("provenance", {}).get("created_unix", int(time.time())),
            "updated_unix": int(time.time()),
        },
    }
    path.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8")
