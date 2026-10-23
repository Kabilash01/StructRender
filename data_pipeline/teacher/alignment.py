"""Alignment validation for a sidecar teacher-target set (Milestone 3).

Mirrors the M2 dataset validator's boundaries exactly (KB §4 *Dataset
Validator*, error philosophy "explicit over silent, no auto-repair"):

  - it **reports only**. It never repairs a target, never regenerates one, never
    mutates the target set and never touches the structural session;
  - the single file it may write is the ``ALIGNED`` marker, and only on a clean
    full run (never for a ``--limit`` run or an incomplete target set).

The verdict is the M3 exit criterion made reproducible: "target frames preserve
intended scene layout sufficiently for supervised training".
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..validation.validator import Finding
from .cache import target_rel_path
from .generate import TARGET_SCHEMA_VERSION
from .metrics import Thresholds, distribution, frame_metrics, gate_failures
from .structural import StructuralSession, StructuralSessionError

ALIGNMENT_VALIDATOR_VERSION = "m3-alignment-1"

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_TOPLEVEL = {"manifest.json", "manifest.jsonl", "frames", "ALIGNED"}
_TARGET_FILE_RE = re.compile(r"^frame_(\d{5})_target\.png$")
_SCHEMA_RE = re.compile(r"^t(\d+)\.(\d+)$")


@dataclass
class AlignmentReport:
    target_dir: str
    style_id: str | None = None
    session_id: str | None = None
    schema_version: str | None = None
    frame_count: int | None = None
    teacher_version: str | None = None
    checks_run: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    frame_limit: int | None = None
    max_fail_frac: float = 0.0
    per_frame: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, code: str, message: str) -> None:
        self.findings.append(Finding(code, message, "error"))

    def warn(self, code: str, message: str) -> None:
        self.findings.append(Finding(code, message, "warning"))

    def check(self, name: str) -> None:
        self.checks_run.append(name)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def complete(self) -> bool:
        """Whole target set inspected — the precondition for the marker."""
        return self.ok and self.frame_limit is None and bool(self.stats.get("target_set_complete"))


# --------------------------------------------------------------------------- #
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


def resolve_structural_session(session_id: str, explicit: str | os.PathLike | None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        return p if (p / "manifest.json").is_file() else None
    root = Path(os.environ.get("STRUCTRENDER_DATASET_ROOT", REPO_ROOT / "data_pipeline" / "recorder"))
    cand = root / session_id
    return cand if (cand / "manifest.json").is_file() else None


# --------------------------------------------------------------------------- #
def validate_alignment(
    target_dir: str | os.PathLike,
    *,
    structural_session_dir: str | os.PathLike | None = None,
    thresholds: Thresholds | None = None,
    frame_limit: int | None = None,
    max_fail_frac: float = 0.0,
    write_marker: bool = True,
) -> AlignmentReport:
    th = thresholds or Thresholds()
    target_dir = Path(target_dir).resolve()
    report = AlignmentReport(
        target_dir=str(target_dir),
        thresholds=th.as_dict(),
        frame_limit=frame_limit,
        max_fail_frac=max_fail_frac,
    )

    # -- layout --------------------------------------------------------------
    report.check("layout")
    if not target_dir.is_dir():
        report.error("layout.target_dir", f"not a directory: {target_dir}")
        return report
    _check_layout(target_dir, report)

    header_path = target_dir / "manifest.json"
    jsonl_path = target_dir / "manifest.jsonl"
    if not header_path.is_file() or not jsonl_path.is_file():
        report.error("layout.missing_core", "manifest.json and/or manifest.jsonl absent — cannot continue")
        return report

    # -- header --------------------------------------------------------------
    report.check("header")
    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        report.error("header.parse", f"manifest.json is not valid JSON: {exc}")
        return report
    if not isinstance(header, dict):
        report.error("header.type", "manifest.json is not a JSON object")
        return report
    _check_header(header, target_dir, report)

    # -- the structural session this target set claims to be aligned to ------
    report.check("structural_reference")
    session = None
    sid = header.get("session_id")
    if isinstance(sid, str):
        sdir = resolve_structural_session(sid, structural_session_dir)
        if sdir is None:
            report.error(
                "struct.not_found",
                f"structural session {sid!r} not found (looked at "
                f"{structural_session_dir or os.environ.get('STRUCTRENDER_DATASET_ROOT', 'data_pipeline/recorder/')}) "
                "— alignment cannot be checked without the engine ground truth",
            )
        else:
            try:
                session = StructuralSession(sdir)
            except StructuralSessionError as exc:
                report.error("struct.unreadable", str(exc))
    if session is not None:
        if session.session_id != sid:
            report.error("struct.session_mismatch",
                         f"structural session id {session.session_id!r} != target manifest {sid!r}")
        if not session.validated:
            # An ERROR, not a warning: an ALIGNED marker asserts the targets are
            # usable supervision, and that claim is meaningless over a structural
            # base that never passed M2 — the alignment metrics are computed
            # entirely from that session's own buffers, so if they are wrong the
            # verdict is wrong in the same direction. Warning here also made the
            # M3 validator laxer than the M2 one, which refuses to write
            # VALIDATED for anything short of a clean full run.
            report.error("struct.unvalidated",
                         f"structural session {session.session_id} has no VALIDATED marker — "
                         "alignment is measured against that session's buffers, so an unvalidated "
                         "base makes the verdict unsound (KB §5: consume only validated, frozen "
                         "datasets). Re-run the M2 validator on it.")
        if header.get("resolution") != session.resolution:
            report.error("struct.resolution",
                         f"target resolution {header.get('resolution')!r} != structural {session.resolution!r}")
    if session is None:
        return report

    # Pixel-unit thresholds are stated at the reference resolution; rescale them
    # to this session's (KB Decision 100). The identity at 512x288.
    th = th.scaled_for(session.resolution)
    report.thresholds = th.as_dict()

    # -- per-frame manifest lines -------------------------------------------
    report.check("frame_lines")
    rows = _read_jsonl(jsonl_path, report)
    if rows is None:
        return report
    _check_frame_lines(rows, header, session, report)

    # target count == structural frame count (the M3 brief's cache criterion)
    complete = len(rows) == session.frame_count and [r.get("frame_id") for r in rows] == session.frame_ids()
    report.stats["target_set_complete"] = complete
    report.stats["target_frames"] = len(rows)
    report.stats["structural_frames"] = session.frame_count
    if not complete:
        msg = (f"target set has {len(rows)} frames for a {session.frame_count}-frame structural "
               "session (target count must equal structural frame count, id-for-id)")
        if frame_limit is None:
            report.error("frame.count_mismatch", msg)
        else:
            report.warn("frame.count_mismatch", msg + " — tolerated under --limit")

    # -- alignment metrics ---------------------------------------------------
    report.check("alignment_metrics")
    subset = rows if frame_limit is None else rows[:frame_limit]
    _check_alignment(subset, target_dir, session, th, max_fail_frac, report)

    # -- marker --------------------------------------------------------------
    if report.complete and write_marker:
        try:
            write_aligned_marker(target_dir, report)
            report.stats["marker_written"] = True
        except OSError as exc:
            report.warn("marker.write", f"could not write ALIGNED marker: {exc}")

    return report


# --------------------------------------------------------------------------- #
def _check_layout(target_dir: Path, report: AlignmentReport) -> None:
    frames_dir = target_dir / "frames"
    if not frames_dir.is_dir():
        report.error("layout.frames_dir", "frames/ directory missing")
    for child in sorted(target_dir.iterdir()):
        if child.name not in ALLOWED_TOPLEVEL:
            report.error("layout.stray", f"undeclared artifact in target dir: '{child.name}'")
    if frames_dir.is_dir():
        for child in sorted(frames_dir.iterdir()):
            if not (child.is_file() and _TARGET_FILE_RE.match(child.name)):
                report.error("layout.frames_stray", f"unexpected entry in frames/: '{child.name}'")


def _check_header(header: dict, target_dir: Path, report: AlignmentReport) -> None:
    for key in ("schema_version", "style_id", "style_ids", "session_id", "frame_count",
                "resolution", "complete", "structural_session", "teacher", "provenance"):
        if key not in header:
            report.error("header.missing_key", f"manifest.json missing '{key}'")

    sv = header.get("schema_version")
    report.schema_version = sv if isinstance(sv, str) else None
    if not (isinstance(sv, str) and _SCHEMA_RE.match(sv)):
        report.error("header.schema_version", f"schema_version {sv!r} is not 't<major>.<minor>'")
    elif sv.split(".")[0] != TARGET_SCHEMA_VERSION.split(".")[0]:
        report.error("header.schema_major", f"schema_version {sv!r} major != {TARGET_SCHEMA_VERSION!r}")

    style_id = header.get("style_id")
    report.style_id = style_id if isinstance(style_id, str) else None
    report.session_id = header.get("session_id") if isinstance(header.get("session_id"), str) else None

    # <out_root>/<style_id>/<session_id>/  — style-keyed layout (KB Decision 31)
    if isinstance(style_id, str) and style_id != target_dir.parent.name:
        report.error("header.style_dir",
                     f"style_id {style_id!r} != parent directory {target_dir.parent.name!r} "
                     "(targets live at <root>/<style_id>/<session_id>/)")
    if isinstance(header.get("session_id"), str) and header["session_id"] != target_dir.name:
        report.error("header.session_dir",
                     f"session_id {header['session_id']!r} != directory name {target_dir.name!r}")
    if header.get("style_ids") != [style_id]:
        report.error("header.style_ids", f"style_ids {header.get('style_ids')!r} != [{style_id!r}]")

    teacher = header.get("teacher")
    if not isinstance(teacher, dict):
        report.error("header.teacher", "teacher block missing / not an object")
    else:
        report.teacher_version = teacher.get("teacher_version")
        for key in ("teacher_version", "backend", "seed", "style_config_digest", "params"):
            if key not in teacher:
                report.error("header.teacher_key", f"teacher block missing '{key}' "
                                                   "(KB §6: teacher metadata is versioned)")

    fc = header.get("frame_count")
    report.frame_count = fc if isinstance(fc, int) else None


def _read_jsonl(path: Path, report: AlignmentReport) -> list[dict] | None:
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        report.error("frame_lines.read", f"cannot read manifest.jsonl: {exc}")
        return None
    for i, line in enumerate(lines):
        if not line.strip():
            report.error("frame_lines.blank", f"manifest.jsonl line {i + 1} is blank")
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            report.error("frame_lines.parse", f"manifest.jsonl line {i + 1} is not valid JSON: {exc}")
            continue
        if not isinstance(rec, dict):
            report.error("frame_lines.type", f"manifest.jsonl line {i + 1} is not a JSON object")
            continue
        rows.append(rec)
    if not rows:
        report.error("frame_lines.empty", "manifest.jsonl contains no usable target records")
        return None
    return rows


def _check_frame_lines(rows: list[dict], header: dict, session: StructuralSession,
                       report: AlignmentReport) -> None:
    hdr_sid = header.get("session_id")
    hdr_style = header.get("style_id")
    hdr_tv = (header.get("teacher") or {}).get("teacher_version")
    known_ids = set(session.frame_ids())

    prev = None
    for i, r in enumerate(rows):
        tag = f"manifest.jsonl line {i + 1}"
        for key in ("schema_version", "session_id", "frame_id", "style_id",
                    "target_rgb_path", "cache_key", "teacher_version", "seed"):
            if key not in r:
                report.error("frame.missing_key", f"{tag} missing '{key}'")

        fid = r.get("frame_id")
        if not isinstance(fid, int) or isinstance(fid, bool):
            report.error("frame.frame_id_type", f"{tag}: frame_id {fid!r} is not an int")
            continue
        if prev is not None and fid <= prev:
            report.error("frame.frame_id_order", f"{tag}: frame_id {fid} not increasing (previous {prev})")
        prev = fid

        if fid not in known_ids:
            report.error("frame.frame_id_unknown",
                         f"{tag}: frame_id {fid} does not exist in structural session {session.session_id}")
        if r.get("session_id") != hdr_sid:
            report.error("frame.session_mismatch", f"{tag}: session_id {r.get('session_id')!r} != header")
        if r.get("style_id") != hdr_style:
            report.error("frame.style_mismatch", f"{tag}: style_id {r.get('style_id')!r} != header")
        if r.get("teacher_version") != hdr_tv:
            report.error("frame.teacher_version",
                         f"{tag}: teacher_version {r.get('teacher_version')!r} != header {hdr_tv!r} "
                         "(a mixed-teacher target set is not a valid supervision source)")
        expect_rel = target_rel_path(fid)
        if r.get("target_rgb_path") != expect_rel:
            report.error("frame.target_path",
                         f"{tag}: target_rgb_path {r.get('target_rgb_path')!r} != {expect_rel!r} "
                         "(the path encodes the frame id — this is the id-alignment contract)")
        native = session.record(fid)["native_rgb_path"] if fid in known_ids else None
        if native is not None and "native_rgb_path" in r and r["native_rgb_path"] != native:
            report.error("frame.native_path",
                         f"{tag}: native_rgb_path {r['native_rgb_path']!r} != structural {native!r}")


def _check_alignment(rows: list[dict], target_dir: Path, session: StructuralSession,
                     th: Thresholds, max_fail_frac: float, report: AlignmentReport) -> None:
    from PIL import Image

    W, H = session.resolution
    failing_frames = 0
    per_gate: dict[str, int] = {}

    for r in rows:
        fid = r.get("frame_id")
        if not isinstance(fid, int) or fid not in set(session.frame_ids()):
            continue
        rel = r.get("target_rgb_path")
        if not isinstance(rel, str):
            continue
        path = target_dir / rel
        if not path.is_file():
            report.error("artifact.target_missing", f"frame {fid}: target {rel} does not resolve")
            continue
        try:
            with Image.open(path) as im:
                im.load()
                if im.size != (W, H):
                    report.error("artifact.target_size",
                                 f"frame {fid}: target is {im.size}, expected {(W, H)} "
                                 "(a resized target is not frame-aligned)")
                    continue
                if im.mode not in ("RGB", "RGBA"):
                    report.error("artifact.target_mode", f"frame {fid}: target mode {im.mode!r} unexpected")
                target = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except OSError as exc:
            report.error("artifact.target_decode", f"frame {fid}: target will not decode: {exc}")
            continue

        structure = session.load_structure(fid)
        m = frame_metrics(structure, target, th, native_rgb=session.load_native_rgb(fid))
        report.per_frame.append(m)

        fails = gate_failures(m, th)
        if fails:
            failing_frames += 1
        for code, msg in fails:
            per_gate[code] = per_gate.get(code, 0) + 1
            report.error(code, msg)

    n = len(report.per_frame)
    report.stats["frames_measured"] = n
    report.stats["frames_failing_a_gate"] = failing_frames
    report.stats["failures_by_gate"] = per_gate
    report.stats["fail_frac"] = (failing_frames / n) if n else 0.0
    report.stats["distributions"] = {
        "boundary_f": distribution([m["boundary_f"] for m in report.per_frame]),
        "boundary_recall": distribution([m["boundary_recall"] for m in report.per_frame]),
        "boundary_precision": distribution([m["boundary_precision"] for m in report.per_frame]),
        "spurious_edge_ratio": distribution([m["spurious_edge_ratio"] for m in report.per_frame]),
        "min_silhouette_support": distribution([m["min_silhouette_support"] for m in report.per_frame]),
        "stylisation_hist": distribution([m["stylisation_hist"] for m in report.per_frame]),
    }

    # An explicit, documented tolerance knob (mirrors --dropout-tolerance at M2).
    # Default 0.0 = strict: one bad frame rejects the target set.
    if failing_frames and n and (failing_frames / n) <= max_fail_frac:
        for f in report.findings:
            if f.code.startswith("align.") and f.severity == "error":
                f.severity = "warning"
        report.warn("align.within_tolerance",
                    f"{failing_frames}/{n} frames failed a gate, within --max-fail-frac {max_fail_frac}")


# --------------------------------------------------------------------------- #
def write_aligned_marker(target_dir: str | os.PathLike, report: AlignmentReport) -> Path:
    target_dir = Path(target_dir)
    marker = target_dir / "ALIGNED"
    payload = {
        "alignment_validator_version": ALIGNMENT_VALIDATOR_VERSION,
        "validated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "style_id": report.style_id,
        "session_id": report.session_id,
        "schema_version": report.schema_version,
        "teacher_version": report.teacher_version,
        "frame_count": report.frame_count,
        "checks_run": list(report.checks_run),
        "thresholds": report.thresholds,
        "max_fail_frac": report.max_fail_frac,
        "distributions": report.stats.get("distributions", {}),
        "frames_measured": report.stats.get("frames_measured"),
        "warnings": [str(w) for w in report.warnings],
        "result": "PASS",
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker
