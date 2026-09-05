"""Core validator for a recorded StructRender dataset session (Milestone 2).

Contract sources (read these, do not work from this docstring):
  - data_pipeline/schemas/dataset_manifest.md   (M2 recorder/dataset contract)
  - data_pipeline/schemas/frame_structure.md    (per-channel value semantics)
  - docs/StructRender_KB.md  Decisions 22, 25, 26, 27, 28; Milestone 2

The validator only *reports*. It never repairs, regenerates, or mutates any
frame / manifest. On a fully clean session it writes a ``VALIDATED`` marker
(the only file it is allowed to create).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VALIDATOR_VERSION = "m2-validator-1"

# --- pinned conventions (dataset_manifest.md §0, frame_structure.md §0) --------
SCHEMA_MIN = (2, 0)
EXPECT_RESOLUTION = [512, 288]          # [W, H]
EXPECT_TICK_RATE = 60

# manifest structure key -> (npz member, numpy dtype str, channel count / None=2D)
NPZ_SPEC = {
    "depth_path": ("depth", "<f2", 1),
    "normal_path": ("normal", "<f2", 3),
    "albedo_or_material_path": ("albedo", "<f2", 3),
    "motion_path": ("motion", "<f2", 2),
    "entity_id_map_path": ("entity_id", "<i4", None),
}

_SESSION_ID_RE = re.compile(r"^ds_\d+_[0-9a-f]{8}$")
_SCHEMA_RE = re.compile(r"^m(\d+)\.(\d+)$")
_FRAME_FILE_RE = re.compile(r"^frame_(\d{5})(?:\.npz|_rgb\.png|_camera\.json)$")

ALLOWED_TOPLEVEL = {"manifest.json", "manifest.jsonl", "events.jsonl", "frames", "VALIDATED"}
VALID_EVENTS = {"spawn", "destroy", "visible", "hidden"}

# Cross-agreement / range tolerances. The M1 exporter renders all five channels
# from the same geometry pass and guarantees per-pixel agreement (normal is
# exactly (0,0,0) iff depth is +inf; a non-background entity_id implies rendered
# geometry implies finite depth). Every frame of the reference session produces
# 0 violations, so the default dropout tolerance is 0 (strict). It is exposed as
# a knob only for future half-float silhouette edge cases.
DEFAULT_DROPOUT_TOLERANCE = 0
# KB Decision 33: below this fraction of distinct native frames the session is
# flagged as low-diversity (a warning, not a structural error).
DEFAULT_MIN_DISTINCT_FRAC = 0.5
_NORMAL_ABS_MAX = 1.02
_ALBEDO_MIN = -0.01
_TS_TICK_TOL = 1e-6
_MATRIX_TOL = 1e-4
_UNSEEN_SECONDS_TOL = 1e-6


# ----------------------------------------------------------------------------- #
# report types
# ----------------------------------------------------------------------------- #
@dataclass
class Finding:
    code: str
    message: str
    severity: str = "error"          # "error" | "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    session_dir: str
    session_id: str | None = None
    frame_count: int | None = None
    schema_version: str | None = None
    checks_run: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    dropout_tolerance: int = DEFAULT_DROPOUT_TOLERANCE
    frame_limit: int | None = None
    stats: dict = field(default_factory=dict)

    # -- mutation helpers (used by the checks) --
    def error(self, code: str, message: str) -> None:
        self.findings.append(Finding(code, message, "error"))

    def warn(self, code: str, message: str) -> None:
        self.findings.append(Finding(code, message, "warning"))

    def check(self, name: str) -> None:
        self.checks_run.append(name)

    # -- queries --
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
    def replayable(self) -> bool:
        """Exit criterion (2): the whole session reads back consistently."""
        return self.ok and self.frame_limit is None


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _git_commit(session_dir: Path) -> str:
    env = os.environ.get("STRUCTRENDER_GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "-C", str(session_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite_number(x) -> bool:
    return _is_number(x) and math.isfinite(x)


def _matrix_ok(m) -> bool:
    return isinstance(m, list) and len(m) == 16 and all(_finite_number(v) for v in m)


def _parse_schema(s) -> tuple[int, int] | None:
    if not isinstance(s, str):
        return None
    mo = _SCHEMA_RE.match(s)
    if not mo:
        return None
    return int(mo.group(1)), int(mo.group(2))


# ----------------------------------------------------------------------------- #
# entry point
# ----------------------------------------------------------------------------- #
def validate_session(
    session_dir: str | os.PathLike,
    *,
    dropout_tolerance: int = DEFAULT_DROPOUT_TOLERANCE,
    frame_limit: int | None = None,
    write_marker: bool = True,
    min_distinct_frac: float = DEFAULT_MIN_DISTINCT_FRAC,
) -> ValidationReport:
    """Validate a recorded dataset session directory.

    Returns a :class:`ValidationReport`. ``report.ok`` is the pass/fail verdict.
    When ``frame_limit`` is set only the first N frames' buffers are inspected
    (a debugging shortcut — such a run is *not* a full replay and never writes
    the VALIDATED marker).
    """
    session_dir = Path(session_dir).resolve()
    report = ValidationReport(
        session_dir=str(session_dir),
        dropout_tolerance=dropout_tolerance,
        frame_limit=frame_limit,
    )

    # -- layout (fatal if the core artifacts are missing) --
    report.check("layout")
    if not session_dir.is_dir():
        report.error("layout.session_dir", f"not a directory: {session_dir}")
        return report
    _check_layout(session_dir, report)

    manifest_path = session_dir / "manifest.json"
    jsonl_path = session_dir / "manifest.jsonl"
    events_path = session_dir / "events.jsonl"
    if not manifest_path.is_file() or not jsonl_path.is_file():
        report.error("layout.missing_core", "manifest.json and/or manifest.jsonl absent — cannot continue")
        return report

    # -- header --
    report.check("header")
    try:
        header = _load_json(manifest_path)
    except (json.JSONDecodeError, OSError) as exc:
        report.error("header.parse", f"manifest.json is not valid JSON: {exc}")
        return report
    if not isinstance(header, dict):
        report.error("header.type", "manifest.json is not a JSON object")
        return report

    _check_header(header, session_dir, report)

    # -- per-frame manifest lines --
    report.check("frame_lines")
    frames = _read_jsonl(jsonl_path, report)
    if frames is None:
        return report
    _check_frame_lines(frames, header, report)

    by_id = {f["frame_id"]: f for f in frames if isinstance(f.get("frame_id"), int)}

    # -- sequences --
    report.check("sequences")
    _check_sequences(header, frames, by_id, report)

    # -- per-frame artifacts (paths, camera sidecars, rgb) --
    report.check("frame_artifacts")
    _check_frame_artifacts(frames, header, session_dir, report)

    # -- buffer contents (dtype/shape/range/dropout/consistency) --
    report.check("frame_buffers")
    _check_frame_buffers(frames, session_dir, report, dropout_tolerance, frame_limit)

    # -- events --
    report.check("events")
    _check_events(events_path, header, frames, by_id, report)

    # -- frame diversity (KB Decision 33) --
    report.check("frame_diversity")
    _check_frame_diversity(frames, session_dir, report, min_distinct_frac, frame_limit)

    # -- success marker --
    if report.replayable and write_marker:
        try:
            write_validated_marker(session_dir, report)
            report.stats["marker_written"] = True
        except OSError as exc:  # reporting only — a marker failure is not a data fault
            report.warn("marker.write", f"could not write VALIDATED marker: {exc}")

    return report


# ----------------------------------------------------------------------------- #
# individual checks
# ----------------------------------------------------------------------------- #
def _check_layout(session_dir: Path, report: ValidationReport) -> None:
    """Declared-artifacts-only + debug-log separation (dataset_manifest.md §6.2)."""
    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        report.error("layout.frames_dir", "frames/ directory missing")

    for child in sorted(session_dir.iterdir()):
        name = child.name
        if name in ALLOWED_TOPLEVEL:
            continue
        if name.startswith("session_"):
            report.error(
                "layout.debug_namespace",
                f"'{name}': debug-log 'session_' namespace leaked into a dataset session "
                "(dataset ids are 'ds_'; the two artifacts share no files — KB Decision 22)",
            )
        elif name == "logs" or name.endswith(".log"):
            report.error("layout.debug_leak", f"'{name}': gameplay-log-style artifact inside the dataset session")
        else:
            report.error("layout.stray", f"undeclared artifact in session dir: '{name}'")

    if frames_dir.is_dir():
        for child in sorted(frames_dir.iterdir()):
            if not (child.is_file() and _FRAME_FILE_RE.match(child.name)):
                report.error("layout.frames_stray", f"unexpected entry in frames/: '{child.name}'")


def _check_header(header: dict, session_dir: Path, report: ValidationReport) -> None:
    required = [
        "session_id", "schema_version", "resolution", "frame_count",
        "tick_rate", "has_targets", "style_ids", "sequences", "provenance",
    ]
    for key in required:
        if key not in header:
            report.error("header.missing_key", f"manifest.json missing '{key}'")

    sid = header.get("session_id")
    report.session_id = sid if isinstance(sid, str) else None
    if not isinstance(sid, str):
        report.error("header.session_id", f"session_id {sid!r} is not a string")
    elif sid.startswith("session_"):
        report.error("header.session_id_namespace",
                     f"session_id {sid!r} uses the reserved GameLog debug-log namespace "
                     "(dataset ids are 'ds_<unix>_<8 hex>' — KB Decision 22)")
    elif not _SESSION_ID_RE.match(sid):
        report.error("header.session_id", f"session_id {sid!r} is not 'ds_<unix>_<8 hex>'")
    if isinstance(sid, str) and sid != session_dir.name:
        report.error("header.session_id_dir", f"session_id {sid!r} != directory name {session_dir.name!r}")

    sv = header.get("schema_version")
    report.schema_version = sv if isinstance(sv, str) else None
    parsed = _parse_schema(sv)
    if parsed is None:
        report.error("header.schema_version", f"schema_version {sv!r} is not 'm<major>.<minor>'")
    elif parsed < SCHEMA_MIN:
        report.error("header.schema_too_old", f"schema_version {sv!r} < m{SCHEMA_MIN[0]}.{SCHEMA_MIN[1]}")

    res = header.get("resolution")
    if res != EXPECT_RESOLUTION:
        report.error("header.resolution", f"resolution {res!r} != {EXPECT_RESOLUTION} (a change is a major schema bump)")

    fc = header.get("frame_count")
    if not isinstance(fc, int) or fc <= 0:
        report.error("header.frame_count", f"frame_count {fc!r} is not a positive int")
    else:
        report.frame_count = fc

    if header.get("tick_rate") != EXPECT_TICK_RATE:
        report.error("header.tick_rate", f"tick_rate {header.get('tick_rate')!r} != {EXPECT_TICK_RATE}")

    if header.get("has_targets") is not False:
        report.error("header.has_targets", "has_targets must be false at M2 (styled targets are M3)")
    if header.get("style_ids") != ["native"]:
        report.error("header.style_ids", f"style_ids {header.get('style_ids')!r} != ['native'] at M2")

    prov = header.get("provenance")
    if not isinstance(prov, dict):
        report.error("header.provenance", "provenance missing / not an object")
    else:
        for key in ("recorder_version", "godot_version", "git_commit", "created_unix"):
            if key not in prov:
                report.warn("header.provenance_key", f"provenance missing '{key}'")


def _read_jsonl(path: Path, report: ValidationReport) -> list[dict] | None:
    frames: list[dict] = []
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        report.error("frame_lines.read", f"cannot read manifest.jsonl: {exc}")
        return None
    for i, line in enumerate(raw):
        if not line.strip():
            report.error("frame_lines.blank", f"manifest.jsonl line {i + 1} is blank")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            report.error(
                "frame_lines.parse",
                f"manifest.jsonl line {i + 1} is not valid JSON (truncated/corrupt?): {exc}",
            )
            continue
        if not isinstance(obj, dict):
            report.error("frame_lines.type", f"manifest.jsonl line {i + 1} is not a JSON object")
            continue
        frames.append(obj)
    if not frames:
        report.error("frame_lines.empty", "manifest.jsonl contains no usable frame records")
        return None
    return frames


def _check_frame_lines(frames: list[dict], header: dict, report: ValidationReport) -> None:
    hdr_sv = header.get("schema_version")
    hdr_sid = header.get("session_id")
    tick_rate = header.get("tick_rate") if isinstance(header.get("tick_rate"), (int, float)) else EXPECT_TICK_RATE
    fc = header.get("frame_count")

    required = ["schema_version", "session_id", "frame_id", "timestamp_s", "tick",
                "motion_valid", "structure", "camera", "entities", "style",
                "native_rgb_path", "target_rgb_path"]

    prev_id = None
    for idx, f in enumerate(frames):
        tag = f"manifest.jsonl line {idx + 1}"
        for key in required:
            if key not in f:
                report.error("frame.missing_key", f"{tag} missing '{key}'")

        fid = f.get("frame_id")
        if not isinstance(fid, int) or isinstance(fid, bool):
            report.error("frame.frame_id_type", f"{tag}: frame_id {fid!r} is not an int")
        else:
            if prev_id is None:
                if fid != 0:
                    report.error("frame.frame_id_base", f"{tag}: first frame_id is {fid}, must be 0")
            elif fid != prev_id + 1:
                report.error(
                    "frame.frame_id_consecutive",
                    f"{tag}: frame_id jumped {prev_id} -> {fid} (must be strictly +1; a reduced "
                    "recording rate shows up as a tick jump, never a frame_id gap)",
                )
            prev_id = fid

        if f.get("schema_version") != hdr_sv:
            report.error("frame.schema_mismatch",
                         f"{tag}: schema_version {f.get('schema_version')!r} != header {hdr_sv!r}")
        if f.get("session_id") != hdr_sid:
            report.error("frame.session_mismatch",
                         f"{tag}: session_id {f.get('session_id')!r} != header {hdr_sid!r}")

        tick = f.get("tick")
        ts = f.get("timestamp_s")
        if _is_number(tick) and _is_number(ts) and tick_rate:
            if abs(ts - tick / tick_rate) > _TS_TICK_TOL:
                report.error("frame.timestamp",
                             f"{tag}: timestamp_s {ts} != tick/{tick_rate} ({tick / tick_rate})")

        style = f.get("style")
        if not (isinstance(style, dict) and style.get("style_id") == "native" and style.get("prompt") == ""):
            report.error("frame.style", f"{tag}: style must be {{'style_id':'native','prompt':''}} at M2, got {style!r}")
        if f.get("target_rgb_path") is not None:
            report.error("frame.target_rgb", f"{tag}: target_rgb_path must be null at M2")

        struct = f.get("structure")
        if not isinstance(struct, dict):
            report.error("frame.structure", f"{tag}: structure missing / not an object")
        else:
            for key, (member, _dt, _c) in NPZ_SPEC.items():
                p = struct.get(key)
                if not isinstance(p, str) or "#" not in p:
                    report.error("frame.structure_path", f"{tag}: structure.{key} {p!r} is not '<npz>#<member>'")
                elif p.split("#", 1)[1] != member:
                    report.error("frame.structure_member",
                                 f"{tag}: structure.{key} points at member {p.split('#', 1)[1]!r}, expected {member!r}")

        cam = f.get("camera")
        if not isinstance(cam, dict):
            report.error("frame.camera", f"{tag}: camera missing / not an object")
        else:
            for key in ("view_matrix", "projection_matrix"):
                if not _matrix_ok(cam.get(key)):
                    report.error("frame.camera_matrix",
                                 f"{tag}: camera.{key} is not 16 finite floats")

        if f.get("frame_id") == 0 and f.get("motion_valid") is not False:
            report.error("frame.motion_valid_frame0",
                         f"{tag}: frame 0 must have motion_valid=false (no previous frame)")

        ents = f.get("entities")
        if not isinstance(ents, list):
            report.error("frame.entities", f"{tag}: entities missing / not a list")
        else:
            for e in ents:
                if not isinstance(e, dict):
                    report.error("frame.entity_type", f"{tag}: entity entry is not an object")
                    continue
                for key in ("entity_key", "session_id", "instance_id", "generation",
                            "engine_class", "visible", "spawned", "destroyed"):
                    if key not in e:
                        report.error("frame.entity_key_missing", f"{tag}: entity missing '{key}'")
                iid, gen, esid = e.get("instance_id"), e.get("generation"), e.get("session_id")
                if isinstance(iid, int) and isinstance(gen, int) and isinstance(esid, str):
                    expect = f"{esid}:{iid}:{gen}"
                    if e.get("entity_key") != expect:
                        report.error("frame.entity_key_format",
                                     f"{tag}: entity_key {e.get('entity_key')!r} != {expect!r}")
                if isinstance(iid, int) and iid < 1:
                    report.error("frame.entity_instance_id", f"{tag}: instance_id {iid} < 1")

    if isinstance(fc, int):
        if len(frames) != fc:
            report.error("frame.count_mismatch",
                         f"manifest.jsonl has {len(frames)} records, header frame_count={fc}")
        last = frames[-1].get("frame_id")
        if isinstance(last, int) and last != fc - 1:
            report.error("frame.last_id", f"last frame_id {last} != frame_count-1 ({fc - 1})")


def _visible_of(frame: dict, iid: int):
    for e in frame.get("entities", []):
        if isinstance(e, dict) and e.get("instance_id") == iid:
            return e
    return None


def _check_sequences(header: dict, frames: list[dict], by_id: dict, report: ValidationReport) -> None:
    seqs = header.get("sequences")
    if not isinstance(seqs, list) or not seqs:
        report.error("seq.missing", "manifest.json has no sequences[]")
        return

    fc = header.get("frame_count") if isinstance(header.get("frame_count"), int) else len(frames)
    spans: list[tuple[int, int, str]] = []

    for s in seqs:
        if not isinstance(s, dict):
            report.error("seq.type", "a sequences[] entry is not an object")
            continue
        sid = s.get("sequence_id", "<no id>")
        kind = s.get("sequence_kind")
        rng = s.get("frame_range")
        if kind not in ("temporal", "reappearance"):
            report.error("seq.kind", f"{sid}: sequence_kind {kind!r} not in temporal/reappearance")
        if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, int) for v in rng)):
            report.error("seq.range", f"{sid}: frame_range {rng!r} is not [int, int]")
            continue
        a, b = rng
        if not (0 <= a <= b <= fc - 1):
            report.error("seq.range_bounds", f"{sid}: frame_range {rng} outside [0, {fc - 1}]")
            continue
        spans.append((a, b, str(sid)))

        frames_present = all(fid in by_id for fid in range(a, b + 1))
        if not frames_present:
            report.error("seq.range_frames", f"{sid}: frame_range {rng} references frames absent from manifest.jsonl")

        if kind == "reappearance" and frames_present:
            _check_reappearance(s, sid, a, b, by_id, report)

        # A temporal window is a dense standalone clip (dataset_manifest.md §4/§5):
        # its first frame has no in-window history, so motion_valid must be false
        # (frame_structure.md §4 — "frame 0 of a window").
        if kind == "temporal" and frames_present and by_id[a].get("motion_valid") is not False:
            report.error("seq.temporal_motion_valid",
                         f"{sid}: first frame {a} of a temporal window has motion_valid != false")

    # exactly-one-sequence-per-frame (dataset_manifest.md §4)
    spans.sort()
    for i, (a, b, sid) in enumerate(spans):
        if i > 0 and a <= spans[i - 1][1]:
            report.error("seq.overlap", f"sequences {spans[i - 1][2]} and {sid} overlap at frame {a}")
    if spans:
        union = set()
        for a, b, _ in spans:
            union.update(range(a, b + 1))
        missing = sorted(set(range(fc)) - union)
        if missing:
            report.error("seq.coverage",
                         f"{len(missing)} frame(s) belong to no sequence (first: {missing[0]})")


def _check_reappearance(s: dict, sid, a: int, b: int, by_id: dict, report: ValidationReport) -> None:
    for key in ("target_entity_key", "target_instance_id", "mechanism", "last_seen_frame",
                "first_seen_again_frame", "unseen_frames", "unseen_seconds",
                "identity_stable", "generation_before", "generation_after"):
        if key not in s:
            report.error("reap.missing_key", f"{sid}: reappearance sequence missing '{key}'")

    iid = s.get("target_instance_id")
    if not isinstance(iid, int):
        report.error("reap.target", f"{sid}: target_instance_id {iid!r} is not an int")
        return
    if s.get("mechanism") not in ("camera_away", "entity_away"):
        report.error("reap.mechanism", f"{sid}: mechanism {s.get('mechanism')!r} not camera_away/entity_away")

    # visibility timeline of the target across the (inclusive) range
    vis = []
    for fid in range(a, b + 1):
        e = _visible_of(by_id[fid], iid)
        if e is None:
            report.error("reap.target_absent",
                         f"{sid}: target instance_id {iid} not in entities[] at frame {fid}")
            return
        vis.append((fid, bool(e.get("visible")), e.get("generation")))

    visible_frames = [fid for fid, v, _ in vis if v]
    if not visible_frames:
        report.error("reap.never_visible", f"{sid}: target never visible in frame_range {[a, b]}")
        return

    # gap = the first stretch of not-visible that follows a visible frame
    gap_start = None
    seen_any = False
    for fid, v, _ in vis:
        if v:
            seen_any = True
        elif seen_any:
            gap_start = fid
            break
    if gap_start is None:
        report.error("reap.no_gap",
                     f"{sid}: target is visible across the whole range — no 'visible -> hidden -> visible'")
        return

    exp_last_seen = gap_start - 1
    after = [fid for fid, v, _ in vis if v and fid >= gap_start]
    if not after:
        report.error("reap.no_reappear", f"{sid}: target never becomes visible again after frame {gap_start}")
        return
    exp_first_again = after[0]

    # the gap must be a clean contiguous hidden stretch
    for fid, v, _ in vis:
        if exp_last_seen < fid < exp_first_again and v:
            report.error("reap.dirty_gap",
                         f"{sid}: target visible at frame {fid} inside the supposed unseen span "
                         f"({exp_last_seen} -> {exp_first_again})")

    if s.get("last_seen_frame") != exp_last_seen:
        report.error("reap.last_seen",
                     f"{sid}: last_seen_frame={s.get('last_seen_frame')} but last visible-before-gap is {exp_last_seen}")
    if s.get("first_seen_again_frame") != exp_first_again:
        report.error("reap.first_again",
                     f"{sid}: first_seen_again_frame={s.get('first_seen_again_frame')} but first visible-after-gap "
                     f"is {exp_first_again}")

    ls = s.get("last_seen_frame")
    fs = s.get("first_seen_again_frame")
    if not (isinstance(ls, int) and isinstance(fs, int) and a <= ls < fs <= b):
        report.error("reap.boundary_range",
                     f"{sid}: last/first_seen frames {ls}/{fs} not ordered within frame_range {[a, b]}")
        return

    # identity stable across the gap
    gen_before = _visible_of(by_id[ls], iid).get("generation")
    gen_after = _visible_of(by_id[fs], iid).get("generation")
    if gen_before != gen_after:
        report.error("reap.identity",
                     f"{sid}: generation changed across the gap ({gen_before} -> {gen_after}) — identity not stable")
    if s.get("generation_before") != gen_before:
        report.error("reap.gen_before",
                     f"{sid}: generation_before={s.get('generation_before')} but frame {ls} has generation {gen_before}")
    if s.get("generation_after") != gen_after:
        report.error("reap.gen_after",
                     f"{sid}: generation_after={s.get('generation_after')} but frame {fs} has generation {gen_after}")
    if bool(s.get("identity_stable")) != (gen_before == gen_after):
        report.error("reap.identity_flag",
                     f"{sid}: identity_stable={s.get('identity_stable')} disagrees with generation_before/after")

    # instance_id constant by construction — assert it is the same object entry
    if _visible_of(by_id[ls], iid).get("instance_id") != iid or _visible_of(by_id[fs], iid).get("instance_id") != iid:
        report.error("reap.instance_id", f"{sid}: target instance_id not constant across the gap")

    # unseen_frames / unseen_seconds are DERIVED from the boundary frames (exact)
    fl, ff = by_id[ls], by_id[fs]
    if _is_number(fl.get("tick")) and _is_number(ff.get("tick")):
        exp_uf = ff["tick"] - fl["tick"]
        if s.get("unseen_frames") != exp_uf:
            report.error("reap.unseen_frames",
                         f"{sid}: unseen_frames={s.get('unseen_frames')} but tick[{fs}]-tick[{ls}] = {exp_uf}")
    if _is_number(fl.get("timestamp_s")) and _is_number(ff.get("timestamp_s")):
        exp_us = ff["timestamp_s"] - fl["timestamp_s"]
        if not _is_number(s.get("unseen_seconds")) or abs(s["unseen_seconds"] - exp_us) > _UNSEEN_SECONDS_TOL:
            report.error("reap.unseen_seconds",
                         f"{sid}: unseen_seconds={s.get('unseen_seconds')} but "
                         f"timestamp_s[{fs}]-timestamp_s[{ls}] = {exp_us}")

    gtc = s.get("gap_ticks_configured")
    if _is_number(gtc) and _is_number(s.get("unseen_frames")) and gtc != s.get("unseen_frames"):
        report.warn("reap.gap_configured",
                    f"{sid}: gap_ticks_configured={gtc} != unseen_frames={s.get('unseen_frames')} (cross-check only)")


def _check_frame_artifacts(frames: list[dict], header: dict, session_dir: Path, report: ValidationReport) -> None:
    res = header.get("resolution") if isinstance(header.get("resolution"), list) else EXPECT_RESOLUTION
    W, H = res[0], res[1]
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        report.error("artifact.pillow", "Pillow not installed — cannot check native RGB frames")
        Image = None

    for f in frames:
        fid = f.get("frame_id")
        tag = f"frame {fid}"

        # structure npz files exist (member/dtype/shape checked in _check_frame_buffers)
        struct = f.get("structure") or {}
        for key, (member, _d, _c) in NPZ_SPEC.items():
            p = struct.get(key)
            if not isinstance(p, str) or "#" not in p:
                continue
            rel, mem = p.split("#", 1)
            if not (session_dir / rel).is_file():
                report.error("artifact.npz_missing", f"{tag}: structure.{key} -> {rel} does not resolve")

        # native rgb
        rgb_rel = f.get("native_rgb_path")
        if not isinstance(rgb_rel, str):
            report.error("artifact.rgb_path", f"{tag}: native_rgb_path missing / not a string")
        else:
            rgb_path = session_dir / rgb_rel
            if not rgb_path.is_file():
                report.error("artifact.rgb_missing", f"{tag}: native_rgb_path -> {rgb_rel} does not resolve")
            elif Image is not None:
                try:
                    with Image.open(rgb_path) as im:
                        im.load()
                        if im.size != (W, H):
                            report.error("artifact.rgb_size",
                                         f"{tag}: native RGB is {im.size}, expected {(W, H)}")
                        if im.mode not in ("RGB", "RGBA", "L"):
                            report.error("artifact.rgb_mode", f"{tag}: native RGB mode {im.mode!r} unexpected")
                        extrema = im.getextrema()
                        chans = extrema if isinstance(extrema[0], tuple) else (extrema,)
                        if all(hi == 0 for _lo, hi in chans):
                            report.error("artifact.rgb_black", f"{tag}: native RGB is all-black")
                except OSError as exc:
                    report.error("artifact.rgb_decode", f"{tag}: native RGB will not decode: {exc}")

        # camera sidecar
        cam_path = session_dir / "frames" / f"frame_{fid:05d}_camera.json" if isinstance(fid, int) else None
        if cam_path is None or not cam_path.is_file():
            report.error("artifact.camera_missing", f"{tag}: frame_{fid}_camera.json missing")
            continue
        try:
            cam = _load_json(cam_path)
        except (json.JSONDecodeError, OSError) as exc:
            report.error("artifact.camera_parse", f"{tag}: camera json will not parse: {exc}")
            continue
        for key in ("view_matrix", "projection_matrix"):
            if not _matrix_ok(cam.get(key)):
                report.error("artifact.camera_matrix", f"{tag}: camera json {key} is not 16 finite floats")
        for key in ("near", "far"):
            if not _finite_number(cam.get(key)):
                report.error("artifact.camera_nearfar", f"{tag}: camera json {key} {cam.get(key)!r} not finite")
        if _finite_number(cam.get("near")) and _finite_number(cam.get("far")):
            if not (0.0 <= cam["near"] < cam["far"]):
                report.error("artifact.camera_nearfar_order",
                             f"{tag}: camera near/far not 0<=near<far ({cam.get('near')}, {cam.get('far')})")
        if cam.get("frame_id") != fid:
            report.error("artifact.camera_frame_id", f"{tag}: camera json frame_id {cam.get('frame_id')} != {fid}")
        if cam.get("resolution") not in (res, list(res)):
            report.error("artifact.camera_res", f"{tag}: camera json resolution {cam.get('resolution')} != {res}")
        if cam.get("session_id") != header.get("session_id"):
            report.error("artifact.camera_session", f"{tag}: camera json session_id != header session_id")
        if cam.get("motion_valid") != f.get("motion_valid"):
            report.error("artifact.camera_motion_valid",
                         f"{tag}: camera json motion_valid {cam.get('motion_valid')} != manifest {f.get('motion_valid')}")
        # inline matrices must agree with the sidecar
        inline = f.get("camera") or {}
        for key in ("view_matrix", "projection_matrix"):
            a_, b_ = inline.get(key), cam.get(key)
            if _matrix_ok(a_) and _matrix_ok(b_):
                if any(abs(x - y) > _MATRIX_TOL for x, y in zip(a_, b_)):
                    report.error("artifact.camera_disagree",
                                 f"{tag}: inline camera.{key} disagrees with the sidecar json")


def _check_frame_buffers(
    frames: list[dict],
    session_dir: Path,
    report: ValidationReport,
    dropout_tolerance: int,
    frame_limit: int | None,
) -> None:
    res_frames = frames if frame_limit is None else frames[:frame_limit]
    dropout_hits = 0
    for f in res_frames:
        fid = f.get("frame_id")
        tag = f"frame {fid}"
        struct = f.get("structure") or {}
        # all five members come from one npz in practice; load per unique file
        arrays: dict[str, np.ndarray] = {}
        load_failed = False
        for key, (member, dtype_str, chans) in NPZ_SPEC.items():
            p = struct.get(key)
            if not isinstance(p, str) or "#" not in p:
                load_failed = True
                continue
            rel, mem = p.split("#", 1)
            fpath = session_dir / rel
            if not fpath.is_file():
                load_failed = True
                continue
            try:
                with np.load(fpath) as npz:
                    if mem not in npz.files:
                        report.error("buf.member_missing", f"{tag}: npz {rel} has no member '{mem}'")
                        load_failed = True
                        continue
                    arr = npz[mem]
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                report.error("buf.npz_unreadable", f"{tag}: npz {rel} will not load: {exc}")
                load_failed = True
                continue

            if arr.dtype.str != dtype_str:
                report.error("buf.dtype",
                             f"{tag}: {member} dtype {arr.dtype.str} != {dtype_str}")
            H, W = EXPECT_RESOLUTION[1], EXPECT_RESOLUTION[0]
            want = (H, W) if chans is None else (H, W, chans)
            if arr.shape != want:
                report.error("buf.shape", f"{tag}: {member} shape {arr.shape} != {want}")
            arrays[member] = arr

        if load_failed or len(arrays) != len(NPZ_SPEC):
            continue

        depth = np.asarray(arrays["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth2 = depth[..., 0]
        else:
            depth2 = depth
        normal = np.asarray(arrays["normal"], dtype=np.float32)
        albedo = np.asarray(arrays["albedo"], dtype=np.float32)
        motion = np.asarray(arrays["motion"], dtype=np.float32)
        eid = np.asarray(arrays["entity_id"])

        # -- ranges (frame_structure.md §2) --
        if np.isnan(depth2).any():
            report.error("buf.depth_nan", f"{tag}: depth contains NaN")
        neg_inf = np.isneginf(depth2).any()
        if neg_inf:
            report.error("buf.depth_neginf", f"{tag}: depth contains -inf (only +inf is a valid sentinel)")
        finite = np.isfinite(depth2)
        if finite.any() and (depth2[finite] <= 0.0).any():
            report.error("buf.depth_nonpositive", f"{tag}: depth has finite values <= 0 (must be > 0 in front of camera)")

        if not np.isfinite(normal).all():
            report.error("buf.normal_nonfinite", f"{tag}: normal has non-finite values")
        elif np.abs(normal).max(initial=0.0) > _NORMAL_ABS_MAX:
            report.error("buf.normal_range", f"{tag}: normal component magnitude exceeds {_NORMAL_ABS_MAX}")

        if not np.isfinite(albedo).all():
            report.error("buf.albedo_nonfinite", f"{tag}: albedo has non-finite values")
        elif albedo.min(initial=0.0) < _ALBEDO_MIN:
            report.error("buf.albedo_negative", f"{tag}: albedo has negative values")

        if not np.isfinite(motion).all():
            report.error("buf.motion_nonfinite", f"{tag}: motion has non-finite values")

        if eid.min(initial=0) < 0:
            report.error("buf.entity_id_negative", f"{tag}: entity_id map has negative ids")

        # -- normal unit-length on geometry (frame_structure.md §6) --
        norm_len = np.linalg.norm(normal, axis=-1)
        bad_unit = finite & ((norm_len < 0.9) | (norm_len > 1.1))
        if bad_unit.sum() > dropout_tolerance:
            report.error("buf.normal_not_unit",
                         f"{tag}: {int(bad_unit.sum())} finite-depth pixels have non-unit normal")

        # -- partial channel dropout (KB M2 notes / dataset_manifest.md §6.1) --
        nonzero_normal = (normal != 0).any(axis=-1)
        ent = eid != 0
        s1 = int((ent & ~finite).sum())                    # entity pixel lost depth
        s2 = int((ent & ~nonzero_normal).sum())            # entity pixel lost normal
        s3 = int((finite & ~nonzero_normal).sum())         # geometry pixel lost normal
        s4 = int((~finite & nonzero_normal).sum())         # background pixel with spurious normal / depth dropout
        worst = max(s1, s2, s3, s4)
        if worst > dropout_tolerance:
            dropout_hits += 1
            report.error(
                "buf.partial_dropout",
                f"{tag}: partial channel dropout — a subset of geometry pixels reads as background "
                f"(entity&+inf-depth={s1}, entity&zero-normal={s2}, finite-depth&zero-normal={s3}, "
                f"+inf-depth&nonzero-normal={s4}; tolerance={dropout_tolerance})",
            )

        # -- entity_id map vs manifest entities[] --
        map_ids = set(int(v) for v in np.unique(eid) if v != 0)
        ents = f.get("entities") or []
        declared = {e.get("instance_id") for e in ents if isinstance(e, dict)}
        stray = map_ids - declared
        if stray:
            report.error("buf.entity_id_unregistered",
                         f"{tag}: entity_id map contains ids {sorted(stray)} with no entities[] record")
        for e in ents:
            if not isinstance(e, dict):
                continue
            iid = e.get("instance_id")
            present = iid in map_ids
            if bool(e.get("visible")) != present:
                report.error("buf.visible_flag",
                             f"{tag}: entity {iid} visible={e.get('visible')} but "
                             f"{'present in' if present else 'absent from'} the entity_id map")

        # -- motion_valid semantics (frame_structure.md §4) --
        mv = f.get("motion_valid")
        motion_absmax = float(np.abs(motion).max(initial=0.0))
        if mv is False and motion_absmax > 0.0:
            report.error("buf.motion_valid_false_nonzero",
                         f"{tag}: motion_valid=false but motion buffer is non-zero (absmax={motion_absmax})")
        if fid == 0 and mv is True:
            report.error("buf.motion_valid_frame0", f"{tag}: motion_valid=true on frame 0")

    report.stats["frames_buffer_checked"] = len(res_frames)
    report.stats["frames_with_dropout"] = dropout_hits


def _check_frame_diversity(
    frames: list[dict],
    session_dir: Path,
    report: ValidationReport,
    min_distinct_frac: float,
    frame_limit: int | None,
) -> None:
    """How many of the recorded frames are actually *different* (KB Decision 33).

    Every other check in this validator asks whether the session is internally
    correct. A session can pass all of them and still be near-useless: the first
    M2 seed held its poses through each reappearance gap and shipped 990 frames
    that were only **163 distinct images**, three of them repeated 295/295/240
    times. Sequence semantics were perfect, so nothing failed — and a downstream
    "model overfits a tiny set" criterion would have passed vacuously against
    what is effectively three pictures.

    That is why this is a check and not a note in a document: a defect that makes
    success criteria *easier* to meet cannot be left to whoever remembers to look.

    Reported as a WARNING, not an error: a short or deliberately static session
    is legitimate, and this validator's errors mean "structurally invalid". The
    warning is recorded in the VALIDATED marker, so it stays visible.
    """
    if frame_limit is not None:
        return  # a partial run cannot judge the whole session's diversity

    digests: set[str] = set()
    counts: dict[str, int] = {}
    missing = 0
    for f in frames:
        rel = f.get("native_rgb_path")
        if not isinstance(rel, str):
            continue
        p = session_dir / rel
        if not p.is_file():
            missing += 1
            continue
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        digests.add(d)
        counts[d] = counts.get(d, 0) + 1

    n = len(frames)
    if not n or missing == n:
        return
    distinct = len(digests)
    frac = distinct / n
    worst = max(counts.values()) if counts else 0
    report.stats["distinct_frames"] = distinct
    report.stats["distinct_frac"] = round(frac, 4)
    report.stats["most_repeated_frame"] = worst

    if frac < min_distinct_frac:
        report.warn(
            "diversity.low",
            f"only {distinct}/{n} native frames are distinct ({frac:.1%} < "
            f"{min_distinct_frac:.0%}); the most repeated frame appears {worst}x. "
            "The session is structurally valid but carries little visual variety, "
            "which silently weakens anything trained or measured on it "
            "(KB Decision 33). Check that the recorder keeps the scene moving.",
        )


def _check_events(events_path: Path, header: dict, frames: list[dict], by_id: dict, report: ValidationReport) -> None:
    if not events_path.is_file():
        report.error("events.missing", "events.jsonl missing")
        return
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        report.error("events.read", f"cannot read events.jsonl: {exc}")
        return

    fc = header.get("frame_count") if isinstance(header.get("frame_count"), int) else len(frames)
    hdr_sid = header.get("session_id")
    events: list[dict] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as exc:
            report.error("events.parse", f"events.jsonl line {i + 1} invalid JSON: {exc}")
            continue
        for key in ("session_id", "instance_id", "generation", "event", "frame_id"):
            if key not in e:
                report.error("events.key", f"events.jsonl line {i + 1} missing '{key}'")
        if e.get("event") not in VALID_EVENTS:
            report.error("events.event", f"events.jsonl line {i + 1}: event {e.get('event')!r} invalid")
        if e.get("session_id") != hdr_sid:
            report.error("events.session", f"events.jsonl line {i + 1}: session_id != header")
        fid = e.get("frame_id")
        if not isinstance(fid, int) or not (0 <= fid < fc):
            report.error("events.frame_id", f"events.jsonl line {i + 1}: frame_id {fid!r} out of range")
        events.append(e)

    # spawn / destroy must match the per-frame manifest flags exactly
    def _flag(fid, iid, name):
        fr = by_id.get(fid)
        if fr is None:
            return None
        e = _visible_of(fr, iid)
        return None if e is None else bool(e.get(name))

    spawn_ev = {(e["frame_id"], e["instance_id"]) for e in events
                if e.get("event") == "spawn" and isinstance(e.get("frame_id"), int)}
    destroy_ev = {(e["frame_id"], e["instance_id"]) for e in events
                  if e.get("event") == "destroy" and isinstance(e.get("frame_id"), int)}

    for (fid, iid) in spawn_ev:
        if _flag(fid, iid, "spawned") is not True:
            report.error("events.spawn_flag",
                         f"spawn event (frame {fid}, instance {iid}) but manifest entity.spawned is not true")
    for (fid, iid) in destroy_ev:
        if _flag(fid, iid, "destroyed") is not True:
            report.error("events.destroy_flag",
                         f"destroy event (frame {fid}, instance {iid}) but manifest entity.destroyed is not true")

    # reverse: every manifest spawned/destroyed flag needs its event
    for fr in frames:
        fid = fr.get("frame_id")
        for e in fr.get("entities", []):
            if not isinstance(e, dict):
                continue
            iid = e.get("instance_id")
            if e.get("spawned") is True and (fid, iid) not in spawn_ev:
                report.error("events.spawn_missing",
                             f"manifest frame {fid} entity {iid} spawned=true but no spawn event")
            if e.get("destroyed") is True and (fid, iid) not in destroy_ev:
                report.error("events.destroy_missing",
                             f"manifest frame {fid} entity {iid} destroyed=true but no destroy event")

    # visible / hidden events must line up with visible-flag transitions
    for e in events:
        if e.get("event") not in ("visible", "hidden"):
            continue
        fid, iid = e.get("frame_id"), e.get("instance_id")
        if not isinstance(fid, int):
            continue
        want_visible = e["event"] == "visible"
        cur = _flag(fid, iid, "visible")
        if cur is None:
            report.error("events.vis_absent",
                         f"{e['event']} event (frame {fid}, instance {iid}) but entity absent from that frame")
            continue
        if cur is not want_visible:
            report.error("events.vis_state",
                         f"{e['event']} event at frame {fid} but entity {iid} visible={cur} there")
        prev = _flag(fid - 1, iid, "visible") if fid > 0 else None
        if prev is not None and prev == want_visible:
            report.error("events.vis_no_transition",
                         f"{e['event']} event at frame {fid} for entity {iid} but visible flag did not change from frame {fid - 1}")


# ----------------------------------------------------------------------------- #
# success marker
# ----------------------------------------------------------------------------- #
def write_validated_marker(session_dir: str | os.PathLike, report: ValidationReport) -> Path:
    session_dir = Path(session_dir)
    marker = session_dir / "VALIDATED"
    payload = {
        "validator_version": VALIDATOR_VERSION,
        "validated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(session_dir),
        "session_id": report.session_id,
        "schema_version": report.schema_version,
        "frame_count": report.frame_count,
        "checks_run": list(report.checks_run),
        "dropout_tolerance": report.dropout_tolerance,
        "warnings": [str(w) for w in report.warnings],
        "result": "PASS",
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker
