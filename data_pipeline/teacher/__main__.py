"""CLI: python -m data_pipeline.teacher <command>

  styles                       list the style configs in configs/teacher/
  generate <session_dir>       generate/refresh the sidecar targets for a style
  validate <target_dir>        alignment-validate a sidecar target set

Exit codes: 0 = ok, 1 = rejected/failed, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .alignment import ALIGNMENT_VALIDATOR_VERSION, validate_alignment
from .generate import GENERATOR_VERSION, generate_targets, target_root
from .metrics import Thresholds
from .styles import StyleConfigError, config_dir, list_styles, load_style


def _cmd_styles(args) -> int:
    d = Path(args.config_dir) if args.config_dir else config_dir()
    ids = list_styles(d)
    if not ids:
        print(f"no style configs under {d}", file=sys.stderr)
        return 1
    for sid in ids:
        try:
            s = load_style(sid, d)
        except StyleConfigError as exc:
            print(f"  {sid}: INVALID — {exc}")
            continue
        print(f"  {s.style_id}  backend={s.backend}  seed={s.seed}  digest={s.config_digest[:12]}")
        print(f"      prompt: {s.prompt[:100]}")
    return 0


def _cmd_generate(args) -> int:
    try:
        style = load_style(args.style, args.config_dir)
    except StyleConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not Path(args.session_dir).is_dir():
        print(f"error: {args.session_dir} does not exist", file=sys.stderr)
        return 2

    report = generate_targets(
        args.session_dir,
        style,
        out_root=args.out_root or target_root(),
        backend_name=args.backend,
        start=args.start,
        stride=args.stride,
        limit=args.limit,
        dry_run=args.dry_run,
        force=args.force,
        allow_unvalidated=args.allow_unvalidated,
        progress_every=args.progress_every,
    )

    print(f"teacher {GENERATOR_VERSION}  style={report.style_id}  session={report.session_id}")
    print(f"teacher_version={report.teacher_version}")
    print(f"target_dir={report.target_dir}")
    for e in report.errors:
        print(f"  [ERROR] {e}")
    verb = "would generate" if report.dry_run else "generated"
    print(f"planned={report.planned}  cache_hits={report.cache_hits}  "
          f"{verb}={report.planned - report.cache_hits if report.dry_run else report.generated}  "
          f"complete={report.complete}  {report.seconds:.1f}s")
    if report.ok:
        print("TEACHER RESULT: PASS" + (" (dry run)" if report.dry_run else ""))
        return 0
    print(f"TEACHER RESULT: FAIL  [{len(report.errors)} error(s)]")
    return 1


def _cmd_validate(args) -> int:
    if not Path(args.target_dir).is_dir():
        print(f"error: {args.target_dir} does not exist", file=sys.stderr)
        return 2
    th = Thresholds(
        boundary_f=args.boundary_f,
        spurious_ratio=args.spurious_ratio,
        silhouette_support=args.silhouette_support,
        tolerance_px=args.tolerance_px,
    )
    report = validate_alignment(
        args.target_dir,
        structural_session_dir=args.session_dir,
        thresholds=th,
        frame_limit=args.limit,
        max_fail_frac=args.max_fail_frac,
        write_marker=not args.no_marker,
    )

    if args.report_json:
        out = Path(args.report_json).resolve()
        if out.is_relative_to(Path(args.target_dir).resolve()):
            print("error: --report-json must not write inside the target set "
                  "(the validator writes only the ALIGNED marker there)", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "target_dir": report.target_dir,
            "style_id": report.style_id,
            "session_id": report.session_id,
            "teacher_version": report.teacher_version,
            "thresholds": report.thresholds,
            "stats": report.stats,
            "per_frame": report.per_frame,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"per-frame metrics written to {out}")

    if not args.quiet:
        print(f"alignment {ALIGNMENT_VALIDATOR_VERSION}  style={report.style_id}  "
              f"session={report.session_id}  frames={report.frame_count}")
        print(f"checks: {', '.join(report.checks_run)}")
        for w in report.warnings[:40]:
            print(f"  {w}")
        for e in report.errors[:40]:
            print(f"  {e}")
        if len(report.errors) > 40:
            print(f"  ... and {len(report.errors) - 40} more error(s)")
        dists = report.stats.get("distributions", {})
        for name, d in dists.items():
            if d.get("n"):
                print(f"  {name:24s} n={d['n']:4d} min={d['min']:.3f} p05={d['p05']:.3f} "
                      f"median={d['median']:.3f} mean={d['mean']:.3f} max={d['max']:.3f}")
        print(f"stats: frames_measured={report.stats.get('frames_measured')} "
              f"failing={report.stats.get('frames_failing_a_gate')} "
              f"by_gate={report.stats.get('failures_by_gate')}")

    if report.ok:
        marker = " (ALIGNED marker written)" if report.complete and not args.no_marker else ""
        print(f"ALIGNMENT RESULT: PASS{marker}  [{len(report.warnings)} warning(s)]")
        return 0
    print(f"ALIGNMENT RESULT: FAIL  [{len(report.errors)} error(s), {len(report.warnings)} warning(s)]")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.teacher",
        description="StructRender Milestone 3 — offline teacher targets (sidecar) and "
                    "alignment validation. The structural session is opened read-only.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("styles", help="list style configs")
    ps.add_argument("--config-dir", default=None)
    ps.set_defaults(func=_cmd_styles)

    pg = sub.add_parser("generate", help="generate the sidecar targets for one style")
    pg.add_argument("session_dir", type=Path, help="a recorded m2.0 structural session dir")
    pg.add_argument("--style", required=True, help="style_id (configs/teacher/<style_id>.json)")
    pg.add_argument("--config-dir", default=None)
    pg.add_argument("--out-root", type=Path, default=None,
                    help="target root (default data_pipeline/teacher_targets/, "
                         "env STRUCTRENDER_TARGET_ROOT)")
    pg.add_argument("--backend", default=None, help="override the style's backend (testing)")
    pg.add_argument("--start", type=int, default=0)
    pg.add_argument("--stride", type=int, default=1, help="take every Nth frame (calibration subsets)")
    pg.add_argument("--limit", type=int, default=None, help="stop after N selected frames")
    pg.add_argument("--dry-run", action="store_true",
                    help="walk the whole selection, resolve inputs and cache keys, generate nothing")
    pg.add_argument("--force", action="store_true", help="regenerate even on a cache hit")
    pg.add_argument("--allow-unvalidated", action="store_true",
                    help="allow a structural session with no VALIDATED marker (scratch runs only)")
    pg.add_argument("--progress-every", type=int, default=0)
    pg.set_defaults(func=_cmd_generate)

    pv = sub.add_parser("validate", help="alignment-validate a sidecar target set")
    pv.add_argument("target_dir", type=Path, help="<target_root>/<style_id>/<session_id>/")
    pv.add_argument("--session-dir", type=Path, default=None,
                    help="structural session dir (default: resolved from the session_id)")
    pv.add_argument("--no-marker", action="store_true", help="do not write the ALIGNED marker")
    pv.add_argument("--limit", type=int, default=None,
                    help="only measure the first N targets (debugging; never writes the marker)")
    pv.add_argument("--max-fail-frac", type=float, default=0.0, metavar="F",
                    help="fraction of frames allowed to fail a gate before the set is rejected "
                         "(default 0.0 — strict)")
    pv.add_argument("--boundary-f", type=float, default=Thresholds.boundary_f)
    pv.add_argument("--spurious-ratio", type=float, default=Thresholds.spurious_ratio)
    pv.add_argument("--silhouette-support", type=float, default=Thresholds.silhouette_support)
    pv.add_argument("--tolerance-px", type=int, default=Thresholds.tolerance_px)
    pv.add_argument("--report-json", default=None,
                    help="write per-frame metrics to this path (for threshold calibration); "
                         "must be outside the target set")
    pv.add_argument("--quiet", action="store_true")
    pv.set_defaults(func=_cmd_validate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
