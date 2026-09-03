"""CLI: python -m data_pipeline.validation <session_dir> [options]

Exit code 0 = session valid (VALIDATED marker written unless --no-marker /
--limit), 1 = rejected, 2 = usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .validator import VALIDATOR_VERSION, validate_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_pipeline.validation",
        description="Validate a recorded StructRender dataset session (Milestone 2). "
                    "Reports only — never repairs or mutates the session.",
    )
    parser.add_argument("session_dir", type=Path, help="path to a <dataset_root>/<session_id>/ directory")
    parser.add_argument("--no-marker", action="store_true",
                        help="do not write the VALIDATED marker on success")
    parser.add_argument("--dropout-tolerance", type=int, default=0, metavar="N",
                        help="max cross-channel disagreement pixels per frame before flagging "
                             "partial dropout (default 0 — the reference session produces 0)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="only inspect the first N frames' buffers (debugging shortcut; "
                             "not a full replay, never writes the marker)")
    parser.add_argument("--quiet", action="store_true", help="only print the final verdict line")
    args = parser.parse_args(argv)

    if not args.session_dir.exists():
        print(f"error: {args.session_dir} does not exist", file=sys.stderr)
        return 2

    report = validate_session(
        args.session_dir,
        dropout_tolerance=args.dropout_tolerance,
        frame_limit=args.limit,
        write_marker=not args.no_marker,
    )

    if not args.quiet:
        print(f"validator {VALIDATOR_VERSION}  session={report.session_id}  "
              f"frames={report.frame_count}  schema={report.schema_version}")
        print(f"checks: {', '.join(report.checks_run)}")
        for w in report.warnings:
            print(f"  {w}")
        for e in report.errors:
            print(f"  {e}")
        if report.stats:
            print(f"stats: {report.stats}")

    n_err = len(report.errors)
    if report.ok:
        marker = " (VALIDATED marker written)" if report.replayable and not args.no_marker else ""
        print(f"VALIDATOR RESULT: PASS{marker}  [{len(report.warnings)} warning(s)]")
        return 0
    print(f"VALIDATOR RESULT: FAIL  [{n_err} error(s), {len(report.warnings)} warning(s)]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
