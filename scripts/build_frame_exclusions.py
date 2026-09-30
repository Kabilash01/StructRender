"""Per-frame exclusion list for the two-style game dataset — KB Decision 94.

For every game session and both styles, run the M3 validator ITSELF
(`validate_alignment`, default thresholds, `write_marker=False` so no marker is
touched) and record every frame that fails ANY alignment gate in EITHER style. Those frames, and every temporal triple
or pair that touches one, are never used for training or evaluation. Gates are
unchanged; only the unit of exclusion changes from session to frame.

A (session, style) with no ALIGNED marker is ADMITTED only if every error the
validator raised is a per-frame `align.*` gate failure (layout, header, frame
lines, frame count all clean) — so exclusion can only remove a bad frame, never
paper over a broken target set. The failures are also cross-checked against the
validator logs written when the targets were generated.
"""
from __future__ import annotations
import datetime as dt, json, re
from concurrent.futures import ProcessPoolExecutor
from data_pipeline.teacher.alignment import validate_alignment
from renderer.training.dataset import target_dir_for
from renderer.training.train_baseline import ROOT

STYLES = ("oil_impasto", "watercolor_bold_s055")
LOGS = {"oil_impasto": "eval/logs/game_teacher_chain.log",
        "watercolor_bold_s055": "eval/logs/game_teacher_watercolor_bold.log"}

def scan(args):
    sid, style = args
    r = validate_alignment(target_dir_for(sid, style), write_marker=False)
    bad, blocking = {}, []
    for f in r.errors:
        m = re.match(r"frame (\d+):", f.message)
        if f.code.startswith("align.") and m:
            bad.setdefault(int(m.group(1)), []).append(f.code)
        else:
            blocking.append(f"{f.code}: {f.message}")
    if not r.stats.get("target_set_complete"):
        blocking.append("target set incomplete")
    has_marker = (target_dir_for(sid, style) / "ALIGNED").is_file()
    return sid, style, bad, blocking, has_marker

def logged(style):
    """(session, frame) pairs the validator logged as failing, for the cross-check."""
    out, sid = set(), None
    for line in open(ROOT / LOGS[style], errors="replace"):
        m = re.search(r"alignment m3-alignment-1\s+style=(\S+)\s+session=(ds_\S+)", line)
        if m:
            sid = m.group(2) if m.group(1) == style else None
        e = re.search(r"\[ERROR\] align\.\S+: frame (\d+)", line)
        if e and sid:
            out.add((sid, int(e.group(1))))
    return out

def main() -> int:
    sessions = [l.strip() for l in open(ROOT / "configs/data/game_sessions_v1.txt") if l.strip()]
    jobs = [(s, st) for s in sessions for st in STYLES]
    with ProcessPoolExecutor(max_workers=24) as ex:
        res = list(ex.map(scan, jobs))
    excl, by_style = {}, {st: set() for st in STYLES}
    admitted = {st: [] for st in STYLES}
    refused = {st: {} for st in STYLES}
    for sid, style, bad, blocking, has_marker in res:
        if has_marker and bad:
            blocking.append("ALIGNED marker present but gates now fail — targets changed after validation?")
        if blocking:
            refused[style][sid] = blocking
            continue
        admitted[style].append(sid)
        for f, codes in bad.items():
            excl.setdefault(sid, {}).setdefault(str(f), {})[style] = codes
            by_style[style].add((sid, f))
    check = {st: {"recomputed": len(by_style[st]), "logged": len(logged(st)),
                  "identical": by_style[st] == logged(st)} for st in STYLES}
    rep = {"purpose": "Frames failing any alignment gate in either style; excluded with every triple/pair touching them (Decision 94).",
           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "styles": list(STYLES), "thresholds": "data_pipeline.teacher.metrics.Thresholds() defaults, as the validator",
           "n_sessions": len(sessions), "admitted": admitted, "refused": refused,
           "n_frames_excluded": sum(len(v) for v in excl.values()),
           "cross_check_vs_validator_logs": check, "excluded": excl}
    (ROOT / "configs/data/game_frame_exclusions_v1.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("n_frames_excluded", "cross_check_vs_validator_logs", "refused")}, indent=1))
    print({st: len(v) for st, v in admitted.items()}, "sessions admitted per style")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
