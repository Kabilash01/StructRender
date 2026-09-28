"""The ONE seed session the validator and teacher test suites run against.

Both suites used to pick "the newest ds_* under data_pipeline/recorder/". That
silently changed what they tested whenever a session was recorded: after the
gameplay dataset was recorded (Decision 87), the validator tests crashed (38
errors — a gameplay session has no reappearance sequences) and the teacher tests
kept passing against a gameplay session with no teacher targets. Decision 65
recorded the same failure for `discover_session()`. A test's subject must be
chosen, not inferred from directory order.

`ds_1788687350_337657ea` is the committed seed session (Decisions 38-40): the
seed recorder's variant 0 reproduces it exactly, it is VALIDATED, it carries a
temporal window plus ten reappearance sequences, and its `oil_impasto` targets
are ALIGNED. `$STRUCTRENDER_SEED_SESSION` still overrides it explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

SEED_SESSION_ID = "ds_1788687350_337657ea"
RECORDER_ROOT = Path(__file__).resolve().parents[1] / "data_pipeline" / "recorder"


def seed_session_path() -> Path | None:
    """The pinned seed session, or None if it is not present on this machine.

    Never falls back to another session: a missing seed must SKIP the tests with
    a reason, not quietly test something else.
    """
    env = os.environ.get("STRUCTRENDER_SEED_SESSION")
    p = Path(env) if env else RECORDER_ROOT / SEED_SESSION_ID
    return p if (p / "manifest.json").is_file() else None
