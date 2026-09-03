# Coding-Agent Build-Test-Repair Workflow

This is the loop every generated or modified game/instrumentation change goes
through before it is accepted (KB §3A "Agent Authorship", Design Philosophy #8:
"Plausible code is not accepted until it is executed and validated").

It is intentionally lightweight: a documented procedure plus a few committed
scripts. There is no bespoke orchestration harness — the single command
`./scripts/test.sh` is what the loop runs.

## The loop

```
1. PLAN    — restate the target from agent/specifications/<spec>.md.
             State assumptions; if ambiguous, stop and ask (CLAUDE.md §1).
2. BUILD   — make the smallest change that moves toward the spec
             (CLAUDE.md §2, §3: surgical, minimal).
3. RUN     — launch the game headless:
                 godot --headless --path game_project
             It must reach `game_won` and exit 0 (sim_driver.gd drives it).
4. TEST    — ./scripts/test.sh
             Runs the launch/scene-load validator + all game_project/tests/*.
             Non-zero exit = failure.
5. INSPECT — on failure, read, in order:
                 a. ./scripts/test.sh stdout (assertion messages)
                 b. the Godot stderr/stdout it captured
                 c. user://logs/session_*.jsonl from the run
             Identify the single root cause. Do not guess-patch.
6. REPAIR  — fix the root cause only. No adjacent "improvements"
             (CLAUDE.md §3).
7. RERUN   — back to step 3. Repeat until TEST is green.
8. ACCEPT  — only when: game launches, reaches win state, exits 0, and the
             full suite passes. Then hand to spec-guardian before closing the
             milestone (docs/AGENTS_README.md).
```

## Entry points

| Command | Purpose |
|---|---|
| `./scripts/check_layout.sh` | Assert repo layout matches KB §8. |
| `godot --headless --path game_project` | Run the game end-to-end, no display. |
| `./scripts/test.sh` | Full automated suite; the one command M0 exit criterion 2 requires. |
| `agent/validators/check_launch.sh` | Just the launch / scene-load gate (KB §3A step 4), called by `test.sh`. |

## Which agent does what

- **godot-engine subagent** — all `game_project/` work, the test runner, the
  validators, and performing repairs in step 6.
- **spec-guardian subagent** — read-only review at step 8, before a milestone
  is declared complete. It cannot fix anything, only report deviations from the
  KB's locked decisions.

## Rules carried from CLAUDE.md / the KB

- Fail loudly offline: a missing scene, parse error, or broken assertion must
  make `test.sh` exit non-zero — never swallow it (KB §10 "Fail loudly
  offline; degrade gracefully online").
- Every changed line traces to the spec or to a test failure. If it doesn't,
  it doesn't belong in the diff.
- If a locked decision seems wrong while implementing, stop and add a dated
  entry to the KB Decision Log — do not quietly deviate (CLAUDE.md, project
  section).
