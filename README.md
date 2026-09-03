# StructRender

Engine-grounded neural game rendering. Godot owns world truth; a lightweight
CNN + window-transformer renderer (StructFormer) owns appearance; **EPAM**
(Engine-Keyed Persistent Appearance Memory) — the headline research
contribution — remembers a live entity's generated appearance and retrieves it
when that same entity re-enters view.

## Source of truth

The full architecture, locked decisions, module boundaries, milestone plan, and
evaluation protocol live in **[`docs/StructRender_KB.md`](docs/StructRender_KB.md)**.
Read the relevant section before working in any part of this repo. Behavioural
guidelines for contributors (human or agent) are in [`CLAUDE.md`](CLAUDE.md);
the Claude Code subagents are described in [`docs/AGENTS_README.md`](docs/AGENTS_README.md).

> The KB is referenced as `docs/StructRender_KB.md`, not the repo-root path
> shown in KB §8 — that is a deliberate, already-applied deviation so the
> `CLAUDE.md` reference resolves.

## Repository layout

Follows KB §8. Directories for later-milestone work exist as empty
`.gitkeep` placeholders; they are populated milestone by milestone per KB §10.

```
agent/          coding-agent workflow: specs, workflow docs, launch validators
configs/        YAML/TOML experiment configuration
game_project/   the Godot game (source of world truth)
data_pipeline/  recorder, structural exporters, entity lifecycle, validation
renderer/       StructFormer, baselines, temporal, EPAM, losses, training
inference/      ONNX export, TensorRT build, CUDA warp, C++ runtime
eval/           protocols, metrics (incl. RCS), plots, reports
tests/          unit / integration / e2e / fixtures
scripts/        one-command entry points (test, collect, train, eval, ...)
checkpoints/    model artifacts (git-ignored)
```

## Current status

**Milestone 0 — Repository + Minimal Godot Game.** See KB §10.

## Development

```bash
./scripts/test.sh        # run the full automated suite (one command; non-zero exit on failure)
```

Godot version is pinned in [`.godot-version`](.godot-version).
