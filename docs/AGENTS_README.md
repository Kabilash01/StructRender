# StructRender Claude Code Setup

## File layout
```
CLAUDE.md                          # your behavioral rules + project routing (repo root)
docs/StructRender_KB.md            # the full locked architecture spec (copy your KB here)
docs/AGENTS_README.md              # this file
.claude/agents/
  godot-engine.md                  # Godot/GDScript/GDExtension, structural export, entity lifecycle
  ml-training.md                   # PyTorch: StructFormer, EPAM, losses, curriculum training
  export-deploy.md                 # ONNX → TensorRT → C++ real-time deployment
  eval-harness.md                  # RCS metric, ablation ladder, verification scenarios
  spec-guardian.md                 # read-only reviewer, checks work against locked decisions
```

## How to use these day to day
- Start each session normally — CLAUDE.md loads automatically and gives Claude the project context and non-negotiables.
- For domain-specific work, either let Claude auto-delegate (its descriptions are written to route correctly) or invoke explicitly: `Use the ml-training agent to implement the EPAM update rule`.
- **Run spec-guardian before closing out any milestone**, not just at the end of the project: `Use the spec-guardian agent to check the entity token packing work against the KB`. It's read-only by design — it can't "helpfully" fix anything, only report.
- All five agents have `memory: project` enabled, so each builds up its own notes in `.claude/agent-memory/<name>/` over time (e.g. Godot quirks discovered, TensorRT export gotchas). Check these into version control — they're genuinely useful institutional knowledge for later sessions, and worth glancing at when you resume work after a break.

## Setting this up
1. Copy your finalized `StructRender_KB_v3.md` into `docs/StructRender_KB.md` in your repo (rename to that exact path so CLAUDE.md's reference resolves).
2. Copy `CLAUDE.md` to your repo root.
3. Copy the `.claude/agents/` folder as-is into your repo root.
4. Commit all of it — these are project-scoped (not `~/.claude/agents/`), so they should live in version control alongside the code they govern.

## Omarchy / Arch-specific practical notes

A few things worth sorting out early rather than discovering mid-Milestone-5, since Arch/Omarchy's setup path differs from the Ubuntu-based instructions most tutorials assume:

- **NVIDIA driver:** confirm `nvidia-smi` works before anything else. Omarchy ships Hyprland (Wayland), and NVIDIA+Wayland has historically been the rougher combination — if you hit compositor/driver friction, check Omarchy's own docs/community threads for the current recommended driver package first rather than generic Arch wiki steps, since Omarchy pins specific configurations.
- **CUDA + PyTorch:** install PyTorch via pip with the CUDA-matched wheel (`pip install torch --index-url https://download.pytorch.org/whl/cu1XX` matching your driver's CUDA version) rather than an Arch community CUDA package — this avoids version-mismatch headaches between system CUDA and what PyTorch expects.
- **TensorRT:** install via NVIDIA's official pip wheel (`pip install tensorrt`) rather than hunting for an AUR package — the pip route is more reliably kept in sync with current CUDA/driver versions and is what most current TensorRT documentation assumes.
- **Godot + GDExtension build toolchain:** Godot itself is straightforward (AUR `godot` or the official Linux binary). For GDExtension, you'll need `scons`, a C++ compiler (`base-devel` covers this), and the `godot-cpp` bindings as a submodule matching your Godot version exactly — a version mismatch here is a common source of confusing build errors, so pin the `godot-cpp` submodule to your installed Godot's version tag from the start.
- **Verify the whole chain early, not at Milestone 10.** Once `nvidia-smi`, a CUDA-enabled PyTorch, and a `scons`-built empty GDExtension module all work, you've cleared the environment risk — do this in Week 1 alongside repo setup, not when you actually reach the C++ integration milestone, since debugging a toolchain issue at that point competes with debugging your actual model.
