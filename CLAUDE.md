# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: StructRender

This repository implements StructRender. The full architecture, component boundaries, locked decisions, milestone plan, and evaluation protocol are the single source of truth in **`docs/StructRender_KB.md`** — read the relevant section before starting any work here, don't rely on summary/memory of it.

**Non-negotiable, from the KB's Decision Log — do not silently "fix" these, even if they look like reasonable engineering improvements in isolation:**
- Godot owns all physics/logic/entity identity. Nothing in the renderer, EPAM, or any training code may become a source of gameplay truth.
- EPAM (Engine-Keyed Persistent Appearance Memory) is the protected headline novelty. If schedule or compute is tight, cut other stretch features before touching EPAM's scope.
- EPAM is session-local only. No cross-session persistence, no zero-shot identity generalization — do not add these even as a "quick win" without flagging it first.
- GAN/adversarial loss is optional supervision, not the primary renderer. One-step DiT and runtime diffusion are explicitly out of scope for the current milestones.
- Entity IDs are addresses (masks, memory keys), never raw numeric input channels to the network.
- `K_max` entity-token overflow is resolved by ranking visible entities by `pixel_count`, never by arbitrary/spawn order.

**If you hit a case the KB doesn't cover, or a locked decision seems wrong once you're implementing it: stop and flag it — add a new dated entry to the KB's Decision Log explaining the deviation and why, rather than quietly deviating.** This is the same discipline as Section 1 above ("don't hide confusion, ask") applied specifically to this project's architecture.

**Work milestone by milestone**, per the KB's Section 10 roadmap. Before starting a milestone, restate its exit criteria from the KB. Before considering it done, verify each exit criterion explicitly rather than declaring success because the code compiles or runs once.

**Subagents are configured for this project** in `.claude/agents/` — see `docs/AGENTS_README.md` for what each one is for. Delegate to the matching subagent for its domain rather than doing everything in the main conversation; keeps context focused per the KB's own module boundaries.
