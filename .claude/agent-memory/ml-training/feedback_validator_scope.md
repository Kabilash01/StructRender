---
name: feedback-validator-scope
description: How the M2 validator (and data-pipeline checkers generally) should be scoped
metadata:
  type: feedback
---

Data-pipeline validators in this repo report only — they never repair,
regenerate, or mutate frames/manifests. The single file a validator may write is
its success marker (`VALIDATED`).

**Why:** KB error-handling philosophy is "explicit over silent, no auto-repair";
a validated dataset is immutable (KB §5). The task brief for the M2 validator
called this out as non-negotiable.

**How to apply:**
- Prefer strict thresholds over fractional/lenient ones when the reference data
  supports it (e.g. partial-dropout tolerance 0 — a fractional threshold would
  mask exactly the small dropped patches the exporter guard can't see).
- Recompute "derived" manifest fields from the raw frame data and compare, never
  trust the asserted value (reappearance `unseen_frames/seconds`, boundary
  frames, generations).
- Don't add cache/eviction/LRU logic or a "clean up" pass.
- Any new acceptance rule gets a dated KB Decision Log entry before it ships.
- Build corrupted-session test fixtures programmatically (copy real session,
  mutate one thing); never commit a corrupted fixture; skip cleanly when the
  gitignored seed data is absent.
