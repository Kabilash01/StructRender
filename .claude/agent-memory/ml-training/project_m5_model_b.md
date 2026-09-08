---
name: m5-model-b
description: M5 step 3 state — StructFormer (ladder Model B) is built and testable but untrained, with one unresolved capacity question and a measured GPU ceiling
metadata:
  type: project
---

2026-09-08: M5 **step 3** built ladder Model B (StructFormer at full training
width, the shared training entry point, the config and the tests). **Nothing was
trained** — the user launches training runs themselves, see
[[feedback-training-visible]] in their own memory.

**Two things here are not derivable from the code and are the reason this file
exists.**

1. **The specified Model B width is a capacity confound, and it is UNRESOLVED.**
   The brief pinned embed dim 96 / depths [2,2] / heads [3,6] with the stated
   rationale "chosen to land near Model A's parameter count so the ladder
   measures architecture and not capacity". Measured, that width is roughly a
   *sixth* of Model A, not near it. Both counts are pinned as
   `MODEL_A_PARAMETERS` / `MODEL_B_PARAMETERS` in
   `renderer/structformer/tests/test_m5_model.py` — **read them there, never from
   this file** (Decision 42: cited by artifact, never by value). The brief's own
   stop condition was one-sided (>1.5x only), so nothing halted; it was flagged
   back to the user instead, with the measured width that *does* reach parity.
   **Until this is settled, an M5-vs-M4 comparison measures architecture AND
   capacity together.** Do not report an M5 ladder result without saying which
   way it was resolved.

2. **The 8 GB RTX 5060 is the binding constraint on resolving it.** Measured at
   288x512, batch 8, bf16, with LPIPS resident: the configured width fits with
   room; the parameter-parity width **OOMs at batch 8** and needs batch 4. The
   memory hog is the RGB head, which convolves at `embed_dim` channels at full
   resolution. So parameter parity and batch parity cannot both be held on this
   card — and changing the batch size would itself become a second difference
   between the rungs, on top of the architecture. M6 temporal and M7 entity
   tokens add to this budget, not to spare.

**How to apply:**
- Re-measure before quoting any VRAM or ms/step figure; none is committed to an
  artifact yet, so nothing here is citable evidence. The measurement script was
  scratchpad-only and deliberately writes nothing.
- The §11 local pre-cloud overfit gate (N=32, four pre-registered thresholds
  imported unchanged from Model A) is **mandatory for Model B and has not been
  run**. It comes before the full run, not after.
- Decision 46's pre-registered M5 margin is already fixed and must not be
  revisited after seeing StructFormer's numbers.
- See [[milestone-status]], [[m5-export-probe]], [[feedback-evidence-discipline]].
