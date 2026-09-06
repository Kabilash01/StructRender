---
name: milestone-status
description: Which StructRender milestone the project is on, and what the M4 baseline established as the reference point for M5
metadata:
  type: project
---

As of 2026-09-06: Milestones 0-4 complete, M4 re-verified after spec-guardian
review. Working through KB §10 in order.

- M0-M2: repo skeleton, structural export, dataset recorder + validator.
- M3: offline teacher targets (SD1.5 + ControlNet-depth, style `gouache`).
- **M4 (2026-09-05, revised 2026-09-06): CNN/U-Net baseline — ladder Model A.**
  First actual NN training in the repo. All four exit criteria verified. Rules
  pinned in **KB Decision 35** (7-channel input, inverse-depth normalization,
  contiguous-sequence holdout split with a guard band, pre-fixed overfit
  thresholds, ONNX opset 17 / fixed shapes).

Next is M5 (StructFormer baseline: CNN stem + window transformer encoder +
decoder + RGB head + style conditioning), whose exit criteria include a
**TensorRT benchmark at M5, not at the end**. Then M6 temporal, M7 entity,
M8 EPAM.

**Why:** this memory previously said "no training until M4" — that gate has now
been passed, so timeline questions have a different answer.

**How to apply:**
- When comparing a new model against the CNN baseline, read the current numbers
  from `eval/reports/m4_metrics.json` rather than quoting remembered ones.
- Three carry-forward obligations from Decision 35, easy to forget: (1) M4 trains
  on **L1 only**, so if M5 adds perceptual/other terms, Model A must be re-run
  under the same loss or the §11 comparison measures the objective, not the
  architecture; (2) **every metric on this session/style is range-compressed by
  a strong do-nothing anchor** — the unstyled engine frame already scores
  35.03 dB / SSIM 0.9899 / LPIPS 0.0425 against the teacher target. LPIPS is the
  *worst* of the three here (the model closes only 1.7 % of the
  do-nothing→perfect gap vs SSIM's 31 %), so never quote an image-quality number
  without its do-nothing anchor, and do not elevate LPIPS as the ladder headline
  — an earlier version of this memory and of the KB said to, and that was wrong.
  The real fix is a higher-contrast headline style; the actual headline result is
  RCS vs time-outside-view anyway; (3) the val split is also the model-selection
  set, so report it as `selection_set` and use the identical selection rule for
  every ladder model.
- Per Decision 19 training is staged, resuming from the previous best checkpoint
  each stage; per Decision 20 paid GPU budget is reserved for high-value runs.
  M4 trained locally on the RTX 5060 in ~19 min, so M5 iteration is likely local
  too.
