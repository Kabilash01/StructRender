---
name: milestone-status
description: Which StructRender milestone the project is on, and what the M4 baseline established as the reference point for M5
metadata:
  type: project
---

As of 2026-09-06: Milestones 0-4 complete. M4 was **re-baselined** after the
scene enrichment (KB Decisions 38-41), not merely re-verified. Working through
KB §10 in order.

- M0-M2: repo skeleton, structural export, dataset recorder + validator.
- M3: offline teacher targets (SD1.5 + ControlNet-depth). The committed style is
  **`oil_impasto`**; `gouache` is **retired** (Decision 40) — config and target
  set deleted. `oil_impasto` is explicitly *not* the deferred style-B, which
  stays an open M8 decision.
- **M4 (2026-09-05, re-baselined 2026-09-06): CNN/U-Net baseline — ladder Model A.**
  All four exit criteria verified on session `ds_1788687350_337657ea`. Rules
  pinned in **KB Decision 35** (7-channel input, inverse-depth normalization,
  contiguous-sequence holdout split with a guard band, pre-fixed overfit
  thresholds, ONNX opset 17 / fixed shapes) as amended by Decisions 41-42.

**M5 is in progress** (StructFormer baseline: CNN stem + window transformer
encoder + decoder + RGB head + style conditioning), whose exit criteria include a
**TensorRT benchmark at M5, not at the end**. Step 2 — the pre-architecture
ONNX/TensorRT attention probe — is complete; see [[m5-export-probe]]. Still open
at M5: training, and "better or competitive output against the CNN baseline".
Then M6 temporal, M7 entity, M8 EPAM.

**Why:** an earlier version of this memory named the retired style, the retired
objective and the retired scene's anchors — every one of which would have sent
an M5 run at data that no longer exists.

**How to apply:**
- **Never quote a number from this file.** Read `eval/reports/m4_metrics.json`
  for the current ladder numbers and its `session_id` / `style_id` / `objective`
  fields for what produced them. Deliberately no figures here (Decision 42:
  a quantity that drifts run-to-run is cited by threshold and artifact, never
  by value). Anything you find quoted elsewhere against session
  `ds_1788624547_cf0ceeda` or style `gouache` is retired-scene and not
  comparable.
- Four carry-forward obligations, easy to forget:
  1. **The objective is `1.0*L1 + 0.5*LPIPS(alex)`**, mandated for *every* §11
     ladder model (Decision 41), not L1. Training raises rather than falling
     back if `lpips` is missing — do not "fix" that raise.
  2. Because LPIPS is now the training objective, it is **not an independent
     metric for any rung**. §11's independent set is SSIM, PSNR, the sharpness
     ratio and the four structural gates.
  3. **Never quote an image-quality number without its do-nothing anchor**
     (Decision 36) — the unstyled engine frame's own score against the teacher
     target. It bounds every metric's usable range, and on `sharpness_ratio`
     the anchor currently *beats* the model (Decision 41/44).
  4. The val split is also the model-selection set, so report it as
     `selection_set`, never `held_out`, and use the identical selection rule for
     every ladder model.
- Per Decision 19 training is staged, resuming from the previous best checkpoint
  each stage; per Decision 20 paid GPU budget is reserved for high-value runs.
  M4 trained locally on the RTX 5060 in ~20 min, so M5 iteration is likely local
  too.
- **The user requires training runs to be visible live** — always surface a
  `tail -f` on a named log path, never bury a run in a background job.
