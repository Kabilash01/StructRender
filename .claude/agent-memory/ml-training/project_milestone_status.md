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

**M5 is CLOSED with a NEGATIVE RESULT** (2026-09-08): StructFormer — ladder
**Model C**, relabelled from "B" by Decision 50 — meets three of four exit
criteria (overfit gate, ONNX export, a built-and-executed TensorRT engine) and
**fails "better or competitive against the CNN baseline"** on both attempts at
parameter parity. Decisions 49/51 scope that failure hard: one implementation,
one recipe, one scene, one style, with unmeasured optimiser-exposure differences
and unmeasured StructFormer variance — *not* a general result about window
attention. Background: [[m5-export-probe]], [[m5-model-b]].

**M6 is in progress.** Steps 0-3 (warp, calibrated confidence mask,
pre-registered temporal gate) landed as Decision 55; step 4 built **Model D**,
untrained — see [[m6-model-d]]. §11's headline RCS metric was also implemented
and measured out of band (Decisions 53/54) and found **no degradation with time
unseen**, which removes the slope EPAM was expected to beat and makes M6 the only
rung that could produce one. Then M7 entity, M8 EPAM.

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
