---
name: m6-model-d
description: M6 step 4 state — Model D (temporal StructFormer) is built and testable but untrained, plus the design decisions and confounds that are not derivable from the code
metadata:
  type: project
---

2026-09-10: M6 **step 4** built ladder **Model D** (temporal StructFormer), its
temporal loss, its pair/triple datasets, its config and its tests. **Nothing was
trained** — the user launches training runs themselves, see
[[feedback-training-visible]] in their own memory.

**Four things here are judgement calls, not code facts, and are the reason this
file exists.**

1. **The "autoregressive at inference" framing in the brief does not survive
   contact with the architecture, and the resolution is the whole design.** The
   history fed to the fusion is `encode(S_{t-1})` — encoder features of the
   previous frame's **structural G-buffer**, never of a rendered or teacher
   image. The engine supplies `S_{t-1}` at inference exactly as the dataset does
   at training, so this is *not* teacher forcing with an exposure gap: both sides
   compute the identical function. The price is that Model D is **depth-1
   temporal** (a 2-frame reprojection), not an unbounded-recurrence IIR filter,
   so it cannot accumulate stability over a long shot. If someone later asks why
   the M6 result is not a TAA-like accumulator, this is why.

2. **The training data unit is a TRIPLE `(t-2, t-1, t)`, not the pair the brief
   asked for**, and the reason is the loss, not the model. `L_temp` needs the
   model's own `pred_{t-1}`, and that reference must be rendered *with frame
   t-1's own history* or the loss trains the model to make its history-informed
   output match its history-free output — i.e. to ignore history. Marginal cost
   is one encoder pass. `pred_{t-1}` is computed under `no_grad`, which also
   rules out the degenerate "corrupt the past to match the present" solution.

3. **Two confounds against Model C, both real, both stated in the config rather
   than hidden.** (a) The fusion conv adds parameters — inside Decision 47's
   two-sided band but *not* zero, so a D-vs-C gate result must say so; the
   measured counts are pinned in
   `renderer/temporal/tests/test_m6_model_d.py`, read them there
   (Decision 42: cited by artifact, never by value). (b) `train.init_from`
   resumes from Model C's best checkpoint per Decision 19's curriculum rule,
   which gives D a second full training budget on top of C's. The control that
   would separate them does not exist.

4. **`init_from` buys far less than it looks like it does, MEASURED.** Resumed
   from Model C the first batches sit at C's converged L1; after two optimiser
   steps at the config's lr with no warmup it is roughly an order of magnitude
   worse. AdamW's first bias-corrected step moves every parameter by ~lr, and the
   schedule is Model C's *deliberately* (a warmup would be a second difference,
   and it is M5 attempt 2's change, which came out worse there). So this is a
   warm **restart**, not a fine-tune. The three ways out — warmup, a lower
   fine-tune lr, `init_from: null` — each add a difference against Model C, and
   the choice was left to the user.

**How to apply:**
- **Do not re-derive the M6 gate.** It is pre-registered in
  `eval/reports/m6_temporal_baseline.json -> m6_gate.pre_registered_threshold`
  (KB Decision 55) and must not be recomputed or adjusted after Model D exists.
  The temporal loss weight was deliberately **not** tuned against it.
- §11's local pre-cloud overfit gate is **mandatory and has not been run** for
  Model D. Its frame selection had to change (evenly-spread frames yield zero
  triples, so it uses contiguous runs) which makes the set *easier*; no threshold
  was moved to compensate, and that asymmetry must be repeated wherever the
  result is quoted.
- Training runs through `renderer/training/train_temporal.py`, a second entry
  point. Decision 35 warns that a forked trainer is how rungs drift apart, so it
  **imports** the split/objective/report/finite-loss helpers from
  `train_baseline` rather than copying them, and `test_m6_config.py` compares the
  configs field by field. Keep it that way.
- Next step (M6 step 5) is the gate evaluation, and it needs work that does not
  exist yet: `train_baseline.load_checkpoint` cannot load a
  `temporal_structformer` checkpoint, and `eval/protocols/run_temporal_baseline.py`
  renders each frame independently — Model D needs its previous frame's encoder
  features.
- Re-measure before quoting any VRAM or ms/step figure; none is committed to an
  artifact.
- See [[milestone-status]], [[m5-model-b]], [[feedback-evidence-discipline]].
