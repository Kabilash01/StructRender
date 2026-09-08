---
name: m5-export-probe
description: What the M5 pre-architecture ONNX/TensorRT attention probe settled, and the three findings that still need KB Decision entries
metadata:
  type: project
---

2026-09-08: M5 **step 2** (design-time export probe) is done — run *before* the
StructFormer architecture was trained, because KB §12's only flagged operator
risk is "custom attention/masking operation exports poorly" and a blocker there
changes the design.

**Nothing was blocked.** Every hazard §12 names reached a *built and executed*
TensorRT engine. Read `eval/reports/m5_export_probe.json` for the matrix — it
self-identifies (Decision 42 fields) and `renderer/structformer/tests/test_export_probe.py`
asserts the committed artifact still says what it is quoted as saying.

**Why:** the probe exists so that M5/M6/M7 architecture choices are made on
measured export behaviour instead of convention, and so M7's headline
entity-masked cross-attention is not discovered to be unexportable at M7.

**How to apply:**
- **Never quote a latency or diff from memory** — read the artifact. The probe
  runs at minimum width (embed 32, 1 head), so its latency numbers are for
  *variant-vs-variant* comparison only and are **not** evidence about §12's
  15-20 ms StructFormer budget. The report says so in `latency_note`; do not
  strip that caveat when citing it.
- Four decisions the probe settled by measurement, all reversible only with new
  measurement: manual `matmul/softmax/matmul` over `F.scaled_dot_product_attention`;
  shifted windows kept; `torch.roll` kept (it is not the hazard §12 assumed);
  relative-position index as a non-persistent buffer.
- **Three items were flagged for the user to write into the KB Decision Log and
  were deliberately NOT written by the agent** (the user writes Decision
  entries): the TensorRT-11 FP16 API change, the entity-mask NaN/gating spelling,
  and style conditioning entering the graph as a second input. Check whether
  those landed before assuming the KB covers them.
- See [[milestone-status]] and [[feedback-evidence-discipline]].
