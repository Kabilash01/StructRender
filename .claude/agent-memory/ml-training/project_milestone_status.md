---
name: milestone-status
description: Which StructRender milestone the project is on and when model training actually starts
metadata:
  type: project
---

As of 2026-09-03: Milestones 0, 1, 2 complete (repo skeleton; structural export;
dataset recorder + validator). Working through KB §10 roadmap in order.

No neural-network training has happened yet and none happens until **Milestone
4** (CNN/U-Net baseline: "model trains / overfits a tiny set / recognizable
styled output / ONNX export"). Milestone 3 (offline teacher targets) comes
first and is data-prep, not training. StructFormer training is M5; temporal M6;
entity/EPAM M7-8; style refinement then optional adversarial after that.

**Why:** the user asked "how long will this training happen?" during the M2
validator task — there was no training running. Worth being explicit that
"training" in this repo is still several milestones out.

**How to apply:** when the user asks about training duration/timeline, clarify
which milestone's training they mean. Per KB Decision 19 training is staged and
resumes from the previous best checkpoint each stage; per Decision 20 paid GPU
budget is reserved for high-value runs (most iteration is local/small).
