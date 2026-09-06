---
name: feedback-evidence-discipline
description: How this user wants measured claims, thresholds, and committed artifacts handled in StructRender training work
metadata:
  type: feedback
---

Every number the KB quotes must be reproducible from a committed artifact, and
no threshold may move to make a result pass.

**Why:** the M4 spec-guardian review (2026-09-06) found four instances of the
same failure mode, all in Decision 35: `test_overfit.py` overwrote
`eval/reports/m4_overfit.json` on every run, so the KB cited a destroyed run;
the overfit set was N=8 when KB §11 mandates 32-128; "recognizable styled
output" rested on a colour-histogram metric that *rises* under blur; and LPIPS
was recommended as the ladder headline without its do-nothing anchor ever being
measured (measured: it is the worst of the three). The user's brief was explicit:
"Do NOT relax a threshold to make 32 pass — if it fails, report that as a
finding; that is the honest outcome and I want to know."

**How to apply:**
- Verification runs are read-only w.r.t. committed evidence. Tests write reports
  to a tmp path; the committed artifact is refreshed only under an explicit
  opt-in (`STRUCTRENDER_REFRESH_REPORTS=1`). Never make refresh the default.
- Before quoting a "we beat X" number, measure the **do-nothing anchor** for the
  same metric. A metric whose range you have not measured cannot be recommended.
- When a gate metric is imported from an earlier milestone, re-ask what failure
  mode it was calibrated against. M3's gate 4 was tuned against a teacher that
  did nothing; a trained regressor's failure mode is blur, which that gate
  rewards.
- Calibrate a threshold against the *failure mode* (e.g. how the metric responds
  to a known Gaussian blur), not against the model's own score, and say plainly
  in the KB when a threshold was set after the model existed rather than
  pre-registered.
- Prose and assertion must agree. If the prose says "comparable" the assertion
  may not accept 0.5x.
- Label metrics by what they actually are: `selection_set`, not `held_out`, when
  the checkpoint was chosen on that same split.
- Corrections go **in place** in the KB Decision Log with a dated "Revised"
  header stating what was wrong — not appended as a new decision that leaves the
  wrong text standing. See [[milestone-status]] and [[feedback-validator-scope]].
