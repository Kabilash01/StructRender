"""M4 exit criterion 2 — "model overfits a tiny set".

``N = 32`` is not a free choice: KB §11's *Local Pre-Cloud Training Gate* item 4
mandates "overfit 32-128 frames" and calls the gate mandatory. 32 is the floor of
that range.

The thresholds live in ``configs/training/m4_unet_baseline.json`` -> ``overfit``
and were fixed **before** any training run, so the verdict can never become a
hindsight judgement call. If the model cannot hit them that is a finding to
report, not a threshold to relax.

Slow (~2000 optimiser steps). Skipped without CUDA — this is a GPU-budget check,
not a CPU one, and a silent 40-minute CPU run is worse than an explicit skip.

Writes its report to a tmp path; pass ``STRUCTRENDER_REFRESH_REPORTS=1`` to
refresh the committed ``eval/reports/m4_overfit.json`` instead.
"""

from __future__ import annotations

import json

import pytest
import torch

from renderer.training.train_baseline import run_overfit


@pytest.fixture(scope="module")
def result(cfg, session, target_dir, device, report_sink):
    if device.type != "cuda":
        pytest.skip("overfit check needs CUDA (2000 steps at 288x512)")
    if torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3:
        pytest.skip("overfit check needs >= 4 GB VRAM")
    res = run_overfit(cfg, device)
    out = report_sink("m4_overfit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def test_overfit_uses_the_declared_protocol(cfg, result):
    """N = 32 is KB §11's mandated floor; the LR is pinned in the overfit block
    so that retuning ``train.lr`` cannot silently move this exit criterion."""
    o = cfg["overfit"]
    assert result["n_frames"] == o["n_frames"] == 32
    assert result["max_steps"] == o["max_steps"] == 2000
    assert result["batch_size"] == o["batch_size"] == 2
    assert result["seed"] == o["seed"]
    assert "lr" in o, "overfit lr must be pinned here, not inherited from train.lr"
    assert result["lr"] == o["lr"]
    assert len(set(result["frame_ids"])) == 32


def test_overfit_psnr_at_least_30_db(result):
    th = result["thresholds"]["min_psnr_db"]
    assert result["psnr_db"] >= th, f"PSNR {result['psnr_db']:.2f} dB < {th} dB"


def test_overfit_final_l1_at_most_0_02(result):
    th = result["thresholds"]["max_final_l1"]
    assert result["final_l1"] <= th, f"final L1 {result['final_l1']:.4f} > {th}"


def test_overfit_loss_dropped_at_least_10x(result):
    """Sanity check, NOT an independent gate. The head is sigmoid-initialised
    (output ~0.5) against dark targets, so step-0 L1 is large by construction and
    a 10x drop is close to vacuous. PSNR / final L1 / SSIM carry the verdict."""
    th = result["thresholds"]["min_loss_drop_factor"]
    assert result["loss_drop_factor"] >= th, (
        f"L1 fell only {result['loss_drop_factor']:.2f}x "
        f"(step0 {result['step0_l1']:.4f} -> final {result['final_l1']:.4f}), need {th}x"
    )


def test_overfit_ssim_at_least_0_95(result):
    th = result["thresholds"]["min_ssim"]
    assert result["ssim"] >= th, f"SSIM {result['ssim']:.4f} < {th}"
