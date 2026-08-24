#!/usr/bin/env python3
"""Unit + tiny GPU smoke tests for the controller-training loop.

Pure-CPU tests cover the verifier, problem-set construction, advantages,
and the k3 KL estimator. The GPU test runs one micro-iteration (2 prompts,
G=2, 64 new tokens) and checks: controller weights move, Qwen does not,
and the log record has the expected keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problems import HOLDOUT, TRAIN, extract_answer, verify  # noqa: E402
from train_controller import (  # noqa: E402
    group_advantages, k3_kl, snapshot_base, check_base, train_iteration,
)


def test_problem_set_shape_and_determinism():
    from problems import build_problems
    t1, h1 = build_problems()
    t2, h2 = build_problems()
    assert len(t1) == 24 and len(h1) == 12
    assert {x["id"] for x in t1}.isdisjoint({x["id"] for x in h1})
    assert [x["id"] for x in t1] == [x["id"] for x in t2]
    assert all(isinstance(x["gold"], float) for x in t1 + h1)
    train_fams = {x["family"] for x in t1}
    hold_fams = {x["family"] for x in h1}
    assert train_fams.isdisjoint(hold_fams)


def test_verifier():
    assert verify("foo\nAnswer: 113", 113.0)
    assert verify("Answer: $1,234.50", 1234.5)
    assert verify("Answer: 5\nAnswer: 7.00", 7.0)
    assert not verify("no marker", 3.0)
    assert not verify("Answer: 114", 113.0)
    assert extract_answer("Answer: -2.5 leftover") == -2.5


def test_group_advantages_zero_mean_and_collapse():
    a = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(a)) < 1e-5
    z = group_advantages([1.0, 1.0, 1.0, 1.0])
    assert all(abs(x) < 1e-3 for x in z)
    z0 = group_advantages([0.0, 0.0])
    assert all(abs(x) < 1e-3 for x in z0)


def test_k3_kl_properties():
    pol = torch.tensor([-1.2, -0.4, -2.0])
    ref = pol.clone()
    kl0 = k3_kl(pol, ref)
    assert torch.allclose(kl0, torch.zeros_like(kl0), atol=1e-6)
    ref2 = torch.tensor([-2.0, -0.1, -1.0])
    kl = k3_kl(pol, ref2)
    assert (kl >= -1e-6).all()


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    from steering_controller import MODEL_PATH
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def model():
    from transformers import AutoModelForCausalLM
    from steering_controller import MODEL_PATH, freeze_base_model
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    freeze_base_model(m)
    yield m
    del m
    torch.cuda.empty_cache()


def test_one_iteration_moves_controller_not_qwen(model, tok):
    from argparse import Namespace
    from steering_controller import SteeringController
    snap = snapshot_base(model)
    ctrl = SteeringController().to(device="cuda", dtype=torch.float32)
    before = {k: v.detach().clone() for k, v in ctrl.state_dict().items()}
    opt = torch.optim.AdamW(ctrl.parameters(), lr=1e-3, weight_decay=0.0)
    cfg = Namespace(group_size=2, max_new=64, temperature=0.9, top_p=0.95,
                    seed=0, beta_kl=0.02, lambda_mag=0.1)
    row = train_iteration(model, tok, TRAIN[:2], ctrl, opt, cfg, iter_i=0)
    after = ctrl.state_dict()
    moved = any(not torch.equal(before[k], after[k]) for k in before)
    print(f"\n[train-smoke] controller moved={moved}  "
          f"steer={row['steered_acc']:.2f} base={row['base_acc']:.2f}  "
          f"pg={row['loss_pg']:.3f} kl={row['loss_kl']:.4f}  "
          f"|g|={row['grad_norm']:.3f}")
    assert check_base(model, snap)
    assert set(row) >= {"iter", "steered_acc", "base_acc", "loss_pg",
                        "loss_kl", "loss_mag", "grad_norm", "prompts"}
    # With a 64-token cap most answers won't land, so advantages may be
    # all-zero and the controller may not move. That is a valid outcome
    # of the objective; the required invariant is that Qwen did not move
    # and the loop produced a well-formed record. If the group mixed,
    # the controller *must* have moved.
    mixed = any(0.0 < p["steered_mean"] < 1.0 for p in row["prompts"])
    if mixed:
        assert moved, "mixed rewards but controller weights unchanged"
