#!/usr/bin/env python3
"""Smoke tests for the causal residual steering controller (plumbing only).

Proves, for local full-precision Qwen3.5-9B with a layer-15 residual hook:
  1. zero-init controller is numerically identical to untouched Qwen
     (greedy + seeded sampling + teacher-forced logits);
  2. no Qwen parameter can receive gradients or change;
  3. a deliberately nonzero intervention measurably changes downstream
     logits and generation (the hook is live);
  4. the intervention at token t depends only on prefix state available at t
     (truncation + continuation-swap);
  5. controller checkpoint save/load reproduces behavior;
  6. intervention magnitude is bounded relative to the residual state.

Passing these tests establishes that the control surface works. It does NOT
establish that steering improves reasoning — the ACE hypothesis is unproven.

Run:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
      test_steering_controller.py -s -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steering_controller import (  # noqa: E402
    ALPHA, MODEL_PATH, STEER_LAYER, DEMO_PROMPT, ResidualSteering,
    SteeringController, build_prompt_ids, freeze_base_model, generate,
    teacher_forced_logits,
)

CONT_A = ("First, let me understand the problem. The farmer starts with 17 "
          "sheep. The phrase 'all but 9 run away' means 9 sheep remain. So "
          "the answer is 9. Let me verify: 17 minus the 8 that ran leaves 9.")
CONT_B = ("Okay, a completely different approach: think about what 'all but "
          "9' means. Every sheep except 9 ran away, so 9 stay put. The flock "
          "size of 17 is a distraction planted to trip up arithmetic.")

# Causality comparisons cross sequence lengths, where bf16 kernel tiling can
# reorder reductions for identical prefixes. Tolerances below are safety
# margins; measured values are printed and expected to be far smaller.
CAUSAL_REL_TOL = 0.05   # relative L2 on delta at the truncation point
CAUSAL_ABS_TOL = 2e-2   # abs elementwise on delta over the shared prefix


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    freeze_base_model(m)
    yield m
    del m
    torch.cuda.empty_cache()


def make_controller(seed: int, up_std: float = 0.5, gate_std: float = 1.0,
                    alpha: float = ALPHA) -> SteeringController:
    """Deterministically randomized (deliberately nonzero) controller."""
    g = torch.Generator().manual_seed(seed)
    ctrl = SteeringController(alpha=alpha)  # constructed fp32 on CPU
    with torch.no_grad():
        ctrl.up.weight.copy_(
            torch.randn(ctrl.up.weight.shape, generator=g) * up_std)
        ctrl.gate.weight.copy_(
            torch.randn(ctrl.gate.weight.shape, generator=g) * gate_std)
        ctrl.gate.bias.copy_(
            torch.randn(ctrl.gate.bias.shape, generator=g) * gate_std)
    return ctrl.to(device="cuda", dtype=torch.bfloat16)


# --- test 1: zero-init is an exact no-op -------------------------------------

def test_zero_init_controller_is_exact_noop(model, tok):
    ctrl = SteeringController().to(device="cuda", dtype=torch.bfloat16)

    # kernel determinism control: untouched Qwen vs itself
    ids_a, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=32)
    ids_b, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=32)
    assert ids_a[0].tolist() == ids_b[0].tolist(), "baseline not deterministic"

    # greedy generation: baseline vs zero-init steered
    ids_s, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=32,
                        controller=ctrl)
    assert ids_a[0].tolist() == ids_s[0].tolist()

    # seeded sampling: same seed must consume RNG identically
    samp_a, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=24,
                         do_sample=True, temperature=0.7, top_p=0.9, seed=42)
    samp_s, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=24,
                         controller=ctrl, do_sample=True, temperature=0.7,
                         top_p=0.9, seed=42)
    assert samp_a[0].tolist() == samp_s[0].tolist()

    # teacher-forced logits: bit-exact
    logits_a, _, _ = teacher_forced_logits(model, tok, DEMO_PROMPT, CONT_A)
    logits_s, _, _ = teacher_forced_logits(model, tok, DEMO_PROMPT, CONT_A,
                                           controller=ctrl)
    diff = (logits_a.float() - logits_s.float()).abs().max().item()
    print(f"\n[1] zero-init max |logit diff| = {diff} (exact equality required)")
    assert torch.equal(logits_a, logits_s)


# --- test 2: base model frozen ------------------------------------------------

def test_base_model_frozen(model, tok):
    assert all(not p.requires_grad for p in model.parameters())

    named = dict(model.named_parameters())
    watch = ["model.embed_tokens.weight",
             "model.layers.15.mlp.down_proj.weight",
             "model.layers.3.self_attn.q_proj.weight",
             "model.norm.weight", "lm_head.weight"]
    watch = [n for n in watch if n in named]
    assert len(watch) >= 4, f"expected param names missing: {watch}"
    fingerprints = {n: (named[n].detach().view(-1)[:256].clone(),
                        named[n].detach().float().sum().item())
                    for n in watch}

    ctrl = SteeringController().to(device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([build_prompt_ids(tok, DEMO_PROMPT)], device="cuda")
    with ResidualSteering(model, ctrl):
        logits = model(input_ids=ids).logits
        loss = logits[:, -1].float().sum()
        loss.backward()

    assert all(p.grad is None for p in model.parameters()), \
        "a Qwen parameter received a gradient"
    assert all(p.grad is not None for p in ctrl.parameters()), \
        "controller parameter missing gradient"
    up_grad = ctrl.up.weight.grad.abs().sum().item()
    print(f"\n[2] qwen grads: all None; controller up.weight |grad| sum = "
          f"{up_grad:.6f} (must be > 0)")
    assert up_grad > 0

    # step ONLY the controller; base fingerprints must be bit-identical
    with torch.no_grad():
        for p in ctrl.parameters():
            p.add_(-0.1 * p.grad)
    for n, (head, total) in fingerprints.items():
        p = named[n].detach()
        assert torch.equal(p.view(-1)[:256], head), f"{n} changed"
        assert p.float().sum().item() == total, f"{n} sum changed"
    ctrl.zero_grad()


# --- test 3: nonzero intervention is live --------------------------------------

def test_nonzero_intervention_changes_logits_and_generation(model, tok):
    ctrl = make_controller(seed=7)
    logits_a, _, _ = teacher_forced_logits(model, tok, DEMO_PROMPT, CONT_A)
    logits_s, _, recs = teacher_forced_logits(model, tok, DEMO_PROMPT, CONT_A,
                                              controller=ctrl, record=True)
    diff = (logits_a.float() - logits_s.float()).abs().max().item()
    dn_max = max(r["delta_norm"].max().item() for r in recs)
    print(f"\n[3] nonzero controller: max |logit diff| = {diff:.4f}, "
          f"max ||delta|| = {dn_max:.4f}")
    assert diff > 1e-3, "intervention had no measurable effect on logits"
    assert dn_max > 0

    base, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=48)
    # Greedy argmax over a 248k vocab is robust to small logit shifts, so the
    # generation check uses a stronger (still bounded) intervention.
    strong = make_controller(seed=8, up_std=1.0, alpha=0.5)
    steered, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=48,
                          controller=strong)
    a, s = base[0].tolist(), steered[0].tolist()
    assert a != s, "intervention had no effect on generation"
    first_div = next(i for i, (x, y) in enumerate(zip(a, s)) if x != y)
    print(f"[3] greedy generation first diverges at position {first_div} "
          f"(of {len(a)}; alpha=0.5 intervention)")


# --- test 4: intervention is prefix-causal --------------------------------------

def test_intervention_depends_only_on_prefix(model, tok):
    ctrl = make_controller(seed=11)
    _, ids_full, rec_full = teacher_forced_logits(
        model, tok, DEMO_PROMPT, CONT_A, controller=ctrl, record=True)
    delta_full = rec_full[0]["delta"][0]          # [T, H] on CPU
    T = len(ids_full)
    n_prompt = len(build_prompt_ids(tok, DEMO_PROMPT))
    cut = n_prompt + 20                            # inside the continuation
    assert cut < T

    # (a) truncation: prefix-only forward must reproduce delta at the cut
    ids_trunc = ids_full[:cut]
    with ResidualSteering(model, ctrl, record=True) as st, torch.no_grad():
        model(input_ids=torch.tensor([ids_trunc], device="cuda"))
    delta_trunc = st.records[0]["delta"][0]       # [cut, H]
    d_full_at_cut = delta_full[cut - 1]
    rel = ((delta_trunc[-1] - d_full_at_cut).norm()
           / d_full_at_cut.norm().clamp_min(1e-12)).item()
    print(f"\n[4a] truncation at {cut}/{T}: relative ||delta diff|| = {rel:.3e}")
    assert rel < CAUSAL_REL_TOL

    # (b) continuation swap: different futures must not change prefix deltas
    _, ids_b, rec_b = teacher_forced_logits(
        model, tok, DEMO_PROMPT, CONT_B, controller=ctrl, record=True)
    delta_b = rec_b[0]["delta"][0]
    abs_diff = (delta_b[:n_prompt] - delta_full[:n_prompt]).abs().max().item()
    print(f"[4b] continuation swap: max |delta diff| over {n_prompt} shared "
          f"prefix positions = {abs_diff:.3e}")
    assert abs_diff < CAUSAL_ABS_TOL


# --- test 5: save/load reproduces behavior --------------------------------------

def test_controller_save_load_roundtrip(model, tok, tmp_path):
    ctrl = make_controller(seed=21)
    ctrl.save(tmp_path / "ctrl",
              extra_meta={"layer_idx": STEER_LAYER, "model_path": MODEL_PATH})
    ctrl2, meta = SteeringController.load(tmp_path / "ctrl", device="cuda",
                                          dtype=torch.bfloat16)
    assert meta["layer_idx"] == STEER_LAYER
    assert meta["hidden_size"] == 4096 and meta["bottleneck"] == 512
    assert meta["alpha"] == ALPHA
    for k, v in ctrl.state_dict().items():
        assert torch.equal(v, ctrl2.state_dict()[k]), f"state mismatch: {k}"

    logits_1, _, _ = teacher_forced_logits(model, tok, DEMO_PROMPT, CONT_A,
                                           controller=ctrl)
    logits_2, _, _ = teacher_forced_logits(model, tok, DEMO_PROMPT, CONT_A,
                                           controller=ctrl2)
    diff = (logits_1.float() - logits_2.float()).abs().max().item()
    print(f"\n[5] save/load max |logit diff| = {diff} (exact equality)")
    assert torch.equal(logits_1, logits_2)

    gen_1, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=32,
                        controller=ctrl)
    gen_2, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=32,
                        controller=ctrl2)
    assert gen_1[0].tolist() == gen_2[0].tolist()


# --- test 6: intervention is bounded relative to the residual -------------------

def test_intervention_is_bounded(model, tok):
    for seed, up_std, alpha in [(1, 0.05, ALPHA), (2, 1.0, ALPHA),
                                (3, 1.0, 0.5)]:
        ctrl = make_controller(seed=seed, up_std=up_std, alpha=alpha)
        _, _, recs = teacher_forced_logits(model, tok, DEMO_PROMPT, CONT_A,
                                           controller=ctrl, record=True)
        dn = torch.cat([r["delta_norm"].reshape(-1) for r in recs])
        hn = torch.cat([r["h_norm"].reshape(-1) for r in recs])
        g = torch.cat([r["gate"].reshape(-1) for r in recs]).float()
        ratio = (dn / hn.clamp_min(1e-12)).max().item()
        n_sat = int(((g == 0.0) | (g == 1.0)).sum())
        print(f"\n[6] up_std={up_std} alpha={alpha}: max ||delta||/||h|| = "
              f"{ratio:.4f} (cap {alpha}), gate in [{g.min():.3f}, "
              f"{g.max():.3f}], {n_sat}/{g.numel()} saturated in bf16")
        # 2% slack absorbs bf16 rounding of norms/products
        assert ratio <= alpha * 1.02
        # sigmoid is mathematically in (0,1); bf16 rounds saturated tails to
        # exactly 0.0/1.0, so the checkable property is the closed interval
        assert g.min() >= 0.0 and g.max() <= 1.0

    ctrl0 = SteeringController().to(device="cuda", dtype=torch.bfloat16)
    _, _, recs0 = teacher_forced_logits(model, tok, DEMO_PROMPT, CONT_A,
                                        controller=ctrl0, record=True)
    dn0 = torch.cat([r["delta_norm"].reshape(-1) for r in recs0])
    print(f"[6] zero-init: max ||delta|| = {dn0.max().item()} (must be 0)")
    assert dn0.max().item() == 0.0
