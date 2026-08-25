#!/usr/bin/env python3
"""Causal residual steering controller for local full-precision Qwen3.5-9B.

Plumbing stage of the rebuilt ACE experiment: a small trainable controller
attached to the residual stream at ONE intermediate decoder layer. The base
model stays completely frozen; the controller is separately saveable/loadable.

Design
------
Hook point: output of ``model.model.layers[STEER_LAYER]`` (the residual stream
after that block). This is the only architecture-agnostic interception point:
Qwen3.5-9B is a hybrid (24 gated-delta-net ``linear_attention`` blocks + 8
``full_attention`` blocks at indices 3,7,...,31). Linear blocks carry a
fixed-size recurrent state instead of a KV cache, so any attention-internal
hook would need two implementations. The residual stream convention is shared
by both block types, and a layer-output hook sits below all cache machinery.

Layer choice: 15 (0-indexed of 32) — the mid-depth ``full_attention`` block.
It is a global prefix-mixing point (linear-attention state has limited recall;
full-attention blocks are where the whole prefix is re-integrated), and 16
blocks remain downstream to propagate an intervention to the logits.

Controller: normalized bottleneck adapter with a scalar per-token gate,

    z     = SiLU(down(LayerNorm(h_t)))                    # 4096 -> bottleneck
    d     = tanh(up(z))                                   # bottleneck -> 4096, |d_i| < 1
    scale = alpha * ||h_t|| / sqrt(4096)
    g_t   = sigmoid(gate(z))                              # in (0, 1)
    h'_t  = h_t + g_t * scale * d

Properties:
  * ``up`` and ``gate`` are zero-initialized -> d == 0 exactly -> the initial
    controller is a bit-exact no-op (bf16 zero matmul yields exact zeros).
  * Bounded: ||h'_t - h_t|| < alpha * ||h_t|| since ||tanh(d)|| < sqrt(4096)
    and g_t < 1. The bound is *relative* to the residual state, which matters:
    measured mean ||h|| grows from ~7 (layer 0) to ~48 (layer 15) to ~263
    (layer 31), so any absolute bound would be meaningless across depths.
  * Causal: the hook fires once per position during prefill ([B,T,H]) and once
    per generated token during cached decode ([B,1,H]). The controller is a
    pointwise function of the current residual state, which is itself computed
    from tokens <= t only (causal token mixers). No future tokens, completed
    trajectories, answers, or hindsight can enter the intervention.

Scope: plumbing + validation only. No training, corpus generation, RL/GRPO,
rewards, or branching here. Passing smoke tests establish that the control
surface works — NOT that steering improves reasoning (the ACE hypothesis
remains unproven).

Run the smoke demo:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m mlfactory.experiments.ace.core.steering_controller
"""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file, save_file

MODEL_PATH = "/home/admin/models/hf/Qwen3.5-9B"

# Stopping: config.json sets eos=248044 (<|endoftext|>), but chat turns end
# with <|im_end|> (248046) and the model ships no generation_config.json, so
# generate() would never stop at turn end without an explicit list.
STOP_TOKEN_IDS = [248044, 248046]
PAD_TOKEN_ID = 248044  # <|endoftext|> — distinct from <|im_end|>

# 0-indexed of 32 blocks; full_attention blocks are 3,7,11,15,19,23,27,31.
STEER_LAYER = 15
HIDDEN_SIZE = 4096
BOTTLENECK = 512
ALPHA = 0.1  # hard cap: ||intervention|| < ALPHA * ||residual||


class SteeringController(nn.Module):
    """Bottleneck adapter producing a bounded, gated residual intervention.

    Zero-initialized output projection and gate -> exact no-op at init.
    Parameter count with defaults: 4096*512+512 (down) + 512*4096 (up)
    + 512+1 (gate) = 4,195,329.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE,
                 bottleneck: int = BOTTLENECK, alpha: float = ALPHA):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck = bottleneck
        self.alpha = alpha
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        self.gate = nn.Linear(bottleneck, 1)
        self.act = nn.SiLU()
        # Zero-init the output path: initial controller is an exact no-op.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def intervention(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (delta, gate) for residual states h of shape [..., H].

        ||delta|| < alpha * ||h|| componentwise-bound via elementwise tanh.
        Compute happens in the controller's parameter dtype (bf16 controller
        -> identical to h; fp32 controller -> h upcast, delta downcast back,
        which keeps zero-init bit-exact in both cases: cast(0) == 0).
        """
        dtype = self.down.weight.dtype
        hc = h.to(dtype) if h.dtype != dtype else h
        z = self.act(self.down(F.layer_norm(hc, (self.hidden_size,))))
        d = torch.tanh(self.up(z))                          # [..., H], |d_i| < 1
        scale = (self.alpha * hc.norm(dim=-1, keepdim=True)
                 / math.sqrt(self.hidden_size))
        g = torch.sigmoid(self.gate(z))                     # [..., 1] in (0, 1)
        delta = g * scale * d
        return delta.to(h.dtype), g.to(h.dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta, _ = self.intervention(h)
        return h + delta

    # -- checkpointing, fully separate from Qwen weights ---------------------
    def save(self, directory: str | Path,
             extra_meta: dict | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous()
             for k, v in self.state_dict().items()},
            str(directory / "controller.safetensors"),
        )
        meta = {"hidden_size": self.hidden_size,
                "bottleneck": self.bottleneck,
                "alpha": self.alpha,
                "dtype": str(next(self.parameters()).dtype)}
        if extra_meta:
            meta.update(extra_meta)
        (directory / "controller.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path, device: str = "cuda",
             dtype: torch.dtype = torch.bfloat16):
        """Load a saved controller. Returns (controller, meta dict)."""
        directory = Path(directory)
        meta = json.loads((directory / "controller.json").read_text())
        ctrl = cls(hidden_size=meta["hidden_size"],
                   bottleneck=meta["bottleneck"], alpha=meta["alpha"])
        ctrl.load_state_dict(load_file(str(directory / "controller.safetensors")))
        return ctrl.to(device=device, dtype=dtype), meta


class ResidualSteering:
    """Context manager attaching a SteeringController to one layer's output.

    The hook fires on every forward through ``model.model.layers[layer_idx]``:
    once per position during prefill and once per generated token during
    cached decode. With ``record=True``, per-call (delta, gate, norms) are
    stashed on CPU for inspection — intended for short test sequences only.
    With ``collect=True``, grad-carrying per-call mean relative intervention
    norms are kept in ``self.collected`` (for the magnitude regularizer).
    """

    def __init__(self, model, controller: SteeringController,
                 layer_idx: int = STEER_LAYER, record: bool = False,
                 collect: bool = False):
        self.model = model
        self.controller = controller
        self.layer_idx = layer_idx
        self.record = record
        self.collect = collect
        self.records: list[dict] = []
        self.collected: list[torch.Tensor] = []
        self._handle = None

    def _hook(self, module, args, output):
        delta, g = self.controller.intervention(output)
        if self.collect:
            rel = (delta.float().norm(dim=-1)
                   / output.float().norm(dim=-1).clamp_min(1e-12))
            self.collected.append(rel.mean())
        if self.record:
            self.records.append({
                "delta": delta.detach().cpu(),
                "gate": g.detach().cpu(),
                "h_norm": output.detach().float().norm(dim=-1).cpu(),
                "delta_norm": delta.detach().float().norm(dim=-1).cpu(),
            })
        return output + delta

    def __enter__(self) -> "ResidualSteering":
        layer = self.model.model.layers[self.layer_idx]
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def freeze_base_model(model):
    """Disable gradients on every base-model parameter (controller excluded)."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_prompt_ids(tok, prompt: str, enable_thinking: bool = True) -> list[int]:
    """Chat-templated user prompt, generation-ready (same pattern as probes).

    ``enable_thinking=True`` (default) opens a live ``<think>`` block — the
    model's native reasoning mode. Training uses ``False`` so the template
    closes thinking immediately; otherwise completions rarely terminate
    within a few hundred tokens and terminal-correctness rewards vanish.
    """
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True,
        enable_thinking=enable_thinking)
    return list(enc.input_ids if hasattr(enc, "input_ids") else enc)


def generate(model, tok, prompt: str, max_new_tokens: int = 32,
             controller: SteeringController | None = None,
             layer_idx: int = STEER_LAYER, record: bool = False,
             do_sample: bool = False, temperature: float = 1.0,
             top_p: float = 1.0, seed: int | None = None,
             enable_thinking: bool = True):
    """Generation path with optional steering. Returns (ids, records).

    With ``do_sample=False`` this is greedy and deterministic. With sampling,
    pass a fixed ``seed`` for reproducibility.
    """
    ids = build_prompt_ids(tok, prompt, enable_thinking=enable_thinking)
    x = torch.tensor([ids], device=model.device)
    mask = torch.ones_like(x)
    if seed is not None:
        torch.manual_seed(seed)
    ctx = (ResidualSteering(model, controller, layer_idx, record)
           if controller is not None else nullcontext())
    kwargs = dict(input_ids=x, attention_mask=mask,
                  max_new_tokens=max_new_tokens, do_sample=do_sample,
                  eos_token_id=STOP_TOKEN_IDS, pad_token_id=PAD_TOKEN_ID)
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p)
    with ctx as active:
        out = model.generate(**kwargs)
    records = active.records if controller is not None else []
    return out, records


def generate_batch(model, tok, prompt: str, n: int,
                   max_new_tokens: int = 384,
                   controller: SteeringController | None = None,
                   layer_idx: int = STEER_LAYER, record: bool = False,
                   do_sample: bool = True, temperature: float = 0.9,
                   top_p: float = 0.95, seed: int | None = None,
                   enable_thinking: bool = True):
    """Batched rollouts of ONE prompt (identical length -> no left padding).

    Returns (seqs, records) where seqs[i] is the full id list of rollout i
    trimmed right after its first EOS at/after the prompt (or untrimmed if
    the rollout never emitted EOS). Deterministic given ``seed``.
    """
    ids = build_prompt_ids(tok, prompt, enable_thinking=enable_thinking)
    x = torch.tensor([ids] * n, device=model.device)
    mask = torch.ones_like(x)
    if seed is not None:
        torch.manual_seed(seed)
    ctx = (ResidualSteering(model, controller, layer_idx, record)
           if controller is not None else nullcontext())
    kwargs = dict(input_ids=x, attention_mask=mask,
                  max_new_tokens=max_new_tokens, do_sample=do_sample,
                  eos_token_id=STOP_TOKEN_IDS, pad_token_id=PAD_TOKEN_ID)
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p)
    with ctx as active:
        out = model.generate(**kwargs)
    records = active.records if controller is not None else []
    stop = set(STOP_TOKEN_IDS)
    seqs = []
    for row in out.tolist():
        end = len(row)
        for i in range(len(ids), len(row)):
            if row[i] in stop:
                end = i + 1          # include the stopping token
                break
        seqs.append(row[:end])
    return seqs, records


def teacher_forced_logits(model, tok, prompt: str, continuation: str,
                          controller: SteeringController | None = None,
                          layer_idx: int = STEER_LAYER, record: bool = False):
    """One forward over prompt+continuation. Returns (logits, ids, records)."""
    ids = (build_prompt_ids(tok, prompt)
           + tok(continuation, add_special_tokens=False).input_ids)
    x = torch.tensor([ids], device=model.device)
    ctx = (ResidualSteering(model, controller, layer_idx, record)
           if controller is not None else nullcontext())
    with ctx as active, torch.no_grad():
        logits = model(input_ids=x).logits
    records = active.records if controller is not None else []
    return logits, ids, records


DEMO_PROMPT = ("A farmer has 17 sheep. All but 9 run away. "
               "How many sheep are left? Think step by step, briefly.")


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    freeze_base_model(model)

    ctrl = SteeringController().to(device=model.device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in ctrl.parameters())
    print(f"controller params: {n_params:,} "
          f"(base model frozen: {sum(p.numel() for p in model.parameters()):,})")

    base, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=32)
    steered, _ = generate(model, tok, DEMO_PROMPT, max_new_tokens=32,
                          controller=ctrl)
    identical = base[0].tolist() == steered[0].tolist()
    print(f"zero-init steered generation identical to baseline: {identical}")
    print("\n--- baseline (greedy) ---")
    print(tok.decode(base[0][len(build_prompt_ids(tok, DEMO_PROMPT)):],
                     skip_special_tokens=True))
    print("--- zero-init steered (greedy) ---")
    print(tok.decode(steered[0][len(build_prompt_ids(tok, DEMO_PROMPT)):],
                     skip_special_tokens=True))

    # Deliberately nonzero controller: prove the hook is live. Greedy argmax
    # is robust to small logit shifts, so use a stronger (still bounded)
    # intervention for the visible generation flip.
    strong = SteeringController(alpha=0.5).to(device=model.device,
                                              dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        strong.up.weight.copy_(
            torch.randn(strong.up.weight.shape, generator=g) * 1.0)
        strong.gate.weight.copy_(
            torch.randn(strong.gate.weight.shape, generator=g) * 1.0)
        strong.gate.bias.copy_(
            torch.randn(strong.gate.bias.shape, generator=g) * 1.0)
    perturbed, recs = generate(model, tok, DEMO_PROMPT, max_new_tokens=32,
                               controller=strong, record=True)
    diverges = base[0].tolist() != perturbed[0].tolist()
    print(f"\nnonzero controller (alpha=0.5) changes generation: {diverges}")
    if recs:
        dn = torch.cat([r["delta_norm"].reshape(-1) for r in recs])
        hn = torch.cat([r["h_norm"].reshape(-1) for r in recs])
        print(f"||delta|| max={dn.max():.4f}  "
              f"bound 0.5*||h|| min={(0.5 * hn).min():.4f}  "
              f"max ratio={(dn / hn).max():.4f} (must be < 0.5)")


if __name__ == "__main__":
    main()
