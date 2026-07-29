#!/usr/bin/env python3
"""
DFT on-policy PPO fine-tuning with MMD witness rewards.

Aligns a language model's output distribution with a human reference corpus by
using Maximum Mean Discrepancy (MMD) witness rewards in embedding space.

Note: the installed TRL (1.9+) removed the classic step-based PPOTrainer from the
public API.  This script provides a small compatible shim that keeps the same
`step(queries, responses, rewards)` interface and uses TRL's experimental
AutoModelForCausalLMWithValueHead for the value head.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Silence the experimental import warning.
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
from trl.experimental.ppo.modeling_value_head import AutoModelForCausalLMWithValueHead

# Re-use cheap diagnostics from eval.py (no network side effects on import).
from eval import (
    TokenDistribution,
    load_jsonl,
    non_english_char_rate,
    repetition_rate,
    self_bleu,
)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_mmd_bandwidth(value: str) -> str | float:
    """Argparse converter accepting only ``median`` or a positive finite number."""
    if value == "median":
        return value
    try:
        bandwidth = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bandwidth must be 'median' or a number") from exc
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise argparse.ArgumentTypeError("numeric bandwidth must be positive and finite")
    return bandwidth


def default_model_name(preferred: str = "Qwen/Qwen3.5-4B", fallback: str = "Qwen/Qwen2.5-7B-Instruct") -> str:
    """Return the preferred model name if cached/reachable; else fallback."""
    from transformers.utils import cached_file

    try:
        cached_file(preferred, "config.json")
        return preferred
    except Exception:
        log(f"{preferred} not available, falling back to {fallback}")
        return fallback


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DFT PPO fine-tuning with MMD witness rewards")

    # data / model
    p.add_argument("--model-name", default=None, help="base HF model (default tries Qwen3.6-4B then Qwen2.5-7B)")
    p.add_argument("--train-file", required=True, help="JSONL with prompt, reference fields")
    p.add_argument("--test-file", required=True, help="JSONL with prompt, reference fields")
    p.add_argument("--out-dir", default="./out_train", help="where to save adapters and logs")

    # embedding reward model
    p.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5", help="sentence-transformers embedder")
    p.add_argument("--embed-device", default="cuda:0", help="device for embedder")
    p.add_argument("--mmd-ref-sample-size", default=256, type=int, help="references sampled per MMD computation")
    p.add_argument("--mmd-bandwidth", default="median", type=parse_mmd_bandwidth, help="kernel bandwidth; 'median' or a positive float")
    p.add_argument("--kernel", default="rq", choices=["rbf", "rq"], help="kernel type")
    p.add_argument("--rq-alpha", default=1.0, type=float, help="rational-quadratic alpha")

    # LoRA
    p.add_argument("--lora-r", default=16, type=int)
    p.add_argument("--lora-alpha", default=32, type=int)
    p.add_argument("--lora-dropout", default=0.0, type=float)
    p.add_argument("--lora-target", default="q_proj,k_proj,v_proj,o_proj", help="comma-separated target modules")

    # model loading
    p.add_argument("--load-in-4bit", action="store_true", help="use bitsandbytes 4-bit quantization")
    p.add_argument("--device", default="cuda:0", help="device for policy; 'auto' for accelerate device_map")
    p.add_argument("--ref-device", default="cuda:1", help="device for reference model")

    # generation
    p.add_argument("--max-prompt-length", default=512, type=int)
    p.add_argument("--max-response-length", default=1024, type=int)
    p.add_argument("--temperature", default=0.8, type=float, help="held-out evaluation temperature")
    p.add_argument("--top-p", default=0.95, type=float, help="held-out evaluation top-p")
    p.add_argument("--top-k", default=50, type=int, help="held-out evaluation top-k")
    p.add_argument("--rollout-temperature", default=1.0, type=float)
    p.add_argument("--rollout-top-p", default=1.0, type=float)
    p.add_argument("--rollout-top-k", default=0, type=int)

    # PPO / optimization
    p.add_argument("--batch-size", default=8, type=int, help="prompts per PPO step")
    p.add_argument("--gradient-accumulation-steps", default=1, type=int)
    p.add_argument("--ppo-epochs", default=4, type=int, help="inner PPO epochs per batch")
    p.add_argument("--num-train-epochs", default=3, type=int, help="outer epochs over the training set")
    p.add_argument("--lr", default=1e-5, type=float)
    p.add_argument("--kl-coef", default=0.2, type=float, help="KL penalty coefficient")
    p.add_argument("--cliprange", default=0.2, type=float, help="PPO clip range")
    p.add_argument("--vf-coef", default=0.5, type=float, help="value-function loss coefficient")
    p.add_argument("--critic-mode", choices=["off", "zero-init"], default="off")
    p.add_argument("--ent-coef", default=0.0, type=float)
    p.add_argument("--gamma", default=1.0, type=float)
    p.add_argument("--lam", default=1.0, type=float)

    # automatic short-validation guards
    p.add_argument("--stop-on-nonfinite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-response-length-fraction", default=0.6, type=float)
    p.add_argument("--max-entropy-increase", default=1.0, type=float)
    p.add_argument("--max-ref-kl-k3-per-token", default=0.02, type=float)

    # eval/checkpointing
    p.add_argument("--eval-every", default=100, type=int, help="steps between eval runs")
    p.add_argument("--num-eval-samples", default=100, type=int, help="max test samples to evaluate")
    p.add_argument("--save-every", default=100, type=int, help="steps between adapter checkpoints")

    # thermal management
    p.add_argument("--cooldown-interval-hours", default=1.5, type=float, help="wall-clock hours between cool-down pauses")
    p.add_argument("--cooldown-minutes", default=5, type=int, help="duration of each cool-down pause")
    p.add_argument("--max-steps", default=-1, type=int, help="if >0, stop training after this many steps")

    p.add_argument("--seed", default=42, type=int)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def _device_map(dev: str) -> str | dict:
    return "auto" if dev == "auto" else {"": dev}


def load_policy_and_ref(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target: list[str],
    load_in_4bit: bool,
    device: str,
    ref_device: str,
) -> tuple[AutoModelForCausalLMWithValueHead, torch.nn.Module]:
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # For 4-bit loading, do NOT pass torch_dtype; the quantizer sets compute dtype.
    # Passing torch_dtype with bnb can force an unquantized fp16/bf16 load and OOM.
    common_kwargs: dict = {
        "quantization_config": bnb_config,
        "trust_remote_code": True,
    }
    if not load_in_4bit:
        common_kwargs["torch_dtype"] = torch.bfloat16

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    log("loading policy base model with LoRA")
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        **common_kwargs,
        device_map=_device_map(device),
        peft_config=lora_config,
    )
    if hasattr(policy.pretrained_model, "print_trainable_parameters"):
        policy.pretrained_model.print_trainable_parameters()

    log("loading reference model")
    ref_kwargs = dict(common_kwargs)
    ref_kwargs["device_map"] = _device_map(ref_device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, **ref_kwargs)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    return policy, ref_model


# ---------------------------------------------------------------------------
# embedding + MMD witness rewards
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_name: str, device: str):
        log(f"loading embedder {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        )
        self.dim = self.model.get_embedding_dimension()
        log(f"embedder dim={self.dim}")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def _median_heuristic(X: np.ndarray, Y: np.ndarray, subsample: int = 2000) -> float:
    if X.shape[0] + Y.shape[0] < 2:
        return 1.0
    Z = np.concatenate([X, Y], axis=0).astype(np.float64, copy=False)
    if Z.shape[0] > subsample:
        idx = np.random.choice(Z.shape[0], subsample, replace=False)
        Z = Z[idx]
    dists = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    med = float(np.median(dists[dists > 0]))
    return med if med > 0 else 1.0


def _metric_dtype(X: np.ndarray, Y: np.ndarray) -> np.dtype:
    """Keep metric arithmetic in float32/64 (never embedding float16)."""
    return np.dtype(np.float64 if X.dtype == np.float64 or Y.dtype == np.float64 else np.float32)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    dtype = _metric_dtype(X, Y)
    delta = X.astype(dtype, copy=False)[:, None, :] - Y.astype(dtype, copy=False)[None, :, :]
    sq_dists = np.sum(delta * delta, axis=2, dtype=dtype)
    return np.exp(-sq_dists / dtype.type(2.0 * bandwidth ** 2))


def _rq_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float, alpha: float = 1.0) -> np.ndarray:
    dtype = _metric_dtype(X, Y)
    delta = X.astype(dtype, copy=False)[:, None, :] - Y.astype(dtype, copy=False)[None, :, :]
    sq_dists = np.sum(delta * delta, axis=2, dtype=dtype)
    return (dtype.type(1.0) + sq_dists / dtype.type(2.0 * alpha * bandwidth ** 2)) ** (-alpha)


def _kernel_matrix(
    X: np.ndarray, Y: np.ndarray, kernel: str, bandwidth: float, rq_alpha: float
) -> np.ndarray:
    if kernel == "rbf":
        return _rbf_kernel(X, Y, bandwidth)
    if kernel == "rq":
        return _rq_kernel(X, Y, bandwidth, rq_alpha)
    raise ValueError(f"unknown kernel: {kernel}")


def compute_mmd_witness_rewards(
    responses: list[str],
    refs: list[str],
    embedder: Embedder,
    kernel: str = "rq",
    bandwidth: str | float = "median",
    rq_alpha: float = 1.0,
) -> np.ndarray:
    """
    Per-response MMD witness reward.

    f(y_i) = mean_j k(y_i, ref_j) - mean_{j != i} k(y_i, resp_j)
    Returned rewards are mean-centered.
    """
    if not responses:
        return np.array([], dtype=np.float64)
    if not refs:
        raise ValueError("at least one reference is required")

    resp_emb = embedder.encode(responses)
    ref_emb = embedder.encode(refs)

    bw = bandwidth if isinstance(bandwidth, (int, float)) else _median_heuristic(resp_emb, ref_emb)
    bw = float(bw)
    if bw <= 0:
        bw = 1.0

    n = len(responses)
    rewards = np.zeros(n, dtype=np.float64)

    for i in range(n):
        k_ref = _kernel_matrix(resp_emb[i : i + 1], ref_emb, kernel, bw, rq_alpha).mean()
        if n > 1:
            k_resp = _kernel_matrix(
                resp_emb[i : i + 1], np.delete(resp_emb, i, axis=0), kernel, bw, rq_alpha
            ).mean()
        else:
            k_resp = 0.0
        rewards[i] = k_ref - k_resp

    rewards -= rewards.mean()
    return rewards


def compute_mmd2(X: np.ndarray, Y: np.ndarray, kernel: str, bandwidth: float, rq_alpha: float = 1.0) -> float:
    """Unbiased U-statistic MMD^2 (diagonals excluded)."""
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        raise ValueError("unbiased MMD requires at least two samples in each set")
    Kxx = _kernel_matrix(X, X, kernel, bandwidth, rq_alpha)
    Kyy = _kernel_matrix(Y, Y, kernel, bandwidth, rq_alpha)
    Kxy = _kernel_matrix(X, Y, kernel, bandwidth, rq_alpha)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term1 = Kxx.sum() / (m * (m - 1))
    term2 = Kyy.sum() / (n * (n - 1))
    term3 = Kxy.sum() / (m * n)
    return float(term1 + term2 - 2.0 * term3)


# ---------------------------------------------------------------------------
# minimal PPO trainer (trl.PPOTrainer-compatible step API)
# ---------------------------------------------------------------------------

def retain_through_first_eos(tokens: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    """Retain the first EOS action (including when EOS is also the pad token)."""
    tokens = tokens.reshape(-1)
    if tokens.numel() == 0:
        if eos_id is None:
            raise ValueError("generation produced no action and no EOS token is configured")
        return torch.tensor([eos_id], dtype=tokens.dtype, device=tokens.device)
    if eos_id is not None:
        eos = (tokens == eos_id).nonzero(as_tuple=False)
        if eos.numel():
            return tokens[: eos[0].item() + 1]
    return tokens


def slice_generated_suffix(generated: torch.Tensor, padded_input_width: int) -> torch.Tensor:
    if generated.ndim != 2 or not 0 <= padded_input_width <= generated.shape[1]:
        raise ValueError("invalid generated tensor or padded input width")
    return generated[:, padded_input_width:]


def select_response_prediction_logits(logits: torch.Tensor, response_length: int) -> torch.Tensor:
    """Select preceding-state logits for response actions from full or reduced logits."""
    if logits.ndim != 3 or response_length < 1 or logits.shape[1] < response_length + 1:
        raise ValueError("logits must contain response_length + 1 trailing states")
    return logits[:, -(response_length + 1):-1, :]


def _assert_contiguous_mask(mask: torch.Tensor, left_padded: bool) -> None:
    if mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(dim=1).all():
        raise AssertionError("mask must be rank-2 bool with a valid token in every row")
    transitions = mask[:, 1:].to(torch.int8) - mask[:, :-1].to(torch.int8)
    forbidden = transitions < 0 if left_padded else transitions > 0
    if forbidden.any():
        raise AssertionError("token mask is not contiguous")


def kl_k3(log_ratio: torch.Tensor) -> torch.Tensor:
    """Nonnegative k3 estimator for l=log(policy/reference)."""
    values = torch.expm1(-log_ratio.float()) + log_ratio.float()
    if not torch.isfinite(values).all():
        raise FloatingPointError("nonfinite k3 estimator")
    return values.clamp_min(0.0)


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != values.shape or rewards.shape != mask.shape or rewards.ndim != 2:
        raise ValueError("rewards, values, and mask must have identical [batch,time] shapes")
    _assert_contiguous_mask(mask, left_padded=False)
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(steps)):
        next_values = values[:, t + 1] if t < steps - 1 else torch.zeros_like(last_gae)
        next_mask = mask[:, t + 1] if t < steps - 1 else torch.zeros_like(mask[:, 0])
        delta = rewards[:, t] + gamma * next_values * next_mask.float() - values[:, t]
        advantages[:, t] = delta + gamma * lam * last_gae * next_mask.float()
        last_gae = advantages[:, t]
    advantages *= mask.float()
    return advantages, advantages + values


@dataclass
class PPOConfig:
    model_name: str
    learning_rate: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    init_kl_coef: float = 0.2
    cliprange: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    critic_mode: str = "off"
    seed: int = 42
    log_with: Any | None = None


class PPOTrainer:
    """
    Tiny PPO trainer that exposes the same `step(queries, responses, rewards)`
    interface as the legacy TRL PPOTrainer.
    """

    def __init__(
        self,
        config: PPOConfig,
        model: AutoModelForCausalLMWithValueHead,
        ref_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        if config.critic_mode not in {"off", "zero-init"}:
            raise ValueError(f"invalid critic mode: {config.critic_mode}")
        summary = self.model.v_head.summary
        if config.critic_mode == "off":
            for param in self.model.v_head.parameters():
                param.requires_grad_(False)
        else:
            with torch.no_grad():
                summary.weight.zero_()
                if summary.bias is not None:
                    summary.bias.zero_()
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        self.global_step = 0

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(self, queries: list[torch.Tensor], **gen_kwargs) -> list[torch.Tensor]:
        """Left-pad queries, pass their mask, and return EOS-inclusive actions."""
        if not queries:
            return []
        device = self._model_device()
        pad_id = gen_kwargs.get("pad_token_id", self.tokenizer.pad_token_id)
        eos_id = gen_kwargs.get("eos_token_id", self.tokenizer.eos_token_id)
        max_len = max(q.shape[0] for q in queries)
        if max_len < 1:
            raise ValueError("queries must contain at least one token")

        input_ids = torch.full((len(queries), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, query in enumerate(queries):
            query = query.reshape(-1).long().to(device)
            if query.numel() < 1:
                raise ValueError("queries must contain at least one token")
            input_ids[i, -query.numel():] = query
            attention_mask[i, -query.numel():] = True

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )
        suffix = slice_generated_suffix(outputs, max_len)
        return [retain_through_first_eos(row, eos_id) for row in suffix]

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized GAE-Lambda over padded response sequences."""
        return compute_gae(rewards, values, mask, self.config.gamma, self.config.lam)

    def step(
        self,
        queries: list[torch.Tensor],
        responses: list[torch.Tensor],
        rewards: list[torch.Tensor | float],
    ) -> dict[str, float]:
        device = self._model_device()
        B = len(queries)
        if B == 0 or len(responses) != B or len(rewards) != B:
            raise ValueError("queries, responses, and rewards must have the same nonzero length")
        q_lens = [q.numel() for q in queries]
        r_lens = [r.numel() for r in responses]
        if min(q_lens) < 1 or min(r_lens) < 1:
            raise ValueError("every query and response must contain at least one token")
        max_q = max(q_lens) if q_lens else 0
        max_r = max(r_lens) if r_lens else 0
        pad_id = self.tokenizer.pad_token_id

        # Build left-padded queries and right-padded responses.
        query_padded = torch.full((B, max_q), pad_id, dtype=torch.long, device=device)
        response_padded = torch.full((B, max_r), pad_id, dtype=torch.long, device=device)
        query_mask = torch.zeros((B, max_q), dtype=torch.bool, device=device)
        response_mask = torch.zeros((B, max_r), dtype=torch.bool, device=device)
        for i in range(B):
            q_len, r_len = q_lens[i], r_lens[i]
            query_padded[i, max_q - q_len :] = queries[i].long().to(device)
            query_mask[i, max_q - q_len :] = True
            response_padded[i, :r_len] = responses[i].long().to(device)
            response_mask[i, :r_len] = True

        _assert_contiguous_mask(query_mask, left_padded=True)
        _assert_contiguous_mask(response_mask, left_padded=False)
        full_seq = torch.cat([query_padded, response_padded], dim=1)
        attention_mask = torch.cat([query_mask, response_mask], dim=1)
        assert full_seq.shape == attention_mask.shape == (B, max_q + max_r)

        response_targets = full_seq[:, max_q : max_q + max_r]
        assert response_targets.shape == response_mask.shape
        logits_to_keep = max_r + 1  # preceding state for each response token, plus final unused state

        # Rollout under the old policy. Eval mode makes old-policy scoring
        # deterministic; validation requires LoRA dropout zero for train/eval parity.
        self.model.eval()
        with torch.no_grad():
            policy_out = self.model(
                input_ids=full_seq,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
                use_cache=False,
            )
            assert policy_out[0].shape[1] == logits_to_keep
            # Keep full-vocabulary tensors in model dtype: casting [B,R,V] to
            # float32 adds several GiB at batch 20. Cast only selected-token
            # statistics after the vocabulary reduction.
            policy_logits = select_response_prediction_logits(policy_out[0], max_r)
            policy_values = policy_out[2].to(device).float()
            assert policy_logits.shape[:2] == response_targets.shape
            rollout_log_probs_all = F.log_softmax(policy_logits, dim=-1)
            old_log_probs = rollout_log_probs_all.gather(
                2, response_targets.unsqueeze(-1)
            ).squeeze(-1).float() * response_mask
            rollout_entropy = (
                -(rollout_log_probs_all.exp() * rollout_log_probs_all)
                .sum(dim=-1)[response_mask]
                .float()
                .mean()
            )
            if self.config.critic_mode == "off":
                old_values = torch.zeros_like(old_log_probs, dtype=torch.float32)
            else:
                old_values = policy_values[:, max_q - 1 : max_q + max_r - 1] * response_mask
                assert old_values.shape == response_mask.shape
            del policy_out, policy_logits, policy_values, rollout_log_probs_all

            # Compute selected reference token log-probs on GPU 1 and transfer
            # only [batch, response_length], never full vocabulary logits.
            ref_device = next(self.ref_model.parameters()).device
            ref_full_seq = full_seq.to(ref_device)
            ref_attention_mask = attention_mask.to(ref_device)
            ref_targets = response_targets.to(ref_device)
            ref_mask = response_mask.to(ref_device)
            ref_out = self.ref_model(
                input_ids=ref_full_seq,
                attention_mask=ref_attention_mask,
                logits_to_keep=logits_to_keep,
                use_cache=False,
            )
            assert ref_out.logits.shape[1] == logits_to_keep
            ref_logits = select_response_prediction_logits(ref_out.logits, max_r)
            ref_log_probs = (
                F.log_softmax(ref_logits, dim=-1)
                .gather(2, ref_targets.unsqueeze(-1))
                .squeeze(-1)
                .float()
            )
            ref_log_probs = (ref_log_probs * ref_mask).to(device)
            del ref_out, ref_logits, ref_full_seq, ref_attention_mask, ref_targets, ref_mask

        # Reference regularization is a detached rollout reward, never a
        # differentiable minibatch loss. Add the centered witness at termination.
        log_ratio = (old_log_probs.float() - ref_log_probs.float()).detach()
        response_rewards = -self.config.init_kl_coef * log_ratio * response_mask
        witness = torch.tensor(
            [r.item() if isinstance(r, torch.Tensor) else float(r) for r in rewards],
            dtype=torch.float32,
            device=device,
        )
        witness = witness - witness.mean()
        for i, r_len in enumerate(r_lens):
            response_rewards[i, r_len - 1] += witness[i]
        if not torch.isfinite(response_rewards).all():
            raise FloatingPointError("nonfinite rollout reward")

        advantages, returns = self._compute_gae(response_rewards, old_values, response_mask)

        # Whiten advantages over the valid response tokens.
        flat_adv = advantages[response_mask]
        if flat_adv.numel() > 1:
            adv_mean = flat_adv.mean()
            adv_std = flat_adv.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = advantages * response_mask.float()

        valid_log_ratio = log_ratio[response_mask]
        ref_k3 = kl_k3(valid_log_ratio)
        if not torch.isfinite(ref_k3).all():
            raise FloatingPointError("nonfinite reference KL diagnostic")
        stats_accum: dict[str, list[float]] = {
            "ppo/loss/policy": [], "ppo/loss/value": [], "ppo/entropy": [],
            "ppo/total_loss": [], "ppo/old_new_kl_k3": [],
            "ppo/clip_fraction": [], "ppo/ratio": [],
        }

        num_samples = B
        mb_size = self.config.mini_batch_size
        grad_accum = max(1, self.config.gradient_accumulation_steps)

        for _ in range(self.config.ppo_epochs):
            perm = torch.randperm(num_samples, device=device)
            self.optimizer.zero_grad()
            mb_starts = list(range(0, num_samples, mb_size))
            for mb_index, start in enumerate(mb_starts):
                end = min(start + mb_size, num_samples)
                idx = perm[start:end]
                group_start = (mb_index // grad_accum) * grad_accum
                group_end = min(group_start + grad_accum, len(mb_starts))
                group_valid = sum(
                    response_mask[perm[s:min(s + mb_size, num_samples)]].sum().item()
                    for s in mb_starts[group_start:group_end]
                )

                mb_full_seq = full_seq[idx]
                mb_attention_mask = attention_mask[idx]
                mb_response_mask = response_mask[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_targets = response_targets[idx]

                self.model.train()
                out = self.model(
                    input_ids=mb_full_seq,
                    attention_mask=mb_attention_mask,
                    logits_to_keep=logits_to_keep,
                    use_cache=False,
                )
                assert out[0].shape[1] == logits_to_keep
                logits = select_response_prediction_logits(out[0], max_r)
                values = out[2].to(device).float()
                assert logits.shape[:2] == mb_targets.shape

                log_probs_all = F.log_softmax(logits, dim=-1)
                new_log_probs = (
                    log_probs_all.gather(2, mb_targets.unsqueeze(-1))
                    .squeeze(-1)
                    .float()
                    * mb_response_mask
                )

                new_values = values[:, max_q - 1 : max_q + max_r - 1] * mb_response_mask

                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.cliprange, 1 + self.config.cliprange) * mb_advantages
                policy_loss = -torch.min(surr1, surr2)[mb_response_mask].mean()

                if self.config.critic_mode == "off":
                    value_loss = policy_loss.new_zeros(())
                else:
                    value_loss = ((new_values - mb_returns.detach()) ** 2)[mb_response_mask].mean()

                entropy = (
                    -(torch.exp(log_probs_all) * log_probs_all)
                    .sum(dim=-1)[mb_response_mask]
                    .float()
                    .mean()
                )
                old_new_delta = (mb_old_log_probs.detach() - new_log_probs)[mb_response_mask]
                old_new_k3 = kl_k3(old_new_delta).mean()
                valid_ratio = ratio[mb_response_mask]
                clip_fraction = ((valid_ratio - 1.0).abs() > self.config.cliprange).float().mean()

                vf_coef = 0.0 if self.config.critic_mode == "off" else self.config.vf_coef
                loss = policy_loss + vf_coef * value_loss - self.config.ent_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite PPO loss")
                weight = mb_response_mask.sum().item() / group_valid
                (loss * weight).backward()

                if (mb_index + 1) % grad_accum == 0 or mb_index + 1 == len(mb_starts):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                stats_accum["ppo/loss/policy"].append(policy_loss.item())
                stats_accum["ppo/loss/value"].append(value_loss.item())
                stats_accum["ppo/entropy"].append(entropy.item())
                stats_accum["ppo/total_loss"].append(loss.item())
                stats_accum["ppo/old_new_kl_k3"].append(old_new_k3.item())
                stats_accum["ppo/clip_fraction"].append(clip_fraction.item())
                stats_accum["ppo/ratio"].append(valid_ratio.mean().item())

        self.model.eval()
        self.global_step += 1
        result = {k: sum(v) / len(v) if v else 0.0 for k, v in stats_accum.items()}
        seq_signed = (log_ratio * response_mask).sum(dim=1)
        seq_k3 = (kl_k3(log_ratio) * response_mask).sum(dim=1)
        result.update({
            "rollout/entropy": rollout_entropy.item(),
            "rollout/ref_log_ratio_signed_per_token": valid_log_ratio.mean().item(),
            "rollout/ref_kl_k3_per_token": ref_k3.mean().item(),
            "rollout/ref_log_ratio_signed_sequence": seq_signed.mean().item(),
            "rollout/ref_kl_k3_sequence": seq_k3.mean().item(),
            "rollout/ref_kl_reward_signed_sequence": (-self.config.init_kl_coef * seq_signed).mean().item(),
            "rollout/ref_kl_penalty_k3_sequence": (self.config.init_kl_coef * seq_k3).mean().item(),
        })
        if not all(np.isfinite(v) for v in result.values()):
            raise FloatingPointError("nonfinite PPO statistic")
        return result


# ---------------------------------------------------------------------------
# generation helpers
# ---------------------------------------------------------------------------

def batched_generate(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_prompt_length: int,
    batch_size: int = 8,
) -> list[str]:
    device = next(model.parameters()).device
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        model.eval()
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        suffix = slice_generated_suffix(gen, input_ids.shape[1])
        for row in suffix:
            resp = retain_through_first_eos(row, tokenizer.eos_token_id)
            outputs.append(tokenizer.decode(resp, skip_special_tokens=True))
    return outputs


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def run_eval(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    test_rows: list[dict],
    embedder: Embedder,
    token_dist: TokenDistribution,
    args: argparse.Namespace,
    fixed_bandwidth: float | None = None,
) -> dict:
    prompts = [r["prompt"] for r in test_rows]
    refs = [r["reference"] for r in test_rows]

    hyps = batched_generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_prompt_length=args.max_prompt_length,
        batch_size=args.batch_size,
    )

    hyp_emb = embedder.encode(hyps)
    ref_emb = embedder.encode(refs)
    bw = fixed_bandwidth if fixed_bandwidth is not None else args.mmd_bandwidth
    if bw == "median":
        bw = _median_heuristic(hyp_emb, ref_emb)
    mmd2 = compute_mmd2(hyp_emb, ref_emb, args.kernel, float(bw), args.rq_alpha)

    return {
        "n": len(hyps),
        "mmd2": mmd2,
        "l2_1gram": token_dist.l2_distance(hyps, refs, n=1),
        "l2_2gram": token_dist.l2_distance(hyps, refs, n=2),
        "l2_3gram": token_dist.l2_distance(hyps, refs, n=3),
        "repetition_rate": repetition_rate(hyps),
        "non_english_char_rate": non_english_char_rate(hyps),
        "self_bleu": self_bleu(hyps),
        "avg_hyp_chars": sum(len(h) for h in hyps) / max(len(hyps), 1),
    }


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    if args.model_name is None:
        args.model_name = default_model_name()
    if (args.rollout_temperature, args.rollout_top_p, args.rollout_top_k) != (1.0, 1.0, 0):
        raise ValueError("training rollouts require temperature=1, top_p=1, top_k=0")
    if args.critic_mode == "off" and (args.gamma != 1.0 or args.lam != 1.0):
        raise ValueError("critic-off validation requires gamma=1 and lam=1")
    if args.lora_dropout != 0.0:
        raise ValueError("LoRA dropout must be 0 so rollout and update log-probs are comparable")
    if args.batch_size < 2 or args.num_eval_samples == 1:
        raise ValueError("training and unbiased MMD evaluation require at least two samples")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    log(f"model={args.model_name}  train={args.train_file}  test={args.test_file}")

    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    train_rows = load_jsonl(args.train_file)
    test_rows_all = load_jsonl(args.test_file)
    if args.num_eval_samples and len(test_rows_all) > args.num_eval_samples:
        test_rows = random.sample(test_rows_all, args.num_eval_samples)
    else:
        test_rows = test_rows_all
    if len(train_rows) < 2 or len(test_rows) < 2:
        raise ValueError("training and unbiased MMD evaluation require at least two rows")
    log(f"loaded {len(train_rows)} train, {len(test_rows)} eval samples")

    tokenizer = load_tokenizer(args.model_name)
    policy, ref_model = load_policy_and_ref(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target.split(","),
        load_in_4bit=args.load_in_4bit,
        device=args.device,
        ref_device=args.ref_device,
    )

    embedder = Embedder(args.embed_model, device=args.embed_device)
    token_dist = TokenDistribution(args.model_name)

    mini_batch_size = max(1, args.batch_size // args.gradient_accumulation_steps)
    if args.batch_size % args.gradient_accumulation_steps != 0:
        log(
            f"warning: batch_size {args.batch_size} not divisible by grad_accum "
            f"{args.gradient_accumulation_steps}; using mini_batch_size={mini_batch_size}"
        )

    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        gamma=args.gamma,
        lam=args.lam,
        init_kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        critic_mode=args.critic_mode,
        seed=args.seed,
        log_with=None,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    gen_kwargs = {
        "max_new_tokens": args.max_response_length,
        "do_sample": True,
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    log_file = out_dir / "logs" / "train.jsonl"
    summary: dict = {"config": vars(args), "train_steps": [], "eval_steps": []}

    # Fixed held-out rows and reference-only bandwidth make monitor points comparable.
    eval_ref_emb = embedder.encode([row["reference"] for row in test_rows])
    fixed_eval_bandwidth = (_median_heuristic(eval_ref_emb, eval_ref_emb)
                            if args.mmd_bandwidth == "median" else float(args.mmd_bandwidth))
    log("running fixed held-out step-0 eval")
    eval_zero = run_eval(policy, tokenizer, test_rows, embedder, token_dist, args, fixed_eval_bandwidth)
    eval_zero.update({"step": 0, "epoch": 0, "fixed_bandwidth": fixed_eval_bandwidth})
    summary["eval_steps"].append(eval_zero)
    with open(out_dir / "logs" / "eval_step_0.json", "w", encoding="utf-8") as f:
        json.dump(eval_zero, f, indent=2)
    log(json.dumps(eval_zero))

    global_step = 0
    guard_triggered = False
    guard_report: dict[str, Any] | None = None
    baseline_response_tokens: float | None = None
    baseline_entropy: float | None = None
    short_count = 0
    entropy_count = 0
    train_start_time = time.time()
    last_cooldown = train_start_time
    for epoch in range(1, args.num_train_epochs + 1):
        log(f"epoch {epoch}/{args.num_train_epochs}")
        indices = list(range(len(train_rows)))
        random.shuffle(indices)

        for start in range(0, len(indices), args.batch_size):
            # thermal cool-down: pause every N hours of wall time
            if time.time() - last_cooldown >= args.cooldown_interval_hours * 3600:
                log(f"thermal cool-down: pausing for {args.cooldown_minutes} minutes after {args.cooldown_interval_hours} hours")
                time.sleep(args.cooldown_minutes * 60)
                last_cooldown = time.time()
                log("thermal cool-down complete, resuming training")

            batch_idx = indices[start : start + args.batch_size]
            if len(batch_idx) < 2:
                log("skipping final singleton batch (unbiased MMD requires m>=2)")
                continue
            if args.max_steps > 0 and global_step >= args.max_steps:
                log(f"reached max-steps {args.max_steps}; stopping training")
                break
            global_step += 1
            if torch.cuda.is_available():
                for gpu_idx in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(gpu_idx)
            batch_prompts = [train_rows[i]["prompt"] for i in batch_idx]

            query_tensors = [
                tokenizer.encode(
                    p,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_prompt_length,
                ).squeeze(0)
                for p in batch_prompts
            ]

            response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)
            responses_text = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

            ref_sample_size = min(args.mmd_ref_sample_size, len(train_rows))
            ref_sample = [train_rows[i]["reference"] for i in random.sample(range(len(train_rows)), ref_sample_size)]
            rewards = compute_mmd_witness_rewards(
                responses_text,
                ref_sample,
                embedder,
                kernel=args.kernel,
                bandwidth=args.mmd_bandwidth,
                rq_alpha=args.rq_alpha,
            )
            reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]

            try:
                stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            except FloatingPointError as exc:
                if not args.stop_on_nonfinite:
                    raise
                stats = {}
                guard_triggered = True
                guard_report = {"step": global_step, "reason": "nonfinite", "detail": str(exc)}
                log(f"guard triggered: {guard_report}")
                break

            # Training diagnostics on the current batch.
            resp_emb = embedder.encode(responses_text)
            ref_emb = embedder.encode(ref_sample)
            bw = args.mmd_bandwidth
            if bw == "median":
                bw = _median_heuristic(resp_emb, ref_emb)
            mmd2_train = compute_mmd2(resp_emb, ref_emb, args.kernel, bw, args.rq_alpha)
            l2_1_train = token_dist.l2_distance(responses_text, ref_sample, n=1)

            train_step_log = {
                "step": global_step,
                "epoch": epoch,
                "batch_reward_mean": float(rewards.mean()),
                "batch_reward_std": float(rewards.std()),
                "mmd2": mmd2_train,
                "l2_1gram": l2_1_train,
                "policy_loss": stats["ppo/loss/policy"],
                "value_loss": stats["ppo/loss/value"],
                "entropy": stats["ppo/entropy"],
                "avg_response_chars": sum(len(r) for r in responses_text) / max(len(responses_text), 1),
                "mean_response_tokens": float(np.mean([r.numel() for r in response_tensors])),
                "eos_rate": float(np.mean([bool((r == tokenizer.eos_token_id).any()) for r in response_tensors])),
                "truncation_rate": float(np.mean([r.numel() >= args.max_response_length and not bool((r == tokenizer.eos_token_id).any()) for r in response_tensors])),
                **stats,
                "gpu_peak_alloc_gib": [
                    round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                    for i in range(torch.cuda.device_count())
                ],
                "gpu_peak_reserved_gib": [
                    round(torch.cuda.max_memory_reserved(i) / 2**30, 2)
                    for i in range(torch.cuda.device_count())
                ],
            }
            summary["train_steps"].append(train_step_log)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(train_step_log) + "\n")
            log(json.dumps(train_step_log))

            mean_tokens = train_step_log["mean_response_tokens"]
            entropy = stats["rollout/entropy"]
            if baseline_response_tokens is None:
                baseline_response_tokens, baseline_entropy = mean_tokens, entropy
            else:
                short_count = short_count + 1 if mean_tokens < args.min_response_length_fraction * baseline_response_tokens else 0
                entropy_count = entropy_count + 1 if entropy > baseline_entropy + args.max_entropy_increase else 0
            reasons = []
            if short_count >= 2:
                reasons.append("response_length_collapse")
            if entropy_count >= 2:
                reasons.append("entropy_increase")
            if stats["rollout/ref_kl_k3_per_token"] > args.max_ref_kl_k3_per_token:
                reasons.append("reference_kl_k3_limit")
            if args.stop_on_nonfinite and not all(np.isfinite(v) for v in train_step_log.values() if isinstance(v, (int, float))):
                reasons.append("nonfinite")
            if reasons:
                guard_triggered = True
                guard_report = {"step": global_step, "reasons": reasons,
                    "baseline_response_tokens": baseline_response_tokens, "baseline_entropy": baseline_entropy,
                    "observed_response_tokens": mean_tokens, "observed_entropy": entropy,
                    "reference_kl_k3_per_token": stats["rollout/ref_kl_k3_per_token"]}
                guard_dir = out_dir / f"guard-checkpoint-{global_step}"
                guard_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(guard_dir)
                tokenizer.save_pretrained(guard_dir)
                with open(out_dir / "guard_report.json", "w", encoding="utf-8") as f:
                    json.dump(guard_report, f, indent=2)
                log(f"guard triggered; checkpoint saved to {guard_dir}: {guard_report}")
                break

            if global_step % args.save_every == 0:
                ckpt_dir = out_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                log(f"saved checkpoint {ckpt_dir}")

            if global_step % args.eval_every == 0:
                log("running eval")
                eval_out = run_eval(policy, tokenizer, test_rows, embedder, token_dist, args, fixed_eval_bandwidth)
                eval_out["step"] = global_step
                eval_out["epoch"] = epoch
                summary["eval_steps"].append(eval_out)
                with open(out_dir / "logs" / f"eval_step_{global_step}.json", "w", encoding="utf-8") as f:
                    json.dump(eval_out, f, indent=2)
                log(json.dumps(eval_out))
        if guard_triggered or (args.max_steps > 0 and global_step >= args.max_steps):
            break

    if guard_triggered:
        if guard_report is not None and not (out_dir / "guard_report.json").exists():
            guard_dir = out_dir / f"guard-checkpoint-{global_step}"
            guard_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(guard_dir)
            tokenizer.save_pretrained(guard_dir)
            with open(out_dir / "guard_report.json", "w", encoding="utf-8") as f:
                json.dump(guard_report, f, indent=2)
        summary["guard"] = guard_report
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log("training terminated cleanly by guard; no final adapter was written")
        return 0

    # final eval on the trained policy
    log("running final eval")
    eval_out = run_eval(policy, tokenizer, test_rows, embedder, token_dist, args, fixed_eval_bandwidth)
    eval_out["step"] = global_step
    eval_out["epoch"] = args.num_train_epochs
    summary["eval_steps"].append(eval_out)
    with open(out_dir / "logs" / f"eval_step_{global_step}.json", "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2)
    log(json.dumps(eval_out))

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"training complete; final adapter saved to {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
