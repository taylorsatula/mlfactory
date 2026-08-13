#!/usr/bin/env python3
"""Small local OpenAI-compatible chat server for the Qwen voice adapter.

This is intentionally localhost-only and sequential: it is a conversation test
server, not a production service.  Run it with CUDA_VISIBLE_DEVICES=0 while the
regular llama services are stopped on that GPU.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from voice_prompt import DEFAULT_BUSINESS_STATE, PromptVariant, build_system_prompt, variant_for_key
from voice_safety import infer_mode, response_violation

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

# Defaults target the human-preference DPO adapter after its release gates
# pass. The path remains overrideable for adapter comparisons.
BASE_MODEL = Path(os.environ.get("VOICE_BASE_MODEL", "/home/admin/models/hf/Qwen3.5-9B"))
ADAPTER = Path(os.environ.get("VOICE_ADAPTER", "/home/admin/mlfactory/runs/voice-qwen35-9b-robust-dpo-v3-20260806T0600Z/artifacts/policy_adapter"))
DEVICE = os.environ.get("VOICE_DEVICE", "cuda:0")
MODEL_ID = os.environ.get("VOICE_MODEL_ID", "voice-qwen35-9b-robust-dpo")
PORT = int(os.environ.get("VOICE_PORT", "3093"))
BIND = os.environ.get("VOICE_BIND", "127.0.0.1")
PROMPT_VARIANT = os.environ.get("VOICE_PROMPT_VARIANT", PromptVariant.PROVIDER.value)
SYSTEM_DEFAULT = build_system_prompt(variant=PROMPT_VARIANT)
DEFAULT_STATE = DEFAULT_BUSINESS_STATE
GENERATION_LOCK = threading.Lock()

print(f"Loading base model {BASE_MODEL} and adapter {ADAPTER} on {DEVICE}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    ),
    device_map={"": 0},
)
model = PeftModel.from_pretrained(model, ADAPTER, local_files_only=True, is_trainable=False).eval()
model.config.use_cache = True
print(f"Voice adapter loaded; server is ready ({MODEL_ID}).", flush=True)


def _generate_once(messages: list[dict[str, Any]], max_tokens: int, temperature: float, top_p: float, mode: str = "business_reply", business_state: Any = None, prompt_variant: str | None = None, extra_instruction: str | None = None) -> str:
    clean = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        clean.append({"role": role, "content": str(content)})
    if isinstance(business_state, dict):
        state_text = "\n".join(f"{key}: {value}" for key, value in sorted(business_state.items()) if value not in (None, "", [], {})) or DEFAULT_STATE
    elif isinstance(business_state, str) and business_state.strip():
        state_text = business_state.strip()
    else:
        state_text = DEFAULT_STATE
    chosen_variant = prompt_variant or PROMPT_VARIANT
    if chosen_variant == "rotate":
        seed_text = " ".join(str(item.get("content") or "") for item in clean if item.get("role") == "user")
        chosen_variant = variant_for_key(seed_text or "empty-request").value
    managed_system = build_system_prompt(mode, business_state if isinstance(business_state, dict) else None, chosen_variant)
    if isinstance(business_state, str) and business_state.strip():
        managed_system = build_system_prompt(mode, {"operator_state": state_text}, chosen_variant)
    if extra_instruction:
        managed_system += f"\n\nAdditional response constraint:\n{extra_instruction}"
    if not clean or clean[0].get("role") != "system":
        clean.insert(0, {"role": "system", "content": managed_system})
    else:
        caller_system = str(clean[0].get("content") or "").strip()
        clean[0] = {"role": "system", "content": managed_system + (f"\n\nAdditional caller instruction:\n{caller_system}" if caller_system and caller_system != SYSTEM_DEFAULT else "")}
    try:
        prompt = tokenizer.apply_chat_template(
            clean, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=4096,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    do_sample = temperature > 0.01
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max(1, min(int(max_tokens), 512)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update({"temperature": max(0.05, min(float(temperature), 2.0)), "top_p": max(0.05, min(float(top_p), 1.0))})
    with GENERATION_LOCK, torch.inference_mode():
        output = model.generate(**kwargs)
    generated = output[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def safe_fallback(violation: str, mode: str) -> str:
    if violation == "identity_claim":
        return "I’m the business’s virtual assistant, and I don’t have access to your calendar."
    if violation == "service_leakage" and mode in {"general_question", "casual_sms"}:
        return "That sounds funny—tell me more."
    if violation == "unsupported_external_action":
        if mode in {"general_question", "casual_sms"}:
            return "That sounds funny—tell me more."
        return "I can draft that message, but I can’t send it from here."
    if violation == "unsupported_action":
        if mode in {"general_question", "casual_sms"}:
            return "That sounds funny—tell me more."
        return "I don’t have calendar access to confirm that time, so I don’t want to guess."
    if violation in {"repetition", "overrefusal"}:
        return "Could you share the next detail you want help with?"
    if violation == "invented_amount":
        return "I want to check the details before I explain the amount."
    if violation == "privacy":
        return "Please keep private details out of text, and I’ll help with the next step."
    return "I want to make sure I give you an accurate answer."


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def _overrefuses_customer_detail(text: str, messages: list[dict[str, Any]]) -> bool:
    latest = " ".join(
        str(row.get("content") or "")
        for row in messages
        if str(row.get("role")) in {"user", "customer"}
    )
    supplied_detail = re.search(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|"
        r"november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"\d{1,2}(?:st|nd|rd|th)?|window(?:s)?|morning|afternoon|evening)\b", latest, re.I
    )
    irrelevant_refusal = re.search(r"\b(?:pricing|inventory|cost)\b|tools? to (?:verify|confirm)", text, re.I)
    return bool(supplied_detail and irrelevant_refusal and not re.search(r"\b(?:price|cost)\b", latest, re.I))


def _repeats_recent_assistant(text: str, messages: list[dict[str, Any]]) -> bool:
    candidate = _normalized(text)
    if len(candidate) < 24:
        return False
    recent = [
        _normalized(row.get("content", ""))
        for row in messages
        if str(row.get("role")) == "assistant"
    ][-3:]
    if candidate in recent:
        return True
    boilerplate = ("i can help with that", "i understand", "that sounds good")
    if any(
        any(candidate.startswith(prefix) and prior.startswith(prefix) for prefix in boilerplate)
        for prior in recent
    ):
        return True
    candidate_first = candidate.split(". ", 1)[0]
    if len(candidate_first) >= 24 and candidate_first.startswith(("i have noted", "i cannot verify", "i do not have", "i don't have")):
        return any(candidate_first == prior.split(". ", 1)[0] for prior in recent)
    return False


def generate(messages: list[dict[str, Any]], max_tokens: int, temperature: float, top_p: float, mode: str = "auto", business_state: Any = None, prompt_variant: str | None = None) -> str:
    effective_mode = infer_mode(messages, mode)
    text = _generate_once(messages, max_tokens, temperature, top_p, effective_mode, business_state, prompt_variant)
    state = business_state if isinstance(business_state, dict) else None
    violation = response_violation(text, effective_mode, state)
    repeated = _repeats_recent_assistant(text, messages)
    overrefusal = _overrefuses_customer_detail(text, messages)
    if violation or repeated or overrefusal:
        # Sampling occasionally produces an operational claim, drifts back
        # into service language, or copies the previous assistant turn. Retry
        # with an explicit correction; the fallback remains deterministic.
        retry_instruction = (
            "Answer the latest user message directly. Do not repeat any previous assistant wording. "
            "Preserve dates, times, counts, and other details the user already supplied; do not ask for them again. "
            "Do not discuss pricing, inventory, or unrelated unavailable tools unless the user asked about them. "
            "Do not claim an action, tool access, appointment, availability, or identity that is not verified."
        )
        retry = _generate_once(
            messages, max_tokens, 0.0 if violation else max(0.8, temperature), top_p,
            effective_mode, business_state, prompt_variant, retry_instruction,
        )
        retry_violation = response_violation(retry, effective_mode, state)
        retry_repeated = _repeats_recent_assistant(retry, messages)
        if retry_violation is None and not retry_repeated:
            return retry
        fallback = safe_fallback(
            retry_violation or ("repetition" if repeated or retry_repeated else ("overrefusal" if overrefusal else violation)),
            effective_mode,
        )
        if _repeats_recent_assistant(fallback, messages):
            return "What would you like to tackle next?"
        return fallback
    return text


class Handler(BaseHTTPRequestHandler):
    server_version = "voice-qwen-poc/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.address_string()}] {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/health"}:
            self.send_json(200, {"status": "ok", "model": MODEL_ID})
        elif self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
        else:
            self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            messages = request.get("messages") or []
            if not isinstance(messages, list):
                raise ValueError("messages must be an array")
            started = time.monotonic()
            text = generate(
                messages,
                int(request.get("max_tokens", request.get("max_completion_tokens", 256))),
                float(request.get("temperature", 0.7)),
                float(request.get("top_p", 0.95)),
                str(request.get("mode", "auto")),
                request.get("business_state"),
                request.get("prompt_variant"),
            )
            response = {
                "id": f"voice-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "metadata": {"duration_seconds": round(time.monotonic() - started, 3)},
            }
            self.send_json(200, response)
        except Exception as exc:
            print(f"request error: {type(exc).__name__}: {exc}", flush=True)
            self.send_json(400, {"error": {"message": str(exc), "type": type(exc).__name__}})


if __name__ == "__main__":
    server = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"Listening on http://127.0.0.1:{PORT}/v1", flush=True)
    server.serve_forever()
