#!/usr/bin/env python3
"""Interactive client for the local Qwen voice-adapter chat server."""
from __future__ import annotations

import json
import urllib.request

URL = "http://127.0.0.1:3093/v1/chat/completions"
MODEL = "voice-qwen35-9b-robust-dpo"
messages = [
    {
        "role": "system",
        "content": (
            "Use a clear, warm, practical SMS voice for the business. Answer the latest "
            "message using only visible context; do not invent business facts."
        ),
    }
]

print("Voice adapter chat. Commands: /reset, /quit")
while True:
    try:
        user = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user == "/quit":
        break
    if user == "/reset":
        messages[:] = messages[:1]
        print("Conversation reset.")
        continue
    messages.append({"role": "user", "content": user})
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 256,
    }).encode("utf-8")
    request = urllib.request.Request(
        URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.load(response)
        reply = body["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[request failed: {exc}]")
        messages.pop()
        continue
    messages.append({"role": "assistant", "content": reply})
    print(f"Model: {reply}\n")
