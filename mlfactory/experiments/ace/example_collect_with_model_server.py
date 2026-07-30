"""Example: run ACE collection with a model loaded on demand.

This shows the friendly ``model()`` resource in action. The server starts,
serves the model for the duration of the experiment, and is stopped on exit.
"""
from __future__ import annotations

from pathlib import Path

from mlfactory.core.model_server import model


def main() -> int:
    prompts = Path("mlfactory/experiments/ace/data/prompts.jsonl")
    out_dir = Path("runs/example_collect_qwen35")
    out_dir.mkdir(parents=True, exist_ok=True)

    # `model(...)` is a context manager. It resolves the alias to a GGUF in
    # /home/admin/models, starts llama-server, waits for health, and tears it
    # down when the block exits.
    with model("qwen3.5:4b", gpu=0) as srv:
        print(f"Server ready at {srv.base_url}")
        client = srv.client()

        # You can now use the OpenAI client directly, or pass srv.base_url to
        # collect.py / classify.py.
        response = client.chat.completions.create(
            model="qwen3.5:4b",
            messages=[
                {"role": "user", "content": "Say hello briefly."},
            ],
            max_tokens=50,
            temperature=0.8,
        )
        print(response.choices[0].message.content)

    print("Server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
