# Voice pipeline iteration dossier

## Objective
Refine the earliest voice-data pipeline to match the real SMS target shape, generate better fictional grounded data, then run a short GPU smoke-training/evaluation before any long run. Preserve all existing adapters/checkpoints. Keep raw SMS local and in memory; do not copy it into synthetic artifacts, logs, prompts, or remote calls. User explicitly prefers a practical PII boundary rather than elaborate redaction that blocks useful experiments.

## Real target profile inspected
Raw corpus: `mlfactory/experiments/voice/data/threads/`

- 449 threads, 6,173 messages; 3,406 business/Taylor and 2,767 customer messages.
- Thread shape: 117 one-message, 89 with 2–5 messages, 67 with 6–10, 176 over 10 messages; largest is 475 messages.
- Median customer message: 9 words; median business reply: 19 words.
- Keyword presence (overlapping, not disjoint): scheduling 413 threads; service scope 435; uncertainty/change 389; casual/personal 305; follow-up 299; pricing/payment 175; logistics/access/arrival 158.
- Full review gate summary: 5,100 KEEP; 267 PSEUDONYMIZE; 55 HUMAN_REVIEW; 38 EXCLUDE_SESSION.

Implication: this is multi-turn small-business coordination, not one-shot generic SMS. Core needs are concise context retention, scheduling/rescheduling/cancellation, scope/quotes/payment, logistics/access/arrival, complaint recovery, follow-ups, ambiguity/change-of-mind, and clean casual pivots.

## Existing deployed model
- Base: `/home/admin/models/hf/Qwen3.5-9B`
- Current adapter: `/home/admin/mlfactory/runs/voice-qwen35-9b-robust-dpo-v3-20260806T0600Z/artifacts/policy_adapter`
- Server port: `3093`; current server was stopped before GPU generation.
- Existing self-play fixture is committed as `cdfe308`.

## Serving iteration already made (uncommitted voice files)
Files:
- `mlfactory/experiments/voice/voice_safety.py`
- `mlfactory/experiments/voice/voice_prompt.py`
- `mlfactory/experiments/voice/voice_chat_server.py`
- `tests/test_voice_safety.py`

Changes: history-aware mode inference, date/time/business signals, explicit casual-pivot handling, provider prompt variant default, non-business mode guidance, identity/action/availability guards, repetition retry/fallbacks, and safer grounding checks. Tests currently report **20 passed**.

Final short probes showed:
- Date/window details are retained and missing details are requested.
- Business multi-turn context is preserved without claiming unknown availability.
- Casual pivot exits the service thread.
- Identity response says virtual assistant/no calendar access.
- Unsupported booking does not claim a booking.
- Some remaining verbosity/repetition exists; this is now a candidate for targeted training data, not a reason to overwrite the adapter.

## New synthetic pipeline work
Created fictional structured catalog:
- `mlfactory/experiments/voice/data/grounded_scenario_catalog.json`
- 24 scenarios spanning scheduling, reschedule/weather, cancellation, service scope, quote/pricing, payment, access, arrival, complaint/recovery, clarification, repeat customers, completion, casual pivot, and long follow-up.
- Each plan has visible conversation, visible `verified_state`, target split (`train`/`eval`), domain, length band, topic terms, and short customer-style variants.

Created generator:
- `mlfactory/experiments/voice/generate_grounded_voice.py`
- Uses frozen **base Qwen3.5-9B by default**, not the current adapter, to avoid self-distillation.
- Supports `--part-index/--part-count` and separate GPUs; outputs only fictional contexts/targets and aggregate manifests.
- Uses canonical prompt/state interface and target-only generation.
- Last fix: if authored history ended with a customer turn, inserts a visible short owner acknowledgement before adding the final customer turn so role alternation is valid.

Created deterministic filter:
- `mlfactory/experiments/voice/filter_grounded_voice.py`
- Role/order, length, simple privacy/meta checks, grounding/action checks against visible state, generic-target rejection, context/target duplicate caps, train/eval/replay outputs.
- It does not read the real corpus.

## First generation attempt (preserve; do not overwrite)
Run: `/home/admin/mlfactory/runs/voice-grounded-synth-20260805T210606Z/`

- Two GPU processes successfully generated 800 total candidates: 400 per GPU.
- Filter result: only 33 accepted because 765 were rejected for even/invalid turn counts; catalog histories that ended in customer turns caused invalid alternation. This was a pipeline/schema bug, not evidence the teacher failed.
- Filter report: `runs/voice-grounded-synth-20260805T210606Z/artifacts/filtered/filter_report.json`
- Raw parts: `runs/voice-grounded-synth-20260805T210606Z/artifacts/raw/part-0.jsonl` and `part-1.jsonl`.
- Generator fix is applied but has not yet been rerun.

## Extended run now launched
The corrected 800-candidate run produced 595 accepted records: 409 train, 186 disjoint eval, and 8 grounded casual replay records. I also merged 34 authored casual/general replay records from the existing safe builder, for 42 replay records total.

- Data: `runs/voice-grounded-synth-fixed-20260805T211214Z/artifacts/extended_data/`
- Spec: `mlfactory/experiments/voice/specs/voice_qwen35_9b_grounded_v1.yaml`
- Launch helper: `mlfactory/experiments/voice/run_grounded_voice_long.sh`
- Long run: `voice-qwen35-9b-grounded-v1-20260805T212500Z`
- Long-run log: `runs/voice-qwen35-9b-grounded-v1-20260805T212500Z.launcher.log`
- Training log: `runs/voice-qwen35-9b-grounded-v1-20260805T212500Z/logs/train.log`
- It is launched with `nohup` on logical `cuda:0` mapped to physical GPU 1; current DPO serving was restarted on physical GPU 0.
- Dashboard config: `mlfactory/experiments/voice/dashboard_voice-robust-train.json`

Configured dashboard command:

```bash
cd /home/admin/mlfactory
/home/admin/dft-eval-harness/.venv312/bin/python -m mlfactory.cli \\
  --registry /home/admin/mlfactory/.mlfactory/registry.db dashboard \\
  --watch-run voice-qwen35-9b-grounded-v1-20260805T212500Z --refresh 2
```

The dashboard reports optimizer progress, train/eval/replay record counts, loss, validation loss, gradient norm, memory, checkpoints, GPU telemetry, and the recent training log. The runner now publishes `running` status to the registry before plugin execution.

## Post-run evaluation
The extended run completed successfully at 400 steps:

- Adapter: `runs/voice-qwen35-9b-grounded-v1-20260805T212500Z/artifacts/adapter`
- Train loss: 1.1782 -> 0.6638; validation loss: 1.7302; peak allocated memory: 13.10 GiB.
- Sealed robust eval: 0/64 deterministic variant failures; 1/40 sampled failures before retry, 0/40 after retry; no privacy, unsupported-action, or service-leakage failures.
- New sealed trajectory gate: **5/5 trajectories, 11/11 turns passed**.
- Current DPO on the same trajectory gate: 3/5 trajectories passed; it missed a reschedule response term and leaked service language during the casual pivot.
- Candidate has lower prompt-variant uniqueness than current DPO (0.5625 vs 1.0), so it is promising on grounding/context but still needs a style/diversity decision before promotion.
- Current DPO remains served on GPU 0; candidate is not promoted.

Evaluation artifacts:
- `runs/voice-qwen35-9b-grounded-v1-20260805T212500Z/artifacts/robust_eval.json`
- `runs/voice-qwen35-9b-grounded-v1-20260805T212500Z/artifacts/trajectory_eval.json`
- `runs/voice-qwen35-9b-grounded-v1-20260805T212500Z/artifacts/current_dpo_trajectory_eval.json`

## Constraints
- No destructive actions; preserve all adapters, checkpoints, manifests, and first failed generation run.
- Do not use the current DPO adapter as the synthetic teacher for this corrective corpus unless a controlled comparison specifically tests it.
- Keep raw SMS out of artifacts and logs. Aggregate target-shape statistics are safe.
- Current unrelated worktree changes exist; do not commit them automatically.
