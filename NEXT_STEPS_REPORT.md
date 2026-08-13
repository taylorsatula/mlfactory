# Qwen3.5-9B Voice Model: Next-Steps Report

Date: 2026-08-05

## Executive assessment

The grounded-v1 candidate is worth advancing through a stronger release evaluation, but it should not replace the current DPO adapter yet.

Candidate:
`/home/admin/mlfactory/runs/voice-qwen35-9b-grounded-v1-20260805T212500Z/artifacts/adapter`

Current production comparison:
`/home/admin/mlfactory/runs/voice-qwen35-9b-robust-dpo-v3-20260806T0600Z/artifacts/policy_adapter`

The candidate completed 400 QLoRA steps in 25.6 minutes. It reached a final training loss of 0.6638 and reported validation loss of 1.7302. It passed all 64 deterministic outputs in the small sealed robust evaluation, had one soft sampled failure in 40 samples before retry and none after retry, and passed all 11 turns in five scripted trajectories. The current DPO had two deterministic service-leakage failures in the robust evaluation and failed two scripted turns.

These results are directional, not yet conclusive:

1. The trajectory suite contains only five trajectories and 11 turns.
2. One current-DPO trajectory failure was a brittle keyword miss despite a semantically reasonable answer. The casual-pivot service leakage was a genuine regression.
3. Forty-two replay examples were included in both training and evaluation. Therefore, the training run's validation loss is not a fully uncontaminated capability holdout. The separate sealed robust and trajectory evaluations did not train on their exact cases and remain useful.
4. Candidate prompt-variant uniqueness was 0.5625 versus 0.71875 for current DPO. Sampled uniqueness was 0.95 versus 1.0. This is a modest style/diversity warning, not by itself a valid reason to reject the candidate.
5. The training mixture was synthetic-heavy: 1,668 grounded teacher records, 472 local real records, and 136 authored replay records after repeats. Template concentration must be measured before another training run.

**Recommendation:** freeze the candidate, repair the evaluation split, broaden blinded and scripted comparisons, then make a release decision. Do not start another extended training run until those measurements identify a concrete remaining defect.

## Priority 1: Repair measurement integrity

### 1.1 Separate replay training and replay evaluation

Current behavior in `train_robust_voice.py` adds the same replay list to both `train_records` and `eval_records`. In the grounded source, all eight generated casual replay examples came from the synthetic eval split. The merged authored replay set was also reused on both sides.

Planned changes:

- Update `build_robust_voice_data.py` and `filter_grounded_voice.py` to emit explicit `replay_train.jsonl` and `replay_eval.jsonl` files.
- Add a backward-compatible `--replay-eval-file` argument to `train_robust_voice.py`.
- Train only on replay-train; evaluate only on replay-eval.
- Make the trainer compute and record example-ID, scenario-ID, and normalized-context overlap. Fail preparation if train/eval example or scenario overlap is nonzero.
- Replace the inaccurate manifest statement `0 by construction` with measured overlap counts and hashes.
- Add unit tests for split disjointness and manifest reporting.

This does not require retraining the existing candidate immediately. First calculate clean held-out losses for base, current DPO, and grounded-v1 using a corrected, never-trained replay holdout.

### 1.2 Keep privacy filtering practical

Synthetic plans are fictional, so ordinary mentions of Venmo, Zelle, PayPal, or the phrase “door code” should not be treated as PII by themselves. Thirty-six otherwise useful candidates were rejected by the broad `payment_or_secret` pattern.

Planned simplification:

- Continue rejecting actual email addresses, phone numbers, credential-shaped values, unresolved markers, and model wrappers.
- Allow ordinary payment-method names and discussion of access codes when no value is supplied.
- Do not add a synthetic pseudonymization subsystem; reject the rare actual secret-shaped output instead.
- Retain the existing local, in-memory redaction/exclusion policy for real SMS without copying raw messages into artifacts.

## Priority 2: Expand the sealed regression suite

Expand `evaluate_voice_trajectories.py` from 5 trajectories/11 turns to approximately 30–40 trajectories and 100–150 turns. Cases will remain fictional and sealed from training generation.

Required families:

- Unknown and known availability
- Booking requests without booking authority
- Rescheduling across several turns
- Cancellation requests and confirmation boundaries
- Weather or operational delays
- Quote/scope clarification
- Invoice and payment questions without invented amounts
- Arrival/access logistics
- Complaint recovery and follow-up
- Ambiguous references and change of mind
- Repeat-customer context
- Explicit casual pivots
- Casual pivot followed by return to business
- General-question capability retention
- Long-thread summarization and stale-topic resistance

Scoring changes:

- Separate hard failures from soft quality findings.
- Hard checks: unsupported actions, invented facts or amounts, identity deception, sensitive-value leakage, and service-thread leakage after an explicit pivot.
- Soft checks: missing acknowledgement, excessive limitation language, repeated questions, verbosity, canned openings, weak empathy, or failure to use supplied context.
- Replace brittle single-word requirements with concept groups and acceptable paraphrases.
- Persist raw and serving-guarded scores separately. Serving retries must not hide raw model regressions.
- Add tests proving that reasonable paraphrases such as “What day would you like to try?” pass a rescheduling concept check.

Proposed hard gate: zero hard failures in deterministic trajectories and at least 98% raw hard-pass rate under sampling. Guarded output must have zero hard failures.

## Priority 3: Measure style and diversity correctly

Prompt-variant uniqueness is too small and noisy to be a release gate. Semantically equivalent system prompts are expected to produce some identical concise replies.

A dedicated diversity report should compare base, current DPO, grounded-v1, and guarded grounded-v1 across the same contexts and seeds.

Measurements:

- Response word count: median, p90, and outliers
- Exact and normalized duplicate rates across different contexts
- Three- and five-token opening concentration
- Repeated-question rate within trajectories
- Semantic template clustering by scenario family
- Limitation/refusal phrase frequency
- Context-fact retention
- Unsupported semantic-claim frequency
- Casual warmth and business directness as separate dimensions

Use 30–50 contexts with five seeds each. Blind a stratified sample of roughly 50 candidate/current-DPO pairs for human preference review. The reviewer should choose among candidate, current, tie, or both bad, and label the reason without seeing model identity.

Tentative soft release targets:

- Business median near the real-channel median of 19 words
- p90 below roughly 55 words unless the case explicitly requires detail
- No repeated availability question after the customer has already answered it
- No single canned opening dominating unrelated scenario families
- Candidate pairwise preference win rate above 55%, with no hard-gate regression

## Priority 4: Run controlled self-play, with simulator realism scored separately

After scripted gates pass, run a small controlled matrix with Laguna S 2.1 and XS 2.1:

- 8–10 representative scenarios
- Current DPO and grounded-v1
- Raw model and guarded serving path
- Fixed simulator personas, turn limits, and stop handling

Use the reusable `SelfPlayRunner`, `BatchJudge`, and `CorpusOverseer`. Produce two independent scores:

1. Subject quality: grounding, progression, context retention, repetition, action honesty, and pivot behavior.
2. Simulator realism: coherent customer behavior, adherence to private state, realistic SMS length, and stop-sequence quality.

Do not blame the subject model for simulator failures. Keep simulator private state out of subject prompts and persisted trajectories.

## Priority 5: Make a release decision before training again

### Promote grounded-v1 if

- It passes the expanded hard gates.
- Clean holdout capability is no worse than current DPO by a material margin.
- Blinded review prefers or ties it on style while confirming its grounding advantage.
- Controlled self-play shows lower repetition/service leakage without new unsupported claims.

### Run a targeted correction only if a specific residual remains

If grounding is strong but style is too templated, do not repeat the full 400-step SFT. Build a small preference set focused on:

- Concise answers over limitation monologues
- Direct use of supplied dates/preferences
- Nonrepetitive follow-up questions
- Warm casual pivots without returning to service language
- Honest cancellation/rescheduling boundaries

Run a 30–40 step DPO smoke from grounded-v1 at a conservative learning rate, evaluate it, and extend toward at most the existing 120-step scale only if the smoke improves the named defect. Do not use the same candidate as its own synthetic judge/teacher.

If the clean evaluation instead shows synthetic-style overfitting, regenerate or rebalance the SFT mixture before any DPO pass. Reduce repeated teacher records and increase diverse real/authored replay rather than merely increasing total steps.

Any extended run will use `nohup` and receive a configured CLI dashboard before launch, as requested.

## Priority 6: Staged serving and rollback

If grounded-v1 passes release review:

1. Serve it temporarily on physical GPU 1 at a separate port while current DPO remains on GPU 0/port 3093.
2. Run identical paired requests against both endpoints.
3. Save aggregate and fictional evaluation artifacts only.
4. Update the default adapter/model ID only after the paired canary passes.
5. Preserve the current DPO path and a one-command rollback procedure.
6. Re-run health, identity, unsupported-action, booking, cancellation, casual-pivot, and general-capability probes after the swap.

No adapter will be overwritten or deleted.

## Priority 7: Code and artifact hygiene

Before promotion:

- Run the complete test suite.
- Add tests for clean splits, semantic trajectory predicates, and release-gate aggregation.
- Rebuild `release_gate.json` from versioned evaluators rather than hand-combining reports.
- Include evaluator/source hashes and adapter checksums in the final decision artifact.
- Commit only validated voice changes in logical commits, leaving unrelated dashboard/manifest work untouched unless required by this pipeline.
- Suggested commit separation:
  1. serving safety and tests;
  2. grounded generation/filtering and clean data splits;
  3. scripted/self-play evaluation and release reporting;
  4. dashboard/runner observability changes.

## Immediate execution order

1. Freeze and retain grounded-v1 and current DPO unchanged.
2. Fix replay train/eval separation and add overlap tests.
3. Build a clean replay holdout and compare held-out loss across base/current/candidate.
4. Expand the trajectory suite and run raw/guarded comparisons.
5. Produce the diversity report and blinded pair sample.
6. Run the small controlled Laguna matrix only after scripted gates pass.
7. Write a final promote/hold/correct decision report.
8. Train again only when that report identifies a narrow, measurable residual.

## Planned decision artifacts

- `data_split_audit.json`
- `clean_holdout_comparison.json`
- `expanded_trajectory_eval.json`
- `diversity_style_report.json`
- `blinded_pair_review.json`
- `controlled_selfplay_report.json`
- `final_release_decision.json` and `final_release_decision.md`

Current operational state remains safe: current DPO is still served on port 3093, grounded-v1 is preserved but not promoted, and no new training job is scheduled.
