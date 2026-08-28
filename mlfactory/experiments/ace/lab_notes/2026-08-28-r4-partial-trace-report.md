# 2026-08-28 — R4 partial-trace report (117 rows, trace-read)

> Descriptive report on the partial traces from the halted R4 run, read
> directly (principal authorized reading traces, not only computed
> metrics). Scope: what the 117 rows contain and what the matched noop /
> toward_healthy pairs show — especially in light of the redesign
> (fork → 2048-token rollout → LLM-judge of three branches). This is a
> **description of evidence, not a verdict** on the steering hypothesis;
> that verdict needs `toward_diverge` and the new design. Predecessor:
> `2026-08-28-r4-attendance-stopped-design-change.md` (halt + recipe).

## Coverage (lopsided, as principal noted)

117 rows, 4 states — **all CYCLE class, layer 18 (the CYCLE focal),
lam 2.0138**, same verifier (`gen_strict_v2`). The states differ only
in **fork position** (where the intervention point sits along the base
trace):

| state | fork_token | noop (n) | toward_healthy (n) | toward_diverge |
|---|---|---|---|---|
| r4_cycle_00 | **17580** (longest) | 24 (full) | 18 | — |
| r4_cycle_01 | 5167 | 20 | — | — |
| r4_cycle_02 | **2533** (earliest) | 24 (full) | 2 | — |
| r4_cycle_03 | 9768 | 24 (full) | 5 | — |

**25 matched (state, seed) pairs** have both noop and toward_healthy
(r4_cycle_00: 18, r4_cycle_03: 5, r4_cycle_02: 2). **`toward_diverge`
never ran** — the third branch of every triplet is absent. r4_cycle_01
has noop only. All four programs were working CYCLE states first (the
shard orderings list CYCLE before LOOP/MUSE), so the partial data is a
deep slice of one class, not a thin slice of all 27.

All problems are state-machine / counter tasks: a small transition
system (modes M0–M3 or event/state tables) and a quantity to track
(commands driving a credit counter negative; an event log to simulate,
counting rejections). They are reasoning-heavy, multi-step, and
verification is strict on the final answer line.

## The decisive finding — branches diverge immediately, but terminal outcome is noise

For all 25 matched pairs I computed where the two completions first
differ. **The branches diverge almost immediately after the fork:**

| | first-differing **token** after fork |
|---|---|
| min | 0 |
| median | ~24–80 |
| max | 154 |

So the steering **is mechanically perturbing the residual stream from
the very first sampled token** — the intervention does something at
the intervention point. That is the signal the redesign wants to
capture.

But that immediate divergence does **not** cohere into a terminal
correctness signal. On the 25 matched pairs:

```
both correct (unchanged):  9
healthy-only (steering helped):  4
noop-only (steering hurt):  3
neither:  9
  → noop correct 12/25,  toward_healthy correct 13/25
```

The "helped" (4) and "hurt" (3) counts are within stochastic noise of
each other. **The intervention does not improve terminal correctness;
it permutes which seeds succeed.** Aggregate correctness per arm
confirms it on the deepest state: r4_cycle_00 noop 9/24 → toward_healthy
6/18 — no gain, possibly a small loss.

This is the principal's thesis made concrete in the data: **a
terminal correctness score over an 8–20k-token horizon cannot
attribute the outcome to the steering.** The signal is drowned by the
model's own long-horizon thrashing dynamics, the 8420-token cap, and
output-format extraction noise — none of which have anything to do
with whether the intervention steered.

## What the traces actually look like (read, not scored)

### r4_cycle_00 — fork at 17580 (the longest fork; intervention is very late)

This is the hardest state: 22/24 noop and **18/18 toward_healthy hit
the 8420 cap**. The model has already thought for 17580 tokens when the
fork happens; the 8420 post-fork tokens are almost entirely
**thrashing** — re-enumerating the same length-4 command candidates
("How about B, C, B, A? … How about B, A, C, A?") and re-deriving the
same M1-reachability argument over and over, never durably pruning.

- **seed 0** — noop wrong (cap, ends mid-sentence `B,A,A,A\`?`),
  toward_healthy wrong (cap, ends `" format.`). But toward_healthy is
  qualitatively cleaner: it commits to "B, A, A, A: 2,1,0,-1" early and
  states it confidently, then spends thousands of tokens
  double-checking M1 paths. Noop is genuinely lost, looping. Both hit
  the cap and botch the final line — terminal score says "tie," but
  the intervention clearly changed the character of the reasoning.
- **seed 1** — noop wrong (never commits; answer
  `<your final answer>'.`), toward_healthy **correct**. Toward_healthy
  committed; noop meandered through M1-loop analysis forever.
- **seed 5** — the cleanest illustration of the conflation. Noop
  solved it **correctly in 7783 tokens**. Toward_healthy hit the 8420
  cap **agonizing over the output FORMAT** ("Does CMDS mean commands
  or credits? … to be safe and concise … Wait, the prompt says …")
  and came out wrong. The steering pushed the model into more
  meta-discourse, not less. A terminal score records "noop right,
  healthy wrong" — and misses that the divergence was the steering
  triggering format-pedantry thrashing, not a math error.

### r4_cycle_03 — fork at 9768 (mid)

- **seed 0** — noop wrong (cap, 16232 tok, garbled answer
  `(FAULT, false, false, 8), 4, 3".`), toward_healthy **correct and
  clean** in 10209 tok (under cap): `State: FAULT, A: false, T: false,
  n: 8, Rejected: 4, First Rejected: 3`. Here toward_healthy helped
  decisively — converged to the right final-state summary; noop got
  lost in re-verification and hit the cap with a malformed line.
- **seed 2** — both correct, nearly identical traces and identical
  answer. Intervention changed little.

### r4_cycle_02 — fork at 2533 (earliest; intervention very early)

- **seed 0** — both correct (`H3 expected=$308.77 over/short=$0.00`),
  first ~2200 chars nearly identical (same arithmetic, same voided-
  entries reasoning). The fork is so early that most reasoning happens
  after it; both converge to the same right answer. Intervention had
  little observable effect.

### Cross-cutting qualitative observations

1. **The dominant failure mode is thrashing, not math errors.** Both
   arms re-derive the same enumerations and the same reachability
   arguments repeatedly without durable pruning — the exact
   "thrashing = revisits without pruning" phenomenon from
   `HYPOTHESIS.md`. The steering does not fix thrashing; it sometimes
   changes *which* thrashing.
2. **Toward_healthy is often more decisive early** — it commits to a
   candidate and states it with less hedging. But "decisive early" ≠
   "correct terminal": it then burns thousands of tokens
   re-verifying and often hits the cap anyway. The early decisiveness
   is exactly the kind of meaningful divergence the redesign's
   2048-token judge would catch; the terminal score misses it.
3. **Output-format thrashing is a valid target failure mode, not just
   verifier noise.** Several "wrong" rows are substantively correct
   (B,A,A,A: 2,1,0,-1) but formatted so the strict verifier rejects
   them (`B,A,A,A\`?`, `B,A,A,A; 2,1,0,-1` vs the canonical `CMDS:
   B,A,A,A: 2,1,0,-1`). This is not an artifact to discard — it is the
   **"solution consolidated but emission blocked"** phenomenon
   (`LAYER_HYPOTHESES.md`, currently a prior, untested, at L19/L23/L27
   + final layer), observed empirically here: r4_cycle_00 seed 5 has the
   right answer in hand and burns 8420 tokens on `CMDS: c1,c2,...`
   semantics instead of emitting it. A steering intervention that helps
   the model **stop format-pedanting and emit** is a meaningful win —
   and the redesign's LLM judge of the trace can credit it where
   terminal correctness cannot. **Format-thrashing belongs in the
   redesign's failure-mode taxonomy alongside CYCLE/LOOP/MUSE onsets**
   (distinct phenomenon; different layer signature than the L18/L2/L17
   onsets, so a separate probe may be needed if it's to be steered
   directly rather than caught as a downstream effect).

## What this says about the redesign

The data **directly supports** the principal's redesign and makes the
cost of the old design measurable:

- **The meaningful signal lives in the first ~150–2048 tokens after the
  fork.** Divergence is immediate (token 0–154) and the qualitative
  differences (decisiveness, commitment, pruning, format confidence)
  are visible in the first 1200 chars. A 2048-token rollout is enough
  to see whether the intervention did something meaningful.
- **Running to terminal drowns that signal.** On r4_cycle_00 alone we
  spent ~8420 × 2 × 18 ≈ **303k yielded tokens** to watch the model
  thrash after the intervention point — when the interesting signal
  was in the first ~150 tokens of each. Across all 117 rows the run
  yielded on the order of 1.4M tokens, the great majority of which
  are post-intervention thrashing rather than evidence about the
  steering. This is the "astronomical yield for the span right after
  the intervention point" the principal named.
- **`toward_diverge` is the missing leg.** Without it we cannot tell
  whether the immediate divergence is *directional* (steering toward
  healthy diverges one way, steering toward diverge diverges the
  other) or merely *any perturbation changes the stochastic
  outcome*. The new design's three-branch judge is the right
  instrument; the partial data only has two branches.
- **Format-thrashing / emission-block** should be a first-class
  failure mode in the new judge rubric (see cross-cutting #3): a
  triplet where the model has the answer but the noop/thrashing
  arm can't emit it, and toward_healthy unblocks emission, is a clear
  positive for the steering — invisible to terminal correctness.
  The partial data has at least one such case (seed 5).

## Loose ends (for the redesign, not blocking)

- **`substrate` field reads `q8`** on every row, while the binding
  context (`OPERATIONS.md`) says precision is bf16 and "q8 is a
  different model." This is likely a label for the steering substrate
  (residual-space basis) rather than model precision, but it should
  be confirmed against `fork_r4.py` before the redesigned run so the
  new design doesn't inherit an ambiguous field.
- **Verifier format-strictness** (above) will matter under the new
  design too if terminal correctness is still scored at all; if the
  new design judges the 2048-token *trace* by an LLM rather than
  scoring a final answer line, this noise largely drops out — another
  point in favor of the redesign.
- **r4_cycle_00 (fork 17580)** is the state where the model is
  deepest in thrashing when the fork fires. If the redesign wants to
  test "intervention at a moment the model is productively
  reasoning," the fork positions should probably be chosen relative
  to per-trace structure, not at a fixed deep token — the deepest
  forks here are mostly intervening into already-thrashing traces.

## What is preserved for the redesign

- The 117 rows: `data/r4fork-{a,b}/fork_r4_results_*.jsonl` (sha256s
  in `2026-08-28-r4-attendance-stopped-design-change.md`).
- The 25 matched pairs are immediately usable as **pilot material for
  the LLM-judge** under the new design: take each pair's first 2048
  post-fork tokens, run the (absent) toward_diverge arm for the same
  25 (state, seed) points, and have the judge score the triplet. The
  existing toward_healthy traces are reusable as-is; only the
  toward_diverge rollouts and the truncation to 2048 are new work.
- Full A6000 recreation recipe: `2026-08-28-r4-attendance-stopped-design-change.md`.

## Decision

No write-back to concept/living docs from this report — it is
descriptive evidence reading, not a settled verdict. The redesign is
the principal's. When the new design runs, the findings here
(especially the immediate-divergence / terminal-noise split) should
inform the judge rubric and the fork-position choice.
