# Observables: candidate measurement metrics and kill-test status

> Update when: a kill test resolves a metric, or a new observable is
> proposed. Each observable carries a status. These are **diagnostics,
> never rewards** (`REWARD_POLICY.md`). The honest test is within-prompt
> separation, not pooled — pooled confounds with prompt difficulty.

## Status key

- `candidate` — proposed, not yet tested
- `supported` — within-prompt sign-consistent across mixed prompts
- `killed` — sign-flips within-prompt (pooled correlation was a composition artifact)
- `underpowered` — test ran but n is too thin to rule

## Kill-test protocol

Pooled rank-biserial (correct vs wrong) is a **first look, never a
verdict** — it confounds with prompt difficulty. The keeper test is
**within-prompt**: rank-biserial per prompt (correct vs wrong), then
sign-consistency across mixed-outcome prompts. If signs flip, the metric
does not generalize even within one family, and is killed.

### Killed (sign-flips within-prompt)

| Observable | Evidence |
|---|---|
| `ent_late` (mean entropy, final third) | pooled +0.63 was composition; within-machine +0.037; sign-flips across mixed prompts |
| `ent_mid` | sign-flips within-prompt |
| `recur_density` | sign-flips within-prompt |
| `n_tokens` (length) | known difficulty-composition confound; length tracks hardness, not per-sample search quality |
| `step_L15` (current hook layer) | weakest mid-layer site (+0.149 pooled); killed within-prompt |
| `step_L17` / `step_L31` | sign-flips |
| `frob_rec_2` / `rec_9` / `rec_12` | sign-flips (despite strong *pooled* separation — composition again) |
| `branch_ledger: opens/waits/elims` | Simpson's paradox on `opens`: −0.40 pooled, +3.0-vs-0.7 within assign |

### Supported (machine, 3 mixed prompts — provisional)

| Observable | Direction | Evidence (p09 / p10 / p12) |
|---|---|---|
| `ent_early` (mean entropy, first third) | wrong = HIGHER early entropy (premature commit) | −0.62 / −0.20 / −1.00 |
| `ent_trend` (late − early) | correct rises, wrong falls | +0.62 / +0.20 / +1.00 |
| `tortuosity` | correct less tortuous | −0.75 / −0.40 / −0.33 |
| `step_L6` (step-cosine-dist, early) | correct moves less early | −0.25 / −0.20 / −0.33 |
| `step_L23` / `step_L25` (late-mid) | correct moves more late-mid | +0.25 / +0.80 / +0.33 |
| `frob_rec_14` / `frob_rec_18` | correct smaller recurrent state | −0.38 / −1.00 / −0.33 |

### Candidate (untested or uncalibrated)

| Observable | Note |
|---|---|
| `branch_ledger: verify_dup` | detector uncalibrated against hand-labeled case; flat result may be instrument failure |

## The two surviving patterns (provisional, machine only)

1. **Wrong traces start more confident and get more confident; correct
   traces start less certain and stay open or get more uncertain.** This
   is premature convergence with within-prompt evidence, not just a
   hypothesis. See `FAILURE_MODES.md` §premature-commitment.

2. **Trajectory shape is layer-dependent.** Correct traces move LESS at
   L6 (early) but MORE at L23/L25 (late-mid). They consolidate early, then
   keep exploring late. Wrong traces do the opposite — churn early, lock
   in late. Recurrent state at L14/L18 is smaller for correct traces —
   less accumulated confusion in compressed memory.

## Caveats on the whole table

- **One family, 3 mixed prompts, 2–5 wrong each.** Sign-consistency across
  3 independent prompts is real but thin. Provisional keep, not confirmed.
- `ent_late`'s death is machine-specific — it might survive as a heuristic
  on families where it held pooled, but it is **not a within-prompt signal
  and should not be trusted as one**. If it can't separate same-problem
  correct from wrong, it can't guide a per-decision controller.
- All survivors must replicate on grid/hypothesis/adversary before being
  treated as general. Cross-family replication is the next data need.

## Loop-onset separability (separate test, underpowered)

Linear probe (pre-onset vs matched-healthy mid-trace states) returned
AUROC ≈ 0.25 (below chance) across all 32 layers — **degenerate because
only 4 loop traces** have usable onset offsets against 128 healthy
controls. This is a data shortage, not a method failure. Re-run when more
loop traces exist (hard-preset regeneration, or the H200 harvest).
*(Annotation 2026-08-26: the 0.25 itself was deflated by the auroc()
normalization bug — the n=4 shortage stands, the number does not. The
re-run happened via the annotation sidestep instead of regeneration: see
the position-level section below and STATUS.md R12.)*

## Position-level annotation-derived observables (2026-08-26, re-scored 2026-08-27)

The annotation sidestep (`ANNOTATION_SIDESTEP.md`) supplies span labels
(LLM-annotated muse/cycle/loop episodes; corpus now xsub 96 + b2 184
traces, 231 teacher-forced captures, 756 resolved pass1 spans); the
observable is the residual at the span's onset position vs depth-matched
within-trace controls, scored by leave-one-trace-out AUROC. These are
position-level — prompt difficulty and trace noise cancel within-trace
by construction, unlike every trace-level observable above.

| Observable | Status | Evidence |
|---|---|---|
| `cycle_onset_vs_controls` | **supported** | merged LOO AUROC 0.989–0.992, n=291 onsets (L16–L19, best L18); xsub-only was 0.988–0.991 n=128 |
| `loop_onset_vs_controls` | **supported** | merged LOO AUROC 0.975–0.978, n=285 onsets (L2–L6, best L2); xsub-only was 0.983–0.988 n=78 |
| `muse_onset_vs_controls` | **supported** | merged LOO AUROC 0.947–0.952 conf=clear n=13 (L16–L19); 0.967–0.968 conf=all n=29 — no longer n=5, still the smallest class |
| `pre_onset_vs_controls` | **supported** | one token before the span: cycle L27–L30 0.985–0.986, loop L26–L29 0.977–0.980, muse L22–L29 0.95–0.96 |
| `onset_succ_vs_fail` (escape vs reheat) | **supported** | loop 0.86 (stable vs xsub 0.856); cycle 0.72 (deflated from xsub 0.78 by the larger corpus); muse 0.957 conf=all (23v6) |
| `lead_time` (lookback before onset) | **supported** | LOO AUROC 0.97 at 2 tokens pre-onset, 0.86 at 4, 0.74 at 8, chance from 32 — sharp flip over the last ~8–16 tokens; earliest reliable intervention point ~4–8 tokens upstream of onset |
| `substrate_transfer` (K5) | **supported** (cycle/loop) | q8-fit directions score cycle 0.995 (L18) and loop 0.985 (L2) on bf16 captures; muse untested (bf16 muse n<10, STATUS Q12) |
| `recurrent_onset_vs_controls` | **supported** (auxiliary) | DeltaNet states: within-trace separable everywhere; LOO best rec_L2 — cycle 0.892, loop 0.859, muse 0.710 (n=9); weaker than residual channel |

Caveats: observational (teacher-forced); onset AUROCs are broadly high
across layers — either distributed signal or residual position confound;
the escape-vs-reheat comparison is the cleaner reading. The merged corpus
held cycle/loop onset where xsub put them and deflated the cycle
escape-vs-reheat reading — the small-corpus numbers were not inflated
beyond their real level except there. Record:
`lab_notes/2026-08-26-scale-annotation-to-r3.md` (xsub),
`lab_notes/2026-08-27-overnight-b2-collect-annotate.md` (b2 corpus),
`lab_notes/2026-08-27-lookback-k5-rec-results.md` (lead time, K5, rec
channel).

## The trajectory-preserving cut (steerability boundary)

From the legacy stratifier, ported to `TRAJECTORY_VOCABULARY.md` §requires-
new-reasoning. A trajectory is a **trajectory-preserving-editing** target
only if full repair does not require reasoning absent from the source.
The cut separates traces fixable by re-expression from traces that need
new reasoning.

Applied to the prospective experiment as a **steerability boundary**: a
trajectory that needs new reasoning is arguably not a steering target
either — you cannot nudge toward reasoning that isn't there. This is a
*candidate* cut on collected traces, not yet applied. It matters because
it bounds what the controller can be expected to recover: a nudge can
reduce reconstruction, re-verification, or re-narration (the reasoning is
present, just re-expressed), but it cannot invent a missing derivation or
a missing strategy change. Status: `candidate` — not yet measured as an
observable, but a conceptual cut that should be applied to the b1 pool to
see how much of the failure mass is steerable vs. needs-new-reasoning.

## Candidate observables not yet measured

These are proposed in `HYPOTHESIS.md` / `lab_notes` but not yet tested at
scale.

**Token entropy** — next-token uncertainty, the baseline observable:

```
H_tok(t) = -Σ_v p(v | x_≤t) log p(v | x_≤t)
```
Useful as a baseline but is not equivalent to semantic search width (many
lexical alternatives can express the same reasoning move). Measured in the
b1 scan; `ent_late` killed, `ent_early`/`ent_trend` supported (machine).

**Semantic branch entropy** — conceptually closer to "search width": sample
short continuations from a prefix, cluster by reasoning strategy, take
entropy of clusters:

```
H_branch(t) = -Σ_c P(c) log P(c)      effective branch count = e^{H_branch}
```
Clustering method and measurement infrastructure remain experimental. Not
yet measured at scale.

**Semantic return-after-elimination** — in representation space, not
string space. Word-8-gram recurrence is blind to the phenomenon — reopening
is paraphrastic, not lexical (zero verbatim returns in 21 machine traces
despite visible semantic returns). Next instrument to build.

**Trajectory tortuosity**:

```
T = Σ_t d(h_t, h_{t+1}) / d(h_0, h_T)
```
High tortuosity is not inherently bad (productive search can produce long
paths); the hypothesis is tortuosity + recurrence + no branch-entropy
reduction = thrash (`HYPOTHESIS.md`). `tortuosity` supported (machine) as a
standalone within-prompt signal; the compound signature is untested.

**Hidden-state dispersion / effective rank** across continuations from the
same prefix — could indicate how broadly the model is exploring
representational space; must be tested rather than assumed to have an
intrinsically semantic interpretation. Not yet measured.
