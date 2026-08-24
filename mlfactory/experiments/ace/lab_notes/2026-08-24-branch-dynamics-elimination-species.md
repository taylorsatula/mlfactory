# Lab note — 2026-08-24 — Branch dynamics: elimination species and paraphrastic reopening

Scope: first branch-resolution measurement pass over the b1 probe traces
(clean finishes only; crashouts excluded by instruction). Instrument:
`branch_ledger.py` (text-marker episode segmentation + ledger features).
Question carried: what does *successful* search look like at branch
resolution — opening, closing, durable pruning, reverification behavior?

## Findings (claim + evidence)

1. **Success is not characterized by less churn.** Correct clean traces
   verify as often as wrong ones (25.1 vs 25.2 episodes/trace), re-verify
   *more* within assign (verify_dup rank-biserial +0.83/+0.86 on p01/p07),
   and show comparable "wait" rates within machine (10.1 vs 11.8 per 10k
   chars). Direct trace reading concurs: the p25 certify success re-checked
   the same edge list 5+ times post-solution and escaped; machine failures
   re-simulate identically and never escape. **The distinguishing feature of
   success is the exit, not the absence of the loop.** Any Phase-1 metric
   premised on "good reasoning churns less" is measuring the wrong thing.

2. **The pooled "explicit elimination" signal (+0.80 rank-biserial) was a
   composition artifact — and its dissection produced a better hypothesis.**
   Correct traces average 18.4 elimination statements vs 2.6 for wrong, but
   within machine (the only family with real clean-wrong mass post-fix)
   correct traces show `elims=0` uniformly. Certify solutions speak
   constraint-propagation natively ("C cannot be red because A is red"), so
   the pooled signal was family language, not search dynamics.

3. **Elimination splits into two semantic species with opposite outcome
   links.** Reading every machine elimination context:
   - *Deductive*: "X cannot be Y because <constraint on the live state>" —
     prunes the space the search actually occupies; success-linked (certify).
   - *Counterfactual*: "if X had been / unless X / even if X" — excursions
     into states the trajectory never visits (e.g., p09_s4 wrong: "If Event 2
     RESUME was accepted?... Unless n was initialized to 1?"; p11_s2 wrong:
     "PAUSED is unreachable"). Off-path expansion **cannot produce durable
     pruning** because the explored region is disjoint from where the
     trajectory will ever go; failure-linked (machine: all elimination-bearing
     machine traces are wrong).
   Refined thrash hypothesis: the poison is not tortuosity or recurrence but
   **expansion into states disjoint from the live trajectory**.

4. **Reopening is paraphrastic, not lexical.** Direct test: after each
   elimination episode, search for verbatim return of the eliminated
   fragment — **zero events in 21 machine traces**, despite visible semantic
   returns when reading ("PAUSE cannot be fired in FAULT" → twenty lines
   later "but if the state were FAULT, PAUSE's allowed list..."). This
   retroactively explains both earlier nulls: word-8-gram recurrence
   (0.073 correct vs 0.066 wrong — no separation) and flat verify_dup.
   **String-space instruments are blind to the target phenomenon; the
   instrument must live in representation space.**

5. **Lexical markers lean consistently but weakly, with a meaningful shape.**
   Failure-leaning: `again` (−0.58), `unless` (−0.58), `but` (−0.33),
   `what if` (−0.20) — *additive hedges* that spawn sibling branches while
   keeping the original alive. Success-leaning (pooled): `actually` (+0.42),
   `however` (+0.40), `just to` (+0.42) — *decisive replacers* that overwrite
   state. `wait` is neutral within machine (−0.11): reconsideration is
   neither good nor bad; what follows it is. Within-machine n is thin
   (6 correct / 15 wrong); directional, not established.

## Decisions with rationale

- **Crashouts excluded from branch profiling.** Loops/truncations are a
  separate failure class (emission pathology) with its own instrument;
  mixing them into branch-dynamics stats conflates the two failure modes.
- **Trace-level scalar aggregates (thirds-of-trace entropy etc.) demoted to
  exploratory.** The doctrinal unit is the branch/episode; trace-level
  averages provably confound with family composition (Simpson's paradox on
  `opens`: −0.40 pooled, +3.0-vs-0.7 within assign).

## Caveats / confounds on record

- Post-verifier-fix, certify has **zero clean-wrong traces** — all certify
  failures are crashouts. Clean-wrong ≈ machine + sparse assign. Pooled
  clean-finish statistics are near-pure family-composition statements.
- verify_dup detector (400-char episodes, 3-gram Jaccard ≥ 0.5) is
  uncalibrated against a hand-labeled case; its flat result may be
  instrument failure rather than phenomenon absence.
- Marker rates are discourse-level English, not model-internal events.

## Next instruments (queued)

1. **v2 segmentation in representation space**: change-point detection on
   the stored layer-15 hidden states (`data/teacherforced_b1/*.npz`), embed
   elimination episodes, measure **semantic return-after-elimination** as
   the reopening metric. If it separates outcomes within-prompt, it is both
   a Phase-1 observable and a Phase-3 fork trigger.
2. **Species classifier** for eliminations (subjunctive markers vs.
   causal-binding markers) to test the two-species split at scale.
3. Replication on grid/hypothesis/adversary when probe rows land —
   hypothesis is the family's native elimination structure and is the
   decisive test of whether "explicit on-path elimination" generalizes.
