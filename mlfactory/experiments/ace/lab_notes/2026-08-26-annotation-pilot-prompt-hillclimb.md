# Annotation pilot log — probe-point prompt hill-climb

Case fixed across versions: q8 p140 s7 (success) + s1 (cap-hit failure),
prompt certify-140 (3-coloring). Framings: A=single trace (A1 success,
A2 failure), B=unlabeled pair, C=pair + compare-&-contrast. Model
glm-5.2-vision (Lunaroute default temp; no ballast variants today).
Anchor char ⟦; digests = out/<ver>/<task>.digest.txt. Quality metric:
flags fully-resolved (both quotes exact-match the trace) + my own read
of whether each flag is real.

## v1 — 2026-08-26
- First shot. Rubric classes + quote format verbatim from RUBRIC.md.
- Result: semantics strong across all four tasks (real CYCLEs on
  pruned coloring branches, real LOOPs on verification re-runs, one
  plausible MUSE on audit-order speculation). Format failures: flag
  lines wrapped across lines (A1); one quote reconstructed from memory
  (A2); generic verification quotes ambiguous.
- Harness bugs found in my own runner (not model failures): label dict
  passed where text dict needed (all resolutions silently failed at
  first); digest KeyError on end-before-start spans. Fixed; v1
  re-parsed baseline: A1 2/2, A2 1/2, B 5/5, C 2/6 resolved.
- Framing signal: pair framings surfaced more episodes than single
  (B: 5, C: 6, vs A: 2+2); C's MUSE got conf=clear vs B's probable.

## v2 — changes: single-line rule; verbatim-copy rule; quote length
4-12 words; distinctiveness rule (names/values not generic sentences).
- Result: A1 4 flags (new MUSE + probable CYCLE; tight 186ch MUSE span
  resolved). A2 3 LOOP flags — granularity improved: per-round re-verify
  loops named ("third/fourth re-verification") instead of one 20k-char
  blob. B: 4 flags all present but concatenated without newlines —
  splitter fixed parser-side (flags split on class head, not line
  start). C: 4 semantically good flags, 0/4 fully resolved — quotes
  still reconstructed on long traces. B used 31.5k output tokens
  (mostly thinking) then stopped cleanly.
- Lesson: default temp → run-to-run variance; single-run framing
  comparisons are noisy.

## v3 — changes: one-flag-per-line rule rewritten ("no other text
between or inside flag lines"); copy discipline rewritten harder
("copy character for character; if you cannot copy it exactly, do not
flag the span"); LOOP class line gains "end each span where the
episode ends". Also max_tokens 32768→65536 (thinking tokens count).
- Result (A1, A2, C completed; B blew the 65k budget,
  finish_reason=length, no content):
  - A1 4 flags, 4/4 resolved — single-trace format quality now good.
    BUT 2 of 4 are class errors: grounded discarded branches labeled
    MUSE instead of CYCLE ("branch produces another valid assignment
    but is discarded" is CYCLE by definition). Systematic confusion.
  - A2 3 flags: Brooks'-Theorem dead-end flagged MUSE (defensible:
    introduced, abandoned, no downstream effect); balanced-coloring
    CYCLE resolved at 7284ch. But no LOOP flags this run — the
    verification wall, the bluntest feature of the cap-hit trace —
    sampling variance at default temp is real.
  - C 7 flags, 2/7 resolved. Semantics: all 7 plausible (greedy-prune
    CYCLE, listing-order MUSE, F=2 CYCLE, re-verify LOOP on s7;
    problem-source MUSE, B/G-swap CYCLE, 20.7k-char re-verify LOOP on
    s1). Quote failures trace to: dropped **markdown** markers
    ("Node A: No previous neighbors" vs trace "**Node A:** No
    previous neighbors") and reconstruction ("If I picked F=2?").
  - B unstable: token spend per version 9.8k → 31.5k → 65k+ (length).
    Per user's GLM note, pair framing + accumulating strictness feeds
    the meander.
- Cross-version stability check (the good news): listing-order muse,
  greedy-backtrack cycle, and re-verification loops are flagged in
  every version and framing that ran. The blunt episodes replicate;
  class boundary and quote quality are the remaining failures.
- Decision → v4: (1) rewrite MUSE/CYCLE boundary — a grounded
  alternative that is explored and discarded is CYCLE, not MUSE;
  (2) quote line rewritten to include "as they appear, markdown
  included"; (3) retry B once — if it blows the budget again, drop
  framing B for this model and keep C as the pair framing (C is
  strictly richer anyway: 7 plausible flags, 13.8k tokens, no
  runaway).

## v4 — result
- B completed cleanly (13.5k tokens, stop): v3's 65k blowout was
  variance, not a hard regression — B stays in the comparison.
- A1 3 flags (greedy CYCLE + 2 LOOPs; no MUSE this run — muse
  detection is run-to-run unstable at default temp). A2 3 flags incl.
  two LOOPs on the verification wall (one 980ch resolved) — the blunt
  cap-hit feature is back. C richest again: 8 flags, 4/8 fully
  resolved (up from 2/7) — class-boundary rewrite held (no
  discarded-branch-as-MUSE errors; one grey-zone LOOP/CYCLE mixup on
  an aesthetic re-comparison). Both F=2 and F=Green alternative
  cycles caught separately on s7.
- Quote quality is now the weakest axis but improving per version:
  fully-resolved totals A1 1/3, A2 2/3, B 1/2, C 4/8.
- User directive → v5: relax the confidence language — flag greyzone
  spans as conf=probable instead of withholding; extra flags can be
  ignored later, lost ones cannot. Rewrite the opener + conf line,
  keep everything else (v4 is otherwise stable).

## v5 — changes: opener rewritten from "precision over recall, if
unsure do not flag" to "flag everything that might qualify; clear vs
probable marks the confidence; extras are droppable, misses are not
recoverable". conf line rewritten to match; the "do not flag
borderline cases" bullet removed from the prompt (it contradicted the
relaxation).
- Result: the relaxation worked across the board. Flag counts: A1 9
  (v4: 3), A2 4, B 11 (v4: 2), C 11. Fully-resolved: A1 6/9, A2 3/4,
  B 4/11, C 7/11. Greyzone material surfaced as intended: aesthetic
  "nicer" musing (probable MUSE), partition-independence re-verify
  (probable LOOP), symmetric-coloring branch (probable CYCLE) — all
  droppable later, none lost. Span granularity excellent on the
  cap-hit trace (556/317/265/724ch tight LOOPs + the 17.7k wall).
  Both exam-venue/proctor muses caught on s7. One empty basis field
  in B (minor). Token spend sane: 8.6k–19k per task, all finish=stop.
- Verdict: mechanics stable enough to stop hill-climbing on this
  case. Quote resolution remains the weak axis (B 4/11 worst) but
  unresolved flags keep their quotes for manual review — nothing is
  lost. Next: generalization test on a new prompt family (p53
  adversary: s0 success + s6 cap-hit) before declaring a
  prompt/framing winner.

## v5 generalization — new case: p53 adversary (s0 success 25k tok, s6
cap-hit 26k tok), different problem family (command sequences / mod
states instead of graph coloring)
- A1 7 flags / 3 fully-resolved: length-3-branch-elimination CYCLE
  (12k ch resolved), length-4 alternatives, BCCC/CCCC recompute LOOPs —
  semantics transfer to the new family.
- A2 3 flags / 0 fully-resolved but the catch of the run: probable
  MUSE — "the prompt never mentions a vendor; the solver fabricates
  this claim" — a fabricated-context muse on the adversary trace,
  exactly the species the adversary family exists to produce.
- B 11 flags / 0 fully-resolved — resolution failure is a RESOLVER
  limit, not annotation noise: these traces re-check the same
  sequences repeatedly, so standalone quotes are genuinely ambiguous.
- C hit finish_reason=length at 65536 (no content, 1088 s) — second
  budget blowout in the pilot (first: v3-B on p140). Pair-framing +
  long thinking = occasional runaway; retry once, then treat as a
  known mode.

## Harness changes this round
- resolve_span: quotes now resolved as an ORDERED PAIR (end must occur
  after start; first valid pairing wins; whitespace-normalized
  fallback retained). Standalone-quote ambiguity was losing correct
  flags on repetitive traces.
- outdir now case-encoded: out/<ver>_p<pid>s<succ>s<fail>.

## Incident — p140 v5 outputs overwritten (self-inflicted)
The p53 generalization run wrote to the shared out/v5/ directory (old
outdir logic keyed on prompt version only) and overwrote the p140 v5
raw responses and digests. Flag counts, resolution stats, and per-flag
bases for p140 v5 survive in this note and the session record; raw
model text did not. Recovery: p53 outputs preserved as
out/v5_p53s0s6; p140 v5 re-run (out/v5_p140s7s1); p53 C retried. Root
cause fixed (case-encoded outdir). Lesson recorded: any output path
that two runs can share must encode every input that varies between
runs — same principle as collection row keys.

## Recovery re-runs + rulings
- p140 v5 restored (out/v5_p140s7s1; fresh sample at default temp):
  A1 5 flags/3 resolved, A2 5 flags/0 (five LOOPs), B 12/4, C 12/6 —
  pattern matches the lost run (C richest + best-resolved; counts vary
  run-to-run). All finish=stop.
- User ruling: thinking-wall hits are FINE — Lunaroute is not
  pay-per-token; a dud or a long run costs nothing. Stop treating
  finish_reason=length as a failure mode to engineer around; retry
  once, keep whatever lands.
- Probable-tier audit launched: 16 conf=probable flags across v5
  outputs judged by three parallel subagents (qwen/qwen3.7-plus)
  against the rubric + trace text. Manifest:
  annotate/pilot/out/probables_check.json. Results below when they
  land.

## Probable-tier audit results (three qwen/qwen3.7-plus subagents,
judging each conf=probable flag against rubric + trace text)
- 16 probable flags audited; all 11 unresolved-quote flags were
  locatable in the traces (0 UNFINDABLE).
- 14/16 point at real episodes. Two failures:
  1. v5_p140s7s1/A2#1 — flagged LOOP, actually a CYCLE (derives a
     genuinely different alternative coloring, then discards it). Real
     episode, wrong class.
  2. v5_p53s0s6/A2#2 — flagged MUSE ("prompt never mentions a
     vendor"), but the auditor checked gen/adversary.py and the vendor
     line IS prompt content. False fabrication claim — the annotator
     misremembered the prompt.
- Strict verdict: 13/16 correct-class real episodes (81%); 14/16 real
  episodes of some class (87.5%).
- Consequence: the v5 relaxation is validated — these 13 episodes were
  flagged only because the prompt stopped withholding greyzone spans;
  under v4's "if unsure do not flag" they would have been lost. The
  probable tier is high-signal, filterable later, worth keeping.
- Method note: the prompt itself (gen/ source) is ground truth the
  annotator can misremember — muse judgments rest on it, so at scale
  the annotation task should keep the prompt text adjacent to every
  trace (it does: surface_question is included in the task).

## Pilot verdict (hill-climb converged)
v5 + case-encoded outputs + ordered-pair quote resolution is the
working annotation protocol: mechanics stable across 5 versions and 2
problem families, blunt episodes replicate across framings and runs,
C (compare & contrast) richest and best-resolved, probable tier
audited high-signal. Ready to scale from 2 pilot traces to the wider
corpus — next session's starting point.
