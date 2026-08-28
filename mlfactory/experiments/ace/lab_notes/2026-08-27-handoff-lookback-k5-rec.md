# HANDOFF — 2026-08-27 — lookback capture inflight; K5 + rec-channel done

> **SUPERSEDED same day by
> `2026-08-27-handoff-detection-complete-r4-pending.md`** — the recapture
> described as inflight here COMPLETED cleanly; read that handoff instead.
>
> Mid-work handoff (principal at breakfast; context window compacted
> after this). State: the lookback recapture is RUNNING; K5 transfer
> test and recurrent-channel probe are DONE with results below. When
> the recapture finishes, run the lookback probe and interpret the
> lead-time curve per the guide here. Predecessor handoff:
> `2026-08-26-handoff-r0-r3-complete-forks-pending.md` (R4 forks still
> held; decisions listed at the bottom).

## objective_and_constraints

```yaml
objective: |
  Three "free detection" investigations on the merged xsub+b2 corpus
  (280 traces, 231 captures), all scenario-A cost, no rental spend:
  1. LOOKBACK (t-k before onset): how far back is divergence readable?
     (intervention lead time; audits the t-1 pre_onset result for
     local-context anticipation; drift shape — flip vs gradual slope)
  2. K5 transfer test (pre-registered kill condition 5): directions fit
     on q8 captures evaluated on bf16 captures and vice versa
  3. Recurrent-channel probe: DeltaNet states at REC_LAYERS scored for
     onset-vs-controls (captured since R1, never scored)
bindings:
  - no rental spend (R4 forks remain held, principal's call)
  - existing capture dirs are immutable evidence — lookback work goes
    to NEW dirs (annot_captures_xsub_lb / annot_captures_b2_lb)
  - lb (lookback) kinds are residual-channel only; rec_keep semantics
    unchanged (onset + onset-anchor controls)
```

## world_state (verify before acting)

```yaml
running:
  LOOKBACK RECAPTURE (nohup, wrapper PID 1306513 — re-check with ps):
    log: annotate/out/lookback_capture.log
    leg 1: capture_activations --tag xsub_lb
           -> data/annot_captures_xsub_lb/ (81 traces, ~7-12 s/trace)
    leg 2: capture_activations --annotations data/annotations_b2_pass1.jsonl
           --corpus data/annot_b2_q8.jsonl
           --candidates data/acegen_live_b2.jsonl --tag b2_lb
           -> data/annot_captures_b2_lb/ (150 traces)
    check: tail the log; done when both legs printed "wrote N capture files"
    expected total wall ~40 min from 2026-08-27 ~08:10 UTC
gpu: GPU1 busy with the recapture; GPU0 desktop only. Kill nothing else.
```

## done_this_session (verified)

```yaml
code (all regression-checked):
  - capture_activations.py: LOOKBACK_KS=(2,4,8,16,32,64); positions
    lb_<k> at t_start-k (skipped if inside any annotated span or token
    already used; core kinds get pos_table priority); controls now 4
    per (annotation, anchor decile) — onset decile + lookback deciles —
    with "decile" and "anchor" fields; pos_table carries both; rec_keep
    = onset + anchor=="onset" controls (missing anchor = old capture =
    treated as onset, so old dirs stay compatible).
  - probe_positions.py: lb_<k> kinds accepted; lb negs are
    DECILE-MATCHED controls (core kinds still use all controls —
    regression on the old xsub dir is BYTE-IDENTICAL vs
    data/probe_results.json, verified); new printed section + JSON key
    results["lead_time"] with best-layer LOO and focal-layer LOO
    (focal: cycle L18, loop L2, muse L17).
  - NEW annotate/transfer_test.py: fit direction on substrate A
    captures, eval on substrate B (no overlap = honest); also within-
    substrate pooled and direction cosines.
  - NEW annotate/probe_recurrent.py: onset vs control in rec states,
    pooled + within-trace + LOO, per class x REC layer.
smoke:
  - one-trace lookback capture (q8 p150 s0): 28 lb positions across the
    six k values, decile-tagged controls, rec rows = onset+onset-anchor
    controls exactly. Scratch dir deleted after.

K5 transfer test (data/transfer_test.json, conf=clear, both capture dirs):
  cycle: q8->bf16 best L18 AUROC 0.995 (focal layer!); all layers 0.887+
  loop:  q8->bf16 best L2  AUROC 0.985 (focal layer!); all layers 0.876+
  muse:  NO cross reading — bf16 has <10 muse onsets (MIN_SIDE); within-q8
         0.988-0.998. K5 for muse is UNTESTED until bf16 muse material exists.
  verdict: K5 DOES NOT FIRE for cycle/loop — directions transfer across
  substrates at the focal layers. Supports "semantic, not token-distribution".

recurrent-channel probe (data/probe_recurrent.json, conf=clear):
  within-trace: median AUROC 1.000 at EVERY rec layer/class (every trace
    individually separable at onset — but within-trace fit)
  LOO (honest): rec_L2 best — cycle 0.892, loop 0.859, muse 0.710 (n=9 tr)
    rec_L9: cycle 0.830, loop 0.852, muse 0.536 (chance)
    rec_L12: cycle 0.793, loop 0.807, muse 0.593
    rec_L20: cycle 0.724, loop 0.781, muse 0.509
    rec_L8:  cycle 0.711, loop 0.709, muse 0.468
  verdict: real but WEAKER than the residual channel (0.97-0.99 LOO);
  rec_L2 strongest, confirming LAYER_HYPOTHESES' b1-era rec_2 hint.
  Residual stream stays the primary steering substrate; rec is auxiliary.
```

## next_steps (after recapture finishes)

```bash
cd /home/admin/mlfactory
P=mlfactory/experiments/ace/.venv/bin/python
LB="--cap-dirs mlfactory/experiments/ace/data/annot_captures_xsub_lb mlfactory/experiments/ace/data/annot_captures_b2_lb"

# 1. verify both legs complete (81 + 150 npz), then probe, conf=clear + all
ls mlfactory/experiments/ace/data/annot_captures_xsub_lb/*.npz | wc -l   # want 81
ls mlfactory/experiments/ace/data/annot_captures_b2_lb/*.npz | wc -l     # want 150
$P -m mlfactory.experiments.ace.annotate.probe_positions $LB --conf clear \
   --out mlfactory/experiments/ace/data/probe_results_lookback.json
$P -m mlfactory.experiments.ace.annotate.probe_positions $LB --conf all \
   --out mlfactory/experiments/ace/data/probe_results_lookback_all.json
# (also sanity: core-kind onset AUROCs in these runs should be ~equal to
#  the merged results — same traces, same kind semantics; differences =
#  extra controls/different rng draws, should be within noise)
```

Interpretation guide for the lead-time curve (results["lead_time"],
printed "lead time" section). Three questions, adjudicate each:

1. LEAD TIME — how far back does LOO AUROC stay materially above 0.5
   (say >0.8)? The largest k with strong separation is the earliest
   reliable intervention point; that is what R4 fork placement should
   nominate, not the onset itself.
2. T-1 AUDIT — if separability collapses within a few tokens (strong at
   lb_2, chance by lb_8/16), the pre_onset result was mostly
   local-context anticipation (preceding text already onset-flavored).
   If it holds at lb_32/64 where context is healthy text, divergence is
   genuinely state-driven. Write whichever verdict lands.
3. DRIFT SHAPE — gradual build (AUROC rising smoothly with shrinking k
   from 64 to 2) = accumulating trajectory in state space, strong for
   HYPOTHESIS.md's dynamics claim. Abrupt flip (flat/chance until small
   k) = discrete transition, weaker.

Watch the conf=all run too: probable-tier labels may smear the curve at
small n (muse). If muse's curve is noise, say so — do not stretch n=13.

After interpreting: lab note
`lab_notes/2026-08-27-lookback-k5-rec-results.md` covering all three
investigations; sidecars (.meta.json matching data/'s schema) for:
probe_results_lookback{,_all}.json, transfer_test.json,
probe_recurrent.json, both new capture manifests. Write-backs per the
Decisions section below. Then report to principal; R4 still HELD.

If the recapture DIED mid-way: logs tell where; both legs are resume-safe
(just rerun the same command — already-captured traces skip). Nothing
else to fix; do not change flags.
```

## Decisions

```yaml
write_back_manifest (after lead-time adjudication, with lab note as Refiner):
  - OBSERVABLES.md  # lead-time observable + K5 verdict + rec channel
  - ANNOTATION_SIDESTEP.md  # §8 operational bullets (K5 done, lead time)
  - LAYER_HYPOTHESES.md  # rec channel confirmation (rec_L2 best, weaker
                          # than residual); K5 cross-substrate row
  - STATUS.md  # R12 line: K5 verdict; open question for muse-K5 (needs
               # bf16 muse material — collection decision for principal)
still_holds:
  - R4 forks HELD — principal's call; lead-time curve now informs placement
  - directions_annot_clear_merged.npz is the fork baseline
  - muse K5 untestable without bf16 muse traces — flag, don't fabricate
principal_decisions_pending (carried over, unchanged):
  - R4 fork placement + budget
  - Vast 48783410: destroy or keep (stopped, $0.09/hr)
  - TeaLeaves: two generalizations uncommitted in /home/admin/TeaLeaves
  - mlfactory: everything since 2026-08-25 uncommitted (user decides)
```
