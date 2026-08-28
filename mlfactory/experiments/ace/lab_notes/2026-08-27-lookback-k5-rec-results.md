# LAB NOTE — 2026-08-27 — lookback curves, K5 transfer, recurrent channel

> Session: 2026-08-27 (morning, unsupervised stretch). Three "free
> detection" investigations on the merged corpus, all scenario-A cost.
> Predecessor handoff: `2026-08-27-handoff-lookback-k5-rec.md`.
> Verdicts below; fork spend (R4) still held.

## 1. Lookback — how far back is divergence readable?

Built: `capture_activations.py` gained `LOOKBACK_KS=(2,4,8,16,32,64)` —
positions lb_<k> at k tokens before each onset (skipped if inside any
annotated span), with controls now sampled 4 per (annotation, anchor
decile) so each lookback depth is compared against same-depth healthy
states. Probe scores lb kinds against decile-matched controls only
(core kinds unchanged — regression on old captures byte-identical).
Recaptured the full merged corpus with the expanded position set into
NEW dirs (old captures untouched): `data/annot_captures_xsub_lb/` (81),
`data/annot_captures_b2_lb/` (150), zero failures. Sanity: core-kind
AUROCs in the new captures match the prior merged numbers within noise
(cycle onset L18 0.992→0.993, loop L2 0.978→0.981, muse L17 0.952→0.949).

Lead-time curves (LOO AUROC, conf=clear; n≈200 per point for
cycle/loop, n=9 for muse; conf=all in parentheses-shape agreement):

| k before onset | cycle | loop | muse |
|---|---|---|---|
| 2  | 0.97 (focal L18 0.971) | 0.98 (best L30; focal L2 0.858) | 0.99/0.97 |
| 4  | 0.87 | 0.86 | 0.86 |
| 8  | 0.75 | 0.73 | 0.76 |
| 16 | 0.66 | 0.65 | 0.73 |
| 32 | 0.55–0.56 | 0.61 | 0.58 |
| 64 | 0.54–0.56 | 0.57 | 0.58 (conf=all dips to 0.44–0.53) |

Adjudication of the three pre-stated questions:

1. **Lead time (intervention point).** Strong detection ≥0.97 sits at
   2 tokens pre-onset; ~0.86 at 4; ~0.73–0.75 at 8; weak at 16; chance
   from 32 onward. **Earliest reliable intervention point is ~4–8
   tokens upstream of onset.** Forks should nominate states in that
   window — not the onset itself (detectable but already committed to
   writing), and not 32+ upstream (indistinguishable from healthy).
2. **t−1 audit (anticipation confound).** PARTIAL PASS. The signal is
   not single-token anticipation: lb_4 (0.86) and lb_8 (0.73–0.76) at
   n≈200 are far above chance, so the state is reorganizing before the
   span's tokens exist. BUT the decay is steep — by lb_32, where the
   preceding text is unambiguously healthy, separability is at chance.
   Divergence is encoded in the state before the episode's tokens, but
   only shortly before (~8–16 token horizon).
3. **Drift shape.** Sharp transition with a short precursor, NOT a long
   gradual slope from 64 out. The state flips over the last ~8–16
   tokens before the episode starts being written. Refines (does not
   kill) the HYPOTHESIS.md dynamics claim: divergence is a state
   property — confirmed, readable pre-token — but its run-up is abrupt
   and local, not a tens-of-tokens accumulation.

Side observation: lookback signal rides mid-to-high layers (best layers
L17–L30 at lb_2/4 for all classes); loop's own focal L2 carries lb_2
only at 0.858 and decays faster than its best layer L30. The layer that
carries "about to diverge" is not necessarily the layer that carries
"diverging."

## 2. K5 transfer test — kill condition 5 does not fire

`annotate/transfer_test.py`: directions fit on one substrate's captures,
evaluated on the other's (disjoint fit/eval = honest). Results
(`data/transfer_test.json`): cycle q8→bf16 best L18 AUROC **0.995**
(the focal layer), all layers ≥0.887; loop q8→bf16 best L2 **0.985**,
all layers ≥0.876. The detection pattern is semantic, not
token-distribution-specific — it crosses the q8/bf16 substrate boundary
at exactly the layers the probes nominated. **K5 does not fire.**

Caveat: muse has NO cross reading — bf16 captures carry <10 muse onsets
(muse material is concentrated in q8/b2 traces). K5 for muse is
UNTESTED; testing it would need bf16 collection on muse-rich prompts
(a principal decision, not spent).

## 3. Recurrent channel — real but auxiliary

`annotate/probe_recurrent.py`: onset vs same-trace controls in DeltaNet
recurrent states, REC_LAYERS [2,8,9,12,20], 231 captures
(`data/probe_recurrent.json`). Within-trace: median AUROC 1.000
everywhere (every trace individually separable at onset). LOO (honest):
**rec_L2 best — cycle 0.892, loop 0.859, muse 0.710 (n=9 traces)**;
other layers 0.47–0.85. Real cross-trace signal, clearly weaker than
the residual channel (0.97–0.99 LOO), best layer rec_L2 confirming the
b1-era rec_2 hint in LAYER_HYPOTHESES. **Residual stream stays the
primary steering substrate; the rec channel is auxiliary** (and the
cheaper one to steer through, if forks ratify either).

## Decisions

```yaml
docs_this_note_changes:
  - OBSERVABLES.md           # lead-time observable row; K5; rec channel
  - ANNOTATION_SIDESTEP.md   # §8 bullets: K5 done, lead time done
  - LAYER_HYPOTHESES.md      # rec-channel confirmation; lookback layers
  - STATUS.md                # R12 clause (K5 + lead time); open row for muse-K5
for_principal:
  - R4 fork placement should use the 2–8 token pre-onset window, not
    the onset token (lead-time curve above)
  - muse-K5 untested: needs bf16 muse material — collection decision
still_holds:
  - R4 forks HELD; directions_annot_clear_merged.npz is the baseline
  - nothing rented, nothing committed, no service touched
```
