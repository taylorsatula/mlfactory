# LAB NOTE — 2026-08-27 — b2 corpus merged into capture/probes (R12 reconfirmed at 3× n)

> Session: 2026-08-27. The overnight batch
> (`2026-08-27-overnight-b2-collect-annotate.md`) landed 184 new q8
> rollouts over all six b2 families + 879 annotation flags. This
> session merged that corpus into the R1–R3 pipeline and re-scored
> detection on 231 captures. Result: everything xsub found holds; muse
> is no longer underpowered.

## What ran

- Pipeline generalized for multi-corpus use (principal sanctioned —
  "we'll be using this script more"): `capture_activations
  --corpus/--candidates/--tag`, `probe_positions --cap-dirs`,
  `compute_directions --cap-dirs/--tag`. Per-tag capture dirs because
  xsub is a prompt subset of b2 — all six xsub pids recur in b2 with
  new seeds, so (pid, sample_i) keys collide; npz filenames would
  silently mix traces under one dir. Regression-checked first: xsub
  defaults reproduce the saved probe_results.json exactly and the R3
  direction arrays bit-identically.
- R1: 150/150 b2 traces captured (~7s each, zero FAIL/SKIP) →
  `data/annot_captures_b2/` (81 xsub captures untouched in
  `data/annot_captures/`).
- R2 merged (231 captures): conf=clear →
  `data/probe_results_merged.json`; conf=all →
  `data/probe_results_merged_all.json`.
- R3 merged: 96 unit directions →
  `data/steering_directions/directions_annot_clear_merged.npz`.

## Results (merged corpus, LOO AUROC unless noted)

| class | onset (clear) | onset (all) | pre_onset | mid | xsub-only was |
|---|---|---|---|---|---|
| cycle | L18 0.992, L16–19 0.989 (n=291) | 0.990 | L27–30 0.985–0.986 | ~0.72 | 0.988–0.991 (n=128) |
| loop  | L2 0.978, L2–6 0.975+ (n=285) | 0.979 | L26–29 0.977–0.980 | ~0.66 | 0.983–0.988 (n=78) |
| muse  | L16–19 0.947–0.952 (n=13) | 0.967–0.968 (n=29) | L22–29 0.95–0.96 | ~0.83 | 0.979–0.982 (n=5) |

Escape vs reheat at onset (succ vs fail onsets, same class):
- loop L11/12/16: 0.861–0.865 (174v92... n_succ=193 v n_fail=92 at
  conf=all) — stable vs xsub 0.856
- cycle L28–30: 0.721–0.723 (174v117) — DEFLATED from xsub 0.781
- muse conf=clear L4–6: 1.000 (8v5 — fragile, small n); conf=all
  L17: 0.957 (23v6)

## Verdicts

- **R12 reconfirmed at ~3× n.** cycle/loop onset separability held
  essentially where xsub put it (L18, L2). K1/K2/K3 remain not-fired.
- **muse upgraded underpowered → supported.** 13 clear / 29 all
  onset positions over 19 traces, six families. Best layers shifted
  (xsub L23–29 → merged L16–19); both readings n≤29, so muse layer
  preference is still not a settled question — but onset separability
  itself is.
- **The larger corpus deflated one number**: cycle escape-vs-reheat
  0.781 → 0.72. The xsub reading was mildly inflated; loop's did not
  move. Anti-Goodhart check passed — scaling up did not manufacture
  signal, it mostly confirmed it.
- Onset AUROCs remain broadly high across layers; the standing caveat
  (distributed signal vs residual position confound) is unchanged.
  Forks (R4) remain the test that distinguishes them.

## Decisions

```yaml
docs_this_note_changes:
  - OBSERVABLES.md  # muse underpowered -> supported; merged numbers; deflation noted
  - STATUS.md       # R12 reconfirmed line
  - ANNOTATION_SIDESTEP.md  # b2-corpus bullet extended with probe verdict
still_holds:
  - HOLD before R4 forks — principal's call, now made on 231-capture
    evidence instead of 81
  - steering directions for any fork: use directions_annot_clear_merged.npz
    (supersedes directions_annot_clear.npz as the constant-lambda baseline)
```
