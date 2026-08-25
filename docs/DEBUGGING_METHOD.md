# Debugging method: investigating failures this runbook doesn't cover

> Update when: the investigation discipline changes. This is the genuinely
> universal content — it applies beyond ML training to deployment,
> inference, networking, data pipelines, and evaluation. The technical
> commands change; the discipline does not. Failure-specific decision trees
> (OOM symptoms, objective collapse) live in `TRAINING_STACK.md`;
> provisioning/ops in `VAST_REMOTE.md`.

## First decide whether to stop the run

Stop immediately when continuing can: corrupt or overwrite the only
checkpoint; spend substantial credits without producing useful evidence;
train under a mathematically invalid objective; produce escalating
NaN/Inf, entropy, KL, or memory; damage data or delete remote artifacts;
leak a credential. Keep the process running briefly only when its live
state is necessary for inspection and the cost/risk is low — taking
`nvidia-smi`, process, stack, and log snapshots before stopping can be
valuable.

Prefer reversible actions: stop a Supervisor process rather than killing
the whole instance; rename an output directory rather than deleting it;
copy logs/checkpoints home before editing in place; create a new launch
script rather than overwriting the known-good one. Ask the user before
irreversible or materially expensive actions (destroying an instance,
changing the target architecture, abandoning a major experiment premise,
launching a large scale-up). Safe diagnostics and small reversible smoke
tests can usually be performed autonomously.

## Write the problem in one falsifiable sentence

Bad: `Training is broken.`
Useful: `On 2x <GPU>, <model> reaches <stage>, reserves <X>/<Y> GB, then
emits no step log for <N> minutes while both GPUs remain at 0% and the
process uses <Z>% CPU.`

Include: exact expected behavior; exact observed behavior; the first stage
where they differ; time scale; hardware/software versions; whether the issue
reproduces. A precise symptom prevents unrelated fixes from becoming
attractive.

## Separate observation, inference, and decision

```
Observation: PID <n> uses <X> GB on GPU 0 and <Y> GB on GPU 1.
Inference:  an unrelated inference server is consuming training headroom.
Decision:   stop Supervisor service <name> and rerun the identical smoke test.
```
Do not write an inference as though it were observed fact. 0% GPU
utilization is an observation; "Triton is deadlocked" is a hypothesis until
tested.

## Locate the failing layer

Classify the problem before changing anything:

1. **Instance** — hardware, disk, network, credits, persistence
2. **Service** — hidden processes, Supervisor, ports, stale workers
3. **Environment** — Python, torch, CUDA, driver, compiled packages
4. **Data** — malformed examples, token lengths, corruption, leakage
5. **Model-load** — authentication, quantization, device mapping
6. **Forward/generation** — tokenizer, masks, cache, attention backend
7. **Backward/optimizer** — gradients, activation memory, extension correctness
8. **Objective** — reward signs, scales, KL, critic, clipping
9. **Evaluation** — slicing, references, sample size, judge bias
10. **Artifact** — checkpoint save/load, sync, checksum, resume semantics

Test the boundary immediately before and after the suspected layer. If
imports and CUDA tensors work but model load fails, do not debug PPO. If
forward works but exact backward fails, do not change the dataset first.

## Preserve a diagnostic snapshot

Before restarting or editing, capture:
```bash
date
supervisorctl status 2>/dev/null || true
nvidia-smi
ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd --sort=-%cpu | head -50
free -h
df -h /workspace
tail -200 /tmp/the_run.log /tmp/the_run.err
```
Also save: parsed run configuration; last healthy and first unhealthy
metric rows; package versions; source file checksum; checkpoint listing
and sizes; exact exception text including the first traceback, not merely
the final wrapper error. Do not rerun until the evidence needed to compare
old and new behavior is preserved.

## Reduce to the smallest faithful reproducer

"Small" is not enough; it must still execute the failing path. An import
test diagnoses binary compatibility but not backward kernels. A forward
pass does not test backward. Batch 1 may not exercise a population
objective. A toy model does not reproduce the target architecture's
failure modes. A different model family may avoid the bug rather than
explain it.

Reduce one dimension at a time: same target model and backend → same
failing operation → fewer examples → shorter sequence (if sequence length
is not the suspected cause) → one optimizer step → minimal
evaluation/checkpointing. A faithful three-step smoke is often more
informative than hours of full training.

## Rank hypotheses before testing them

| Rank | Hypothesis | Evidence for | Evidence against | Cheapest falsifying test |
|---:|---|---|---|---|
| 1 | Hidden GPU process | Baseline VRAM already occupied | None yet | `nvidia-smi` process list |
| 2 | Full-logit memory | Peak occurs at LM head | Model is only 4B | Log tensor shapes / use response-only logits |
| 3 | Allocator fragmentation | Reserved far above allocated | Fresh process also fails | Fresh tiny run with allocator config |

Rank by: fit to the exact symptom; prior probability; cost of testing; risk
of the test; ability to distinguish competing explanations. Test the
cheapest high-information hypothesis first. Do not start by reinstalling
the entire machine.

## Change one meaningful variable at a time

If you simultaneously change GPU architecture, upgrade torch, lower batch
size, enable checkpointing, and switch model family, a successful run
teaches almost nothing about the cause. Keep a control configuration and
record each delta. When several changes are inseparable, say so explicitly
and plan follow-up ablations. Use new output directories for every
diagnostic run — never let two tests write the same metrics or checkpoint
path.

## Instrument the boundary, not everything indiscriminately

Add measurements where information crosses the suspected boundary:
before/after each model load (GPU memory + device map); before/after
generation (input width, output width, EOS, cache); before/after forward
(logits shape/dtype); before/after backward (allocated/reserved peaks,
gradient norm); before/after optimizer step (old/new KL, parameter delta);
before/after evaluation decoding (padded width, response-only slice).
Excessive logging can slow or perturb training — log scalars, shapes,
dtypes, and selected samples rather than full tensors.

## Read source and primary evidence

When an error is unfamiliar: (1) search the exact quoted error; (2) add
model name, GPU architecture, torch, Triton, or extension name; (3) prefer
official documentation, source code, release notes, and maintainer issue
threads; (4) check issue dates and package versions; (5) read the code
that raises a correctness guard before bypassing it; (6) distinguish exact
matches from merely similar symptoms. An analogous report on a different
stack may reveal a relevant pattern, but it is not proof of the same root
cause — use analogous reports to generate hypotheses, then verify on the
actual stack. **Never disable a warning or assertion solely because a
search result says it is safe.** Determine what invariant the check
protects.

## Use decision thresholds and timeboxes

Define in advance what counts as success or pivot — concrete thresholds
that trigger a stop or a stack change, not a vague "see if it works":
```
If <the exact failing operation> does not finish within <N> minutes on <target GPU>, stop.
If <reference KL> exceeds <threshold>/token, save a guard report and stop.
If <output length> remains below <X>% of baseline for <N> rollouts, stop.
If the same <symptom> survives <one allocator test> and <one backend test>, pivot stacks.
```
Timeboxes prevent random-walk debugging. Reasonable sequence: minutes —
hidden processes, disk, auth, obvious config; 15–30 min — minimal
reproducer and instrumentation; 30–90 min — package/backend compatibility
research; longer — only with a strong hypothesis and evidence that the
answer matters. A pivot is rational when it preserves the scientific
target while replacing an unreliable implementation layer (e.g. moving to
a GPU architecture whose kernels are mature for the target model), not
when it changes the target itself (e.g. switching model families to dodge
a bug).

## Distinguish workaround from root cause

- Lowering batch size may avoid an OOM without explaining it.
- Restarting may clear fragmentation without identifying the retained tensor.
- Disabling a critic may fix memory and objective stability, but the root
  causes are its graph path, initialization, and loss scale.
- Moving to a more mature GPU architecture works around kernel immaturity;
  it does not prove the original hardware is defective.

Document both:
```
Root cause: <the actual mechanism producing the failure>.
Immediate workaround: <what you did to stop the bleeding>.
Correct fix: <the change that removes the mechanism>.
```
This prevents future agents from cargo-culting a workaround into unrelated
experiments.

## Turn every solved failure into infrastructure

After identifying a cause: (1) add a focused regression test; (2) add a
startup assertion if the invalid state can be detected cheaply; (3) add a
runtime guard if it can emerge later; (4) correct the bootstrap/launch
script; (5) update the relevant doc (`TRAINING_STACK.md`, `VAST_REMOTE.md`,
or the experiment-specific setup); (6) preserve a failed-run manifest
showing the signature. A debugging session is incomplete if the next fresh
instance can repeat the same failure silently.

## Report a situation clearly

Five parts: **State** (running/stopped/completed/guarded/failed); **Evidence** (step, metrics, memory, exact error); **Interpretation** (what the evidence supports and what remains uncertain); **Action taken** (anything stopped, archived, or launched); **Next decision point** (acceptance threshold and expected time). Template:
```
State: <running/stopped/completed/guarded/failed> at <step>.
Evidence: <the observations that characterize the state>.
Interpretation: <what the evidence supports; what remains uncertain>.
Action: <what you stopped, archived, or launched>.
Next: <the acceptance threshold and the experiment that will test it>.
```
Do not call noisy early metrics a win. Separate "stable enough to continue"
from "scientifically proven improvement."

## Investigation worksheet

Copy into experiment notes:
```
Goal:
Expected behavior:
Observed behavior:
First failing stage:
Last known-good configuration:
Hardware/image:
Python/torch/CUDA/backend versions:
Artifacts at risk:
Immediate stop required? Why?:

Observations:
1.
2.
3.

Ranked hypotheses:
1. Hypothesis — falsifying test — result
2. Hypothesis — falsifying test — result
3. Hypothesis — falsifying test — result

Single variable changed:
Control result:
Treatment result:
Conclusion confidence:
Workaround:
Root-cause fix:
Regression test/guard added:
Artifacts synced and checksummed:
Next decision threshold:
```

## Final principles

1. Architecture compatibility beats advertised VRAM.
2. Clear unrelated GPU processes before reducing the experiment.
3. Install torch first, then compile extensions against that torch.
4. Prove the exact backward path, not merely model loading.
5. Separate rollout population, backward mini-batch, and reference batch deliberately.
6. Log tensor/memory/objective diagnostics, not just loss.
7. A live process can still be a failed experiment.
8. Archive failed runs — they are evidence.
9. Supervisor protects against SSH loss; guards protect against silent collapse.
10. State observations separately from inferences and decisions.
11. Test ranked hypotheses with the smallest faithful reproducer.
12. Change one meaningful variable at a time.
13. Turn every root cause into a test, guard, script fix, or doc update.
14. Nothing on rented hardware is real until a verified copy exists at home.
