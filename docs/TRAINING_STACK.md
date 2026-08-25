# Training stack: GPU memory, smoke-test ladder, OOM, and objective safety

> Update when: the training-stack understanding shifts — a new memory gotcha,
> a new OOM symptom, a revised smoke procedure, a new objective-safety check.
> One file because these update together: a change in how training works
> reframes memory, the smoke ladder, and failure modes at once. Section
> headers carry the read-finding. Operational Vast provisioning lives in
> `VAST_REMOTE.md`; the general debugging method lives in `DEBUGGING_METHOD.md`.

## Where GPU memory goes

Model weights are only one component. Peak VRAM can include: quantized
base weights, LoRA params, gradients, Adam optimizer moments,
activations saved for backward, full-vocabulary logits, KV cache during
generation, frozen reference/reward models, CUDA kernels/workspaces, PyTorch
allocator reservations, unrelated processes.

### Full-vocabulary logits are enormous

For a vocabulary near 248,000, batch 20, 257 response predictor positions:
`20 × 257 × 248,000 × 2 bytes ≈ 2.55 GB` — one BF16 logits tensor. Float32
doubles it; log-softmax may allocate another; projecting for full prompt +
response doubles it again. Mitigations:
- Use `logits_to_keep` to project only response positions.
- Gather selected token log-probs before moving between GPUs.
- Do not transfer full-vocabulary logits across devices.
- Keep full-vocabulary tensors in BF16/FP16; cast selected scalar/token
  statistics to FP32.
- Disable `use_cache` in training forwards.

### A critic can secretly dominate memory

A value head seems tiny, but backpropagating its loss through all hidden
states can retain a large graph and update the shared LoRA backbone. **Do
not add a critic reflexively.** Verify its initialization, reward scale,
gradient path, and memory impact before committing.

### Allocated versus reserved

PyTorch's allocator reserves blocks for reuse. **Allocated**: memory in
tensors. **Reserved**: memory owned by PyTorch, including unused cached
blocks. **`nvidia-smi` used**: process/driver view, often close to reserved
plus overhead. Log both:
```python
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.max_memory_allocated(i)/2**30, torch.cuda.max_memory_reserved(i)/2**30)
```
If reserved is much larger than allocated and allocations fail despite
apparent free memory, fragmentation may be involved.

### Allocator configuration

Set this **before Python imports torch**:
```bash
export PYTORCH_ALLOC_CONF='expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]'
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
```
The first variable is used by newer torch; the second supports older naming.
Native Linux is the validated environment; `expandable_segments` may not
work under some WSL arrangements.

### Triton autotuning

For some setups (e.g. hybrid linear-attention on Hopper),
`export TRITON_DISABLE_AUTOTUNING=1` avoids costly or hanging autotuning
paths. It is **not** a universal performance setting; reassess for a
different model/backend after correctness is established.

## Smoke-test ladder

Never jump directly from package installation to an overnight run.

| Level | What it proves | Notes |
|---|---|---|
| 1: imports + CUDA tensor | correct interpreter; packages import; PyTorch sees the GPU; basic kernel works | |
| 2: model loading | auth; checkpoint compatibility; quantization path; device mapping; baseline weight memory | load each model onto its intended GPU **sequentially**; `nvidia-smi` after each |
| 3: generation only | chat template/tokenizer; attention backend; EOS/padding; KV cache/generation path | generate two responses from the exact target architecture |
| 4: one real backward step | backward backend correctness; CUDA extension compatibility; LoRA gradients; optimizer step; peak activation memory | use the real loss and trainable modules, not a toy linear layer; small batch but intended prompt/response lengths |
| 5: target-batch three-step smoke | optimizer-state allocation on first update; allocator growth; graph leaks; policy drift; compile warmup | intended batch size; include checkpointing and evaluation; three steps reveal whether memory and time settle |
| 6: guarded short validation | KL/entropy/output length/EOS/truncation/reward variance/loss scale/NaN/Inf/task metrics/GPU peak | 10–20 steps with frequent evaluation and automatic stop conditions |
| 7: full run | — | only after every earlier level passes |

Do not resume a checkpoint created under a mathematically incorrect
objective merely to save setup time. Archive it, fix the objective, and
restart from the pristine base.

## Benchmark batch size scientifically

Do not select batch size from intuition alone. For each candidate: start
from the same base/model state; run at least 2–3 steady steps; record
examples/second, step time, peak allocated/reserved VRAM; leave a safety
margin for evaluation, checkpoints, and allocator variability.

**Code-path changes can matter more than reducing batch size** — disabling
an unnecessary critic or correcting an objective can halve step time and cut
reserved memory by a large fraction without touching batch size. Do not
automatically raise batch to consume all newly free VRAM — distributional
objectives may have statistical reasons for a larger rollout population, but
generation, backward mini-batch, and MMD population can be decoupled
deliberately.

## OOM and hang decision tree

### Symptom A: OOM during model loading

Check in order: (1) `nvidia-smi` for unrelated processes; (2) whether 4-bit/
8-bit quantization was actually enabled; (3) whether passing `torch_dtype`
defeated or altered the intended quantized load; (4) device map — did every
model land on GPU 0?; (5) host RAM — did CPU staging die before GPU
placement?; (6) cache/download corruption. When a `BitsAndBytesConfig`
controls 4-bit loading, do not force `torch_dtype` in `from_pretrained` —
let the quantizer set the compute dtype.

### Symptom B: load succeeds, first training forward OOMs

Likely causes: batch too large; prompt+response length larger than
expected; full-vocabulary logits for every prompt token; training KV cache
accidentally enabled; slow fallback attention implementation; padding to a
pathological maximum. Actions: log actual tensor shapes; `use_cache=False`
in loss/PPO forwards; response-only logits projection; length
bucketing/dynamic batching; verify the fast attention backend is active;
reduce batch only after inspecting the above.

### Symptom C: forward succeeds, backward OOMs

Likely causes: saved activations; critic/value loss retaining the backbone
graph; full-precision logits/log-softmax; too many PPO epochs or retained
graphs; optimizer state allocated on first step; accidental gradients
through frozen models. Actions: confirm reference/reward models have
`requires_grad=False` and are under `no_grad`; inspect which losses
backpropagate through the policy backbone; cast only selected-token
statistics to FP32; zero or remove unnecessary critic paths; use gradient
checkpointing only after verifying backend compatibility and performance;
reduce backward mini-batch separately from rollout batch.

### Symptom D: huge VRAM, 0% GPU, high CPU, no traceback

Possible causes: Triton compilation/autotuning; a broken kernel compile;
allocator fragmentation/retries; CPU preprocessing/tokenization; network or
filesystem waiting; deadlock. Actions:
```bash
nvidia-smi
ps -eo pid,stat,etime,%cpu,%mem,cmd --sort=-%cpu | head
ls -lt ~/.triton 2>/dev/null | head
```
Run a tiny exact-architecture backward smoke. A one-time compile can be
acceptable; an unbounded silent stall is not. If the symptom is
architecture-kernel immaturity (huge reservation, 0% GPU, high CPU, no
traceback) and it survives an allocator test and a backend test, pivot to a
GPU architecture whose kernels are mature for the target model rather than
spending hours on random batch reductions.

### Symptom E: reserved memory grows each step

Possible causes: graph/tensor references retained in metric dictionaries or
lists; `retain_graph=True`; outputs stored on GPU; variable shapes causing
allocator fragmentation; evaluation mixed into training without cleanup.
Actions: store scalars with `.item()` and CPU arrays, not live GPU tensors;
delete full logits/outputs as soon as selected statistics are extracted; use
stable length buckets; log allocated and reserved peaks every step;
reproduce in a 10-step memory-only smoke.

### Symptom F: process alive but outputs collapse

Not an OOM. Check: entropy explosion or collapse; response token length;
EOS rate; non-English leakage; KL sign and estimator definition; reward
scale versus value/entropy losses; clipping fraction; NaN/Inf. A run can
remain technically healthy (process alive, no OOM, no NaN) while outputs
collapse — output length cratering and entropy running away. **Supervisor
cannot detect this.** Metric guards and human interpretation must.

## Training-objective safety checks

Infrastructure success is not experiment success. Every RL/on-policy run
should log and guard at least: reward mean/std, policy loss, value loss (if
any), entropy, reference KL or another trust-region metric, PPO old/new KL,
clip fraction, average/median response tokens, EOS rate, truncation rate,
domain correctness/quality metric, GPU peak allocated/reserved.

### Beware fake or misused KL terms

A signed sampled log-ratio is not automatically a nonnegative
differentiable KL loss. Directly minimizing `new_log_prob -
reference_log_prob` on fixed sampled tokens lowers the probability of those
tokens and can cause entropy explosion. Standard PPO-style reference
regularization commonly applies a detached rollout reward:
`-beta * (old_policy_logprob - reference_logprob)`, then PPO optimizes the
resulting reward. Use a separately defined nonnegative estimator for
diagnostics.

### Reward-scale check

Print the relative sizes of every loss component. The intended reward may
not be the dominant learning signal — an entropy coefficient or an
unnecessary value loss can outweigh it. If the reward's standard deviation
is small relative to other loss terms, the model is not learning the task.

### Automatic guards

For a new run, add stop conditions for: any NaN/Inf; response length below
a fraction of the initial baseline for multiple steps; entropy moving far
beyond baseline; excessive reference KL; repeated domain-metric
deterioration. A guard should save a report and stop cleanly without
labeling the adapter `final`.
