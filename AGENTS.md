# AGENTS.md — mlfactory

> Ambient context for any agent working anywhere in mlfactory. mlfactory is a
> bespoke framework (not in training data), so this file names what it
> provides and the non-negotiable rules — enough that you don't reinvent what
> exists or violate a rule you didn't know. **Depth lives in `docs/`**; read
> the topic file when the task names it.

## What mlfactory is

A reproducibility-first experiment factory for ML research. Every
`mlfactory run <spec>` produces a versioned run directory with a source
snapshot, hashed inputs/outputs, frozen environment, hardware provenance,
and a SQLite registry entry. Experiments live in self-contained domains
under `mlfactory/experiments/<name>/`.

## Project layout

```
mlfactory/                      # reusable harness (DO NOT modify experiment internals)
  core/                         # manifest, registry, runner, model_server, datasave, api, ...
  plugins/base.py               # StagePlugin ABC, PluginRegistry, PLUGINS
  remote/                       # ssh_runner, vast
  experiments/                  # self-contained domains (sample/, ace/, dft/, voice/, ...)
agents/skills/                  # pi agent skill docs
docs/                           # reference depth for this file (see doc index below)
runs/                           # per-run output directories (gitignored)
.mlfactory/                     # registry.db and factory state (gitignored)
models.yaml                     # local GGUF alias registry for the model server
pyproject.toml
```

## Capability index (what exists — use it, don't reinvent it)

| Need | Look up |
|---|---|
| Run a disposable llama-server, get an OpenAI client | `docs/CHEAT_SHEET.md` → `model_server` |
| Call an OpenAI-compatible endpoint with retries | `docs/CHEAT_SHEET.md` → `APIClient` |
| A/B judge two candidates | `docs/CHEAT_SHEET.md` → `Judge` |
| Save data / artifacts with lab metadata | `docs/DATASAVE.md` (`datasave` — required) |
| Save a model checkpoint | `docs/DATASAVE.md` → `save_checkpoint` |
| Find artifacts across runs | `docs/CHEAT_SHEET.md` → `Registry.find_artifacts` |
| Write a stage plugin | `docs/PLUGIN_CONTRACT.md` |
| Run / create / dry-run an experiment | `docs/RUNNING_EXPERIMENTS.md` |
| Multi-stage lineage, guard logic | `docs/RUNNING_EXPERIMENTS.md` |
| Declare a live dashboard | `docs/DASHBOARD.md` |
| Run on Vast.ai, manage secrets | `docs/REMOTE_AND_SECRETS.md`, `docs/VAST_REMOTE.md` |
| GPU memory, smoke ladder, OOM, objective safety | `docs/TRAINING_STACK.md` |
| Debugging method (any failure) | `docs/DEBUGGING_METHOD.md` |
| Lab notes + session notes protocols | `docs/NOTES.md` |

If a task needs an import path or signature, the cheat sheet's "source of
truth" column points to the module — read the module docstring; don't trust
the doc over the code.

## ⚠ Lab notes are point-in-time, append-only — NOT current truth

> **⚠ Read this before trusting any lab note. ⚠**
>
> Lab notes (`runs/<run_id>/notes.jsonl`; hand-written `lab_notes/*.md`
> inside experiments) are written **on-the-spot, in the moment** a finding
> or decision is made, and are **append-only — never rewritten**
> (`docs/NOTES.md`). A note records what was known *at the moment it was
> written*, not what is true now.
>
> **Understanding moves; notes don't.** A note's claim may have been
> sharpened, overturned, or revealed as a pooling artifact by evidence
> that landed later. A "this needs action X" decision may already be
> wrong. Treat a note as a fossil of a moment of understanding — evidence
> about *when* something was known, not a statement of *what is* known.
>
> **Before acting on a note's claim, cross-check it** against (1) the
> living status docs the note's `Decisions` section wrote back to, (2) the
> current data, and (3) the current code. If a note and the present
> disagree, **the present wins.**

## Non-negotiable rules (you violate these by default without the framework in your training data)

- **Save every datum through `datasave()`** (or `save_checkpoint()` for
  checkpoints). Never `open()`+`json.dump`/`np.save`/`to_csv`. `title` and
  `description` are required (the only provenance the code cannot derive).
  See `docs/DATASAVE.md`.
- **Run experiments through `run_from_spec()`**, never by executing a
  plugin script directly. See `docs/RUNNING_EXPERIMENTS.md`.
- **Register plugins** by importing them in `mlfactory/core/runner.py`.
  See `docs/PLUGIN_CONTRACT.md`.
- **In `finalize()`, call `finalize_artifacts(self.manifest, self.run_dir)`** —
  do not hand-roll the rglob+sha256+`FileRecord` loop.
- **Outputs go under `self.run_dir`** (artifacts in `artifacts/`, logs in
  `logs/`), never elsewhere.
- **Use `Registry(".mlfactory/registry.db")` in plugins** to match the CLI
  (the code default is `data/registry.db` — a mismatch is silent).
- **Before any Vast.ai work** (provision / configure / debug / destroy),
  read the three Vast/training docs in `docs/`: `VAST_REMOTE.md` (ops),
  `TRAINING_STACK.md` (memory, smoke ladder, OOM, objective safety), and
  `DEBUGGING_METHOD.md` (investigation discipline). DFT/Qwen3.5-specific
  setup lives in `/home/admin/facktry/BABYS_FIRST_VAST_ML_ENGINEER.md`.
  See `docs/REMOTE_AND_SECRETS.md`.
- **Hardcoding model paths**: use `models.yaml` aliases and the `model()`
  context manager, not a bare path.
- **Running outside a git repo**: `git archive` needs a committed tree —
  commit before running.
- **Set `self.manifest.summary = <dict>`** yourself when you save the
  summary via `datasave` (this is what the legacy `save_summary()` did).

## Shell discipline (every command, every agent)

Commands are executed against live state, not against what you believe
the state is. The failure mode this section exists to prevent: issuing
a command that encodes an unverified assumption (a guessed CLI syntax,
a guessed process name, a guessed flag interaction), having it fail or
half-apply, and then compounding the guess with a cleanup command that
hits the wrong target. Measured costs: a `pkill` pattern that matched
its own wrapper shell killed the fix block mid-execution and left the
system half-restarted; an unverified llama-server `--parallel`/
`--ctx-size` combination silently capped every trace at a quarter of
its intended length.

- **Verify before you speculate.** Check that a process, port, path,
  file, or CLI subcommand exists *before* embedding it in a command
  (`ps`/`ss`/`ls`/`--help`). Do not spend a mutating command to find
  out whether an assumption was right when a read-only check answers
  the same question.
- **Kill by explicit PID.** Never `pkill`/`pgrep -f` with a pattern
  broad enough to match the calling shell or an unrelated process.
  Find the PID with `ps` first; kill that PID; verify it is gone.
- **One concern per command.** Avoid long `&&`-chained blocks where a
  failure mid-chain silently skips the remaining steps. When a block
  must be long, make each step self-verifying.
- **Verify effects, not intentions.** After stop/kill/start commands,
  re-check live state (`systemctl is-active`, `ss -tlnp`, `ps`, the
  service's health endpoint) before building on the result.
- **Reproduce a service's environment from its unit, not from memory.**
  Before launching a binary that a systemd unit normally runs, read the
  unit (`systemctl cat`) for `Environment=`, `WorkingDirectory=`, and
  library paths — local builds may need an `LD_LIBRARY_PATH` that the
  unit sets and the bare shell does not.
- **Long-running remote commands detach cleanly.** `nohup ... </dev/null
  > log 2>&1 &`, or an SSH session waits on the child and local
  timeouts fire on jobs that are actually fine.

## Code and prompt hygiene

- **No "harmless" dead code.** Unused imports, unreachable branches,
  commented-out blocks, compat shims that no longer guard anything —
  delete them when you see them. "It's harmless" trades a moment of
  cleanup for every future reader re-deriving whether the code is
  live; if you know it should go, it goes.
- **No stale meta-text in anything sent to a model.** Prompts are
  artifacts with one reader that has no memory: changelogs,
  rationales, and "what changed since last time" blocks stay out of
  prompt files; they live in the log next to the prompt, not inside
  it.

## Model downloads and transfers

- **Use `hf_xet` for all Hugging Face transfers.** `pip install
  hf_xet`, then `hf download <repo>`. Xet's parallel chunked fetch
  measured ~450 MB/s on a rented box where single-stream curl from the
  same HF CDN measured 15–22 MB/s (~30× on the same link, same file).
  curl from HF is the fallback, not the default.
- **Prefer MTP-enabled model variants.** When downloading a model,
  choose the MTP-enabled variant unless something specifically requires
  otherwise (e.g. the non-MTP build is itself the experimental
  variable): `*-MTP-GGUF` repos (unsloth), HF repos shipping `mtp.*`
  weights. MTP self-speculation is lossless — draft verification
  preserves the target token distribution — and the throughput gain is
  large (measured 152–161 tok/s for Q8_0+MTP vs 37 tok/s HF bf16 on the
  same 3090; ~125 tok/s for BF16-GGUF+MTP on a Blackwell RTX PRO 6000).
  No quality downside; the constraint is serving-stack support
  (llama.cpp `--spec-type draft-mtp` — HF transformers ignores
  Qwen3.5's `mtp.*` weights as of 5.14.1, so the speedup currently
  rides with llama.cpp).

## Subagents (delegation discipline)

- **Spawn subagents on `qwen/qwen3.7-plus`** (the Agent tool's `model`
  parameter) unless a task specifically needs another model.
- **Parallelize and background them.** Independent tasks go in one
  message as several Agent calls, each `run_in_background: true`;
  collect the results after. Don't serialize what can run at once.
- **Nearly all work stays in the main thread.** Subagents are for work
  whose *content* would pollute the main context long-term without
  belonging to the main task: reviewing full 20k-token traces,
  auditing large batches, reading zipbomb-sized files, side-analyses.
  The subagent absorbs the bulk; the main thread receives a terse
  verdict.
- **Ask for compact reports.** Give the subagent an exact output
  format (one line per item: verdict + short reason) so the result
  lands small.

## Start here (reference implementation)

`mlfactory/experiments/sample/` is a full-featured 4-stage pipeline
(transform → classify → train → eval) demonstrating every pattern: plugin
lifecycle, MetricsLogger, datasave, `finalize_artifacts`, model server,
APIClient, `inference_env`, multi-stage lineage, guard logic. Read its
README and its four plugins before writing a new experiment.

## Provider preferences

- **Lunaroute:** always prefer the `-ballast` model variant when one is
  available (e.g. `glm-5.2-vision-ballast`). Fall back to the plain model
  only if no ballast variant exists. On first use in a session, query
  `GET /v1/models` to get the list of currently active model names — the
  available set can change between sessions.
- **Lunaroute billing:** not pay-per-token. Long generations, thinking
  runs that hit the token wall, and outright duds cost nothing beyond
  wallclock time — don't engineer around token spend, and don't treat
  `finish_reason=length` as a budget incident. Retry once; keep
  whatever lands.
- **Prompting GLM (via Lunaroute):** GLM is a big thinker. Leave it
  room (large `max_tokens`; thinking tokens count toward it) and leave
  `temperature` at the provider default — the models are tuned for a
  specific temperature target, so override only with a reason. Lessons
  from hillclimbing an annotation prompt on `glm-5.2-vision`:
  - Clear direct instructions beat clever ones; keep secondary
    information out of instruction lines.
  - **Rewrite a rule line instead of appending edge-case clauses.**
    GLM shares the ACE meander problem: accumulated edge-case
    handling feeds wandering thinking traces and can blow the
    thinking budget entirely.
  - Never put meta-commentary in a prompt ("what changed since last
    time") — the model has no memory of last time; it only sees the
    current forward pass. Version provenance lives in the log beside
    the prompt file, never inside it.
  - For judgment tasks prefer recall over false restraint: extra
    low-confidence output is droppable later; output withheld by an
    over-strict "only flag if certain" rule is lost forever.

## CLI

```bash
mlfactory run <spec.yaml>                       # run an experiment
mlfactory init <spec.yaml>                     # create run dir + manifest, no execution
mlfactory ls [--stage ...] [--status ...]      # list runs
mlfactory show <run_id>                        # show a run + its notes
mlfactory note <run_id> <text>                 # append a lab note
mlfactory notes <run_id> | --grep <term>        # read / search notes
mlfactory dashboard [--watch-run <id>] [--config <path>]
mlfactory registry merge <remote.db>          # merge a remote registry
mlfactory remote run <spec>                    # run on a provisioned Vast instance
mlfactory secrets set|get|list|delete ...
```

Install: `python3.14 -m pip install -e ".[remote]" --break-system-packages`
Tests: `python3.14 -m pytest tests/`

## Common mistakes (quick list; see docs/ for why)

- Running experiment scripts directly — go through `run_from_spec()`.
- Writing outputs outside `self.run_dir`.
- Hand-writing data files instead of `datasave()`.
- Saving data without `title`/`description`.
- Hand-rolling the `finalize()` loop — use `finalize_artifacts`.
- Not registering the plugin in `runner.py`.
- Hardcoding model paths — use `models.yaml` + `model()`.
- Forgetting `self.manifest.write()` in `finalize()` — `finalize_artifacts` persists it.
- Running outside a git repo.
- Using `Registry()` default path in plugins — use `.mlfactory/registry.db`.

## Experiment-specific context

Some experiments carry their own `AGENTS.md` with the layer on top of this
one (e.g. `mlfactory/experiments/ace/AGENTS.md`). When working in an
experiment directory, read it — it carries the experiment's binding
context and patterns, and defers to this file for framework concerns.
