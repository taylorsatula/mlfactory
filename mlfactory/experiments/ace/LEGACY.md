# Legacy: inherited conventions and stale references

> Update when: a stale reference is fixed, or a convention is adopted into
  the current experiment. The archived experiment is read-only — never
  write into it.

## The archived experiment

The original ACE experiment (post-hoc trace rewriting: collect → classify
→ stratify → LLM-editor rewrite) is archived in
`mlfactory/experiments/ace-legacyapproach/`.

- Its plugin stages are no longer registered in
  `mlfactory/core/runner.py`.
- The research goal is unchanged but the approach moved from editing
  traces after the fact to steering generation causally
  (`APPROACH_HISTORY.md`).
- **Legacy data is read-only.** The inherited recon probes
  (`scratch/trace_replay.py`, `scratch/waypoint_alignment.py`) read
  immutable source data from `ace-legacyapproach/data/` (see path
  constants at the top of each file). Never write into the archive.

## Inherited record conventions

- **Trace records:** frozen-envelope fields (`envelope_hash`,
  `surface_hash`, `seed`, `domain`, `prose`, `surface_question`) + `trace`
  + `provenance`.
- **Provenance on every produced record:** model id/path, sampling
  params, seed, corpus name, source pointers.
- **Immutable inputs; append-safe JSONL artifacts; sidecar files for
  bulky per-token data.**
- New artifacts for this experiment live under
  `mlfactory/experiments/ace/data/` (gitignored) and must not be written
  into the legacy archive.

## Known stale references (legacy tooling, not yet updated)

| Reference | Status | Fix |
|---|---|---|
| `/home/admin/mlfactory/run_ace_rewrite_lunaroute.py` | untracked legacy script; default trace path still points at `mlfactory/experiments/ace/data/` | not this experiment's file to fix |
| `agents/skills/*/SKILL.md` and repo `README.md` examples | reference legacy spec paths (e.g. `mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml`), which now live under `ace-legacyapproach/specs/` | update if those skills are revived |
| `mlfactory/core/prompts.py` docstring example | references a legacy prompt path | update opportunistically |

## The curated extraction

The legacy prompts and after-action report contained reusable vocabulary
independent of the retrospective-editing use case: a defined vocabulary for
reasoning-state changes (operators) and a pathology taxonomy with
operational tests. These are ported to `TRAJECTORY_VOCABULARY.md` (reference
tier) rather than left in the archive, because they name *what a span does
to the working state* and *how trajectories fail* — reusable in the
prospective approach as annotation language and as names for what the
controller might learn to nudge. The legacy prompts themselves remain in
`ace-legacyapproach/prompts/` as the original source.

## Lab-note pointer notes

The four lab notes under `lab_notes/` were written before the 2026-08-24
code reorg. They reference old script filenames (`teacherforced_scan.py`,
`teacherforced_map.py`, `branch_ledger.py`, etc.) that have since been
renamed and moved. Pointer fixes are applied in-place; the notes' content
is unchanged (they are historical evidence records — never rewrite
history). Current filenames:
- `teacherforced_scan.py` → `analysis/entropy_scan.py`
- `teacherforced_map.py` → `analysis/residual_map.py`
- `teacherforced_analyze.py` → `analysis/analyze_scan.py`
- `teacherforced_map_analyze.py` → `analysis/analyze_map.py`
- `branch_ledger.py` → `analysis/branch_ledger.py` (moved, not renamed)
- `trace_diagnostics.py` → `core/trace_diagnostics.py`
- `collect_qwen_frontier_30.py` → `frontier/collect_rollouts.py`
