"""ACE problem generator: reasoning-shaped, solver-verified, knob-tunable.

Fresh task-specific code (NOT madlibz). Madlibz generates general prompts
via an authoring model; this package generates problems *programmatically*
so every reference answer is exact by construction and every envelope knob
is recorded for calibration joins.

Design contract (from ACE reward analysis):

1. Reasoning-shaped, not computation-shaped. No enumeration ("count all"),
   no VM simulation, no exponential optimization. Families produce bounded
   search, state replay with decisive discrepancies, small-hypothesis
   elimination, and certification tasks.

2. Terminal verifiability. Every family has a strict check(); the probe
   collector's soft substring scoring is advisory only — calibrate.py
   re-scores completions with the strict family verifier.

3. Difficulty knobs. Each family exposes scalar knobs (sizes, trap flags).
   The calibration loop measures per-prompt success under N rollouts and
   accepts only band members (LIVE = 1..6 of 8, prefer 2..5), then joins
   outcomes back to knobs to steer regeneration.

4. Fork-point density. Topologies that create states where premature
   commitment loses: delayed constraint conflicts, competing hypotheses,
   deceptive greedy orderings, guarded-transition traps.

Families:
    assign      constraint assignment (n items -> m bins, unique solution)
    machine     guarded state machine replay (registers + rejected events)
    adversary   bounded counterexample search (shortest violating sequence)
    certify     graph coloring certification (produce one or prove NONE)
    grid        zebra-style logic grid (clues until unique)
    hypothesis  numeric hypothesis elimination (disputed record)
"""
