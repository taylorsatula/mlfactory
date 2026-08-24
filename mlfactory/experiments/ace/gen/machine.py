"""Guarded state machine replay with registers and rejected events.

Reasoning-shaped state replay (generalizes state_machine_workflow) with a
decisive discrepancy: the event log always contains 1-3 events whose guard
FAILS, so blind transition-following gives the wrong answer. The solver
must evaluate guards against the register state at each step.

Knobs:
    n_states   4-6    state count
    n_events   5-7    distinct event types
    log_len    10-16  events in the log
"""
from __future__ import annotations

import random
import re

from .common import Problem, answer_text, norm_ws

STATES = ["INIT", "READY", "ACTIVE", "PAUSED", "FAULT", "DONE"]
EVENTS = ["HELLO", "AUTH", "OPEN", "PAUSE", "RESUME",
          "TICKET", "CLOSE", "RESET", "SYNC", "PING"]
GUARDS = [[], ["A"], ["not A"], ["T"], ["not T"], ["n >= 1"], ["n == 0"],
          ["A", "T"], ["A", "n >= 1"], ["not T", "n >= 1"], ["n == 0", "not A"]]


def _guard_ok(lits, A, T, n):
    for g in lits:
        if g == "A" and not A:
            return False
        if g == "not A" and A:
            return False
        if g == "T" and not T:
            return False
        if g == "not T" and T:
            return False
        if g == "n >= 1" and n < 1:
            return False
        if g == "n == 0" and n != 0:
            return False
    return True


def _simulate(rules, log):
    s, A, T, n = 0, False, False, 0
    rejected = []
    for idx, e in enumerate(log, 1):
        r = rules[e]
        if s not in r["allowed"] or not _guard_ok(r["guard"], A, T, n):
            rejected.append(idx)
            continue
        if r["setA"] is not None:
            A = r["setA"]
        if r["setT"] is not None:
            T = r["setT"]
        n += r["dn"]
        s = r["goto"]
    return s, A, T, n, rejected


def make(rng: random.Random, knobs: dict) -> Problem:
    ns, ne = knobs["n_states"], knobs["n_events"]
    ll = knobs["log_len"]
    states, events = STATES[:ns], EVENTS[:ne]

    for _attempt in range(100):
        rules = {}
        for e in events:
            rules[e] = {
                # bias toward permissive rules so most events fire; the
                # trap is the 1-3 that don't
                "allowed": sorted(rng.sample(range(ns),
                                             rng.randint(2, min(4, ns)))),
                "guard": list(rng.choice(GUARDS)),
                "setA": rng.choice([None, None, True, False]),
                "setT": rng.choice([None, None, True, False]),
                "dn": rng.choice([-1, 0, 0, 1, 1, 2]),
                "goto": rng.randrange(ns),
            }
        # many logs per rule table: want 1-3 rejections, >=6 accepted,
        # and a final state different from the initial one
        for _log_try in range(300):
            log = [rng.choice(events) for _ in range(ll)]
            s, A, T, n, rejected = _simulate(rules, log)
            accepted = ll - len(rejected)
            if 1 <= len(rejected) <= 3 and accepted >= 6 and s != 0:
                break
        else:
            continue

        rule_lines = []
        for e in events:
            r = rules[e]
            parts = [f"allowed in {', '.join(states[i] for i in r['allowed'])}"]
            if r["guard"]:
                parts.append("requires " + " and ".join(r["guard"]))
            eff = []
            if r["setA"] is not None:
                eff.append(f"A={'true' if r['setA'] else 'false'}")
            if r["setT"] is not None:
                eff.append(f"T={'true' if r['setT'] else 'false'}")
            if r["dn"]:
                eff.append(f"n {'+' if r['dn'] > 0 else '-'} {abs(r['dn'])}")
            eff.append(f"go {states[r['goto']]}")
            rule_lines.append(f"  {e}: {'; '.join(parts)}. "
                              f"Effects: {', '.join(eff)}.")
        log_str = "  ".join(f"{i+1}:{e}" for i, e in enumerate(log))
        prose = (
            f"A session controller has states {', '.join(states)} and three "
            "registers: boolean A, boolean T, integer n. Initial state: "
            f"{states[0]}, A=false, T=false, n=0.\n\n"
            "An event fires only if the current state is in its allowed list "
            "AND its guard holds (guards are evaluated against register "
            "values just before the event). A fired event applies its effects "
            "in order. Any other event is REJECTED: nothing changes and the "
            "event is counted.\n\n"
            "Transition rules:\n" + "\n".join(rule_lines) +
            f"\n\nEvent log ({ll} events, in order):\n  {log_str}"
        )
        ans = (f"final={states[s]} A={'true' if A else 'false'} "
               f"T={'true' if T else 'false'} n={n} "
               f"rejected={len(rejected)} first_rejected={rejected[0]}")
        return Problem(
            family="machine", prose=prose,
            question="Replay the full log. Report the final configuration "
                     "(state, A, T, n), how many events were rejected, and "
                     "the log index of the first rejected event.",
            answer=ans,
            verifier_kind="state_transition_replay",
            objective_task="algorithm_trace",
            search_topology="adversarial_edge_case",
            knobs={"n_states": ns, "n_events": ne, "log_len": ll,
                   "n_rejected": len(rejected)},
            seed=rng.randrange(2**31))
    raise RuntimeError("machine: no acceptable instance found")


# Tolerant to the model's natural field labels: "State=" for "final=",
# "first rejected=" for "first_rejected=", and ":" separators. Structural
# validation (all six fields, exact tuple match) is unchanged.
_FIELD_RE = re.compile(
    r"(?:final|state)\s*[=:]\s*([a-z]+).*?\ba\s*[=:]\s*(true|false).*?"
    r"\bt\s*[=:]\s*(true|false).*?\bn\s*[=:]\s*(-?\d+).*?"
    r"\brejected\s*[=:]\s*(\d+).*?first[_ ]rejected\s*[=:]\s*(\d+)",
    re.IGNORECASE | re.DOTALL)


def check(completion: str, reference: str, knobs: dict | None = None) -> bool:
    got = _FIELD_RE.search(norm_ws(answer_text(completion)))
    want = _FIELD_RE.search(norm_ws(reference))
    return bool(got and want) and got.groups() == want.groups()
