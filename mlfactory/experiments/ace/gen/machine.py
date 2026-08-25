"""Guarded state machine replay with registers and rejected events.

Reasoning-shaped state replay (generalizes state_machine_workflow) with a
decisive discrepancy: the event log always contains 1..max(3, log_len//3)
events whose guard FAILS, so blind transition-following gives the wrong
answer. The solver must evaluate guards against the register state at
each step.

Knobs:
    n_states   4-6    state count
    n_events   5-7    distinct event types
    log_len    10-18  events in the log (rejection window scales: 1..max(3, ll//3))
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

# Surface skins: the machine's *story* varies, the transition table, log,
# and answer vocabulary do not (the verifier parses state/register names).
SKINS = [
    {
        "intro": "A session controller",
        "log_noun": "Event log",
        "register_note": "",
    },
    {
        "intro": "An order-processing pipeline",
        "log_noun": "Event log",
        "register_note": " The flags and counter track per-order "
                         "bookkeeping as events fire.",
    },
    {
        "intro": "A device firmware state machine",
        "log_noun": "Signal log",
        "register_note": "",
    },
]


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
    # rejection window scales with log length: longer logs reject more
    # events in expectation, so a fixed 1-3 cap makes ll>15 infeasible.
    rej_hi = max(3, ll // 3)

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
        # many logs per rule table: want 1..rej_hi rejections, >=6
        # accepted, and a final state different from the initial one
        for _log_try in range(300):
            log = [rng.choice(events) for _ in range(ll)]
            s, A, T, n, rejected = _simulate(rules, log)
            accepted = ll - len(rejected)
            if 1 <= len(rejected) <= rej_hi and accepted >= 6 and s != 0:
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
        skin = rng.choice(SKINS)
        prose = (
            f"{skin['intro']} has states {', '.join(states)} and three "
            "registers: boolean A, boolean T, integer n. Initial state: "
            f"{states[0]}, A=false, T=false, n=0." + skin["register_note"] +
            "\n\n"
            "An event fires only if the current state is in its allowed list "
            "AND its guard holds (guards are evaluated against register "
            "values just before the event). A fired event applies its effects "
            "in order. Any other event is REJECTED: nothing changes and the "
            "event is counted.\n\n"
            "Transition rules:\n" + "\n".join(rule_lines) +
            f"\n\n{skin['log_noun']} ({ll} events, in order):\n  {log_str}"
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


# Tolerant to the model's natural field labels and separators: "State=" for
# "final=", "first rejected[=:] index", "firstindex=", count-before-label
# ("3 rejected"), and bare positional answers ("ready, false, false, 0, 2, 11").
# Structural validation is unchanged: all six fields must be present and the
# (state, A, T, n, rejected, first_rejected) tuple must equal the reference
# exactly — only label/serialization tolerance is loosened, never semantics.
_STATE_ALT = "|".join(STATES).lower()


def _grab(label, val, t):
    m = re.search(label + r"\s*[=:]\s*" + val, t)
    return m.group(1) if m else None


def _fields(ans: str):
    """Extract (state, A, T, n, rejected, first_rejected) tolerantly."""
    t = norm_ws(ans)
    m = re.search(r"(?:final|state|configuration)\s*[=:]?\s*(%s)" % _STATE_ALT, t)
    state = m.group(1) if m else None
    if state is None:
        m = re.search(r"\b(%s)\b" % _STATE_ALT, t)
        state = m.group(1) if m else None
    a = _grab(r"\ba\b", r"(true|false)", t)
    T = _grab(r"\bt\b", r"(true|false)", t)
    n = _grab(r"\bn\b", r"(-?\d+)", t)
    if state is not None and a is None and T is None:
        m = re.search(re.escape(state) +
                      r"\s*[,;]\s*(true|false)\s*[,;]\s*(true|false)\s*[,;]\s*(-?\d+)", t)
        if m:
            a, T, n = m.group(1), m.group(2), m.group(3)
    rej = None
    # negative lookbehind: 'rejected: N' inside 'first rejected: N' is the
    # first_rejected field, not the rejection count (extraction F/N source).
    # 'rejections' accepted as a synonymous label (extraction looseness only).
    m = re.search(r"(?<!first[ _])\breject(?:ed|ions?)\b(?:\s*events)?\s*[=:]\s*(\d+)", t)
    if m:
        rej = m.group(1)
    else:
        m = re.search(r"\b(\d+)\s+(?:events\s+)?rejected\b", t)
        if m:
            rej = m.group(1)
    first = None
    for pat in (r"first[_ ]?re(?:jected|jection|jects)?[_ ]?(?:index|idx|position|event)?\s*[=:]\s*(\d+)",
                r"first[_ ]?(?:index|idx|position)\s*[=:]\s*(\d+)",
                r"\b(?:index|idx)\s*[=:]\s*(\d+)"):
        m = re.search(pat, t)
        if m:
            first = m.group(1)
            break
    if rej is None or first is None:
        if n is not None:
            mn = re.search(r"\bn\b\s*[=:]\s*-?\d+", t)
            start = mn.end() if mn else 0
            tail = re.findall(r"-?\d+", t[start:])
            if rej is None and first is None and len(tail) >= 2:
                rej, first = tail[-2], tail[-1]
            elif rej is None and tail:
                rej = tail[0]
            elif first is None and tail:
                first = tail[-1]
    return (state, a, T, n, rej, first)


def check(completion: str, reference: str, knobs: dict | None = None) -> bool:
    got = _fields(answer_text(completion))
    want = _fields(answer_text(reference))
    return all(v is not None for v in got) and got == want
