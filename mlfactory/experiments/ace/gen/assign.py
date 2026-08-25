"""Constraint assignment: assign n items to m bins under stated constraints.

Reasoning-shaped generalization of resource_scheduling / access_policy_rules.
Solver-built: sample a target assignment, accrete constraints until the
solution is unique, optionally present the decisive constraint LAST so a
seductive decoy satisfies every earlier rule (delayed_constraint_conflict).

Knobs:
    n_items   5-8     variables to assign
    n_bins    3-5     assignment targets
    delayed   bool    present decisive constraint last (decoy trap)
"""
from __future__ import annotations

import random
import re

from .common import Problem, answer_text

STAFF = ["Ana", "Bo", "Cy", "Di", "Ed", "Fay", "Gus", "Hal"]
FLAVOR = ["Intake", "Safety", "Quality", "Logistics",
          "Compliance", "Systems", "Customer", "Audit"]

# Surface skins: venue framing varies; session ids (S#), staff names, and
# rule phrasings are pinned because the verifier parses S#=Name pairs.
SKINS = [
    "A training center must staff {n} sessions, S1 through S{n}.",
    "A dispatch office must cover {n} service calls, S1 through S{n}.",
    "A conference team must schedule {n} track sessions, S1 through S{n}.",
]


def _violates(con, a):
    """True if complete/partial assignment a violates constraint con."""
    kind = con[0]
    if kind == "cap":
        _, b, c = con
        return sum(1 for x in a if x == b) > c
    if kind == "diff":
        _, i, j = con
        return a[i] is not None and a[j] is not None and a[i] == a[j]
    if kind == "same":
        _, i, j = con
        return a[i] is not None and a[j] is not None and a[i] != a[j]
    if kind == "not":
        _, i, b = con
        return a[i] is not None and a[i] == b
    if kind == "imp":
        _, i, b1, j, b2 = con
        return (a[i] is not None and a[j] is not None
                and a[i] == b1 and a[j] != b2)
    raise ValueError(kind)


def _solutions(n, m, cons, limit=2):
    sols, a = [], [None] * n

    def rec(i):
        if len(sols) >= limit:
            return
        if i == n:
            sols.append(tuple(a))
            return
        for b in range(m):
            a[i] = b
            if not any(_violates(c, a) for c in cons):
                rec(i + 1)
        a[i] = None

    rec(0)
    return sols


def _phrased(con, names):
    kind = con[0]
    if kind == "cap":
        return f"{names[con[1]]} leads at most {con[2]} sessions."
    if kind == "diff":
        return f"S{con[1]+1} and S{con[2]+1} must have different leads."
    if kind == "same":
        return f"S{con[1]+1} and S{con[2]+1} must have the same lead."
    if kind == "not":
        return f"{names[con[2]]} is not certified for S{con[1]+1}."
    _, i, b1, j, b2 = con
    return (f"If {names[b1]} leads S{i+1}, "
            f"then {names[b2]} must lead S{j+1}.")


def make(rng: random.Random, knobs: dict) -> Problem:
    n, m = knobs["n_items"], knobs["n_bins"]
    names = STAFF[:m]
    for _attempt in range(200):
        t = tuple(rng.randrange(m) for _ in range(n))
        if len(set(t)) < 2:
            continue
        pool = []
        for i in range(n):
            for j in range(i + 1, n):
                if t[i] != t[j]:
                    pool.append(("diff", i, j))
                else:
                    pool.append(("same", i, j))
                pool.append(("imp", i, t[i], j, t[j]))
                pool.append(("imp", j, t[j], i, t[i]))
            for b in range(m):
                if b != t[i]:
                    pool.append(("not", i, b))
        for b in range(m):
            cnt = t.count(b)
            for c in range(cnt, n):
                pool.append(("cap", b, c))
        rng.shuffle(pool)

        cons = []
        for c in pool:
            cons.append(c)
            if len(_solutions(n, m, cons)) == 1:
                break
        if len(_solutions(n, m, cons)) != 1:
            continue
        # prune redundancies (keep presentation lean)
        for c in list(cons):
            rest = [x for x in cons if x is not c]
            if len(_solutions(n, m, rest)) == 1:
                cons = rest
        if not (3 <= len(cons) <= n + m + 2):
            continue

        order = list(cons)
        if knobs.get("delayed"):
            # decisive constraint last: removing it must admit a decoy
            cands = [c for c in cons
                     if len(_solutions(n, m, [x for x in cons if x is not c],
                                       limit=2)) > 1]
            if cands:
                last = rng.choice(cands)
                order = [x for x in cons if x is not last] + [last]
        else:
            rng.shuffle(order)

        lines = "\n".join(f"  {k+1}. {_phrased(c, names)}"
                          for k, c in enumerate(order))
        skin = rng.choice(SKINS)
        prose = (
            skin.format(n=n) + " " +
            f"Each session is led by exactly one of {m} staff: "
            f"{', '.join(names)}. Every rule below must hold.\n\n"
            f"Sessions: " + ", ".join(
                f"S{i+1} {FLAVOR[i]}" for i in range(n)) +
            f"\n\nRules:\n{lines}"
        )
        ans = ", ".join(f"S{i+1}={names[t[i]]}" for i in range(n))
        return Problem(
            family="assign", prose=prose,
            question=f"Which staff member leads each session S1-S{n}? "
                     "Give the complete assignment as S#=Name pairs.",
            answer=ans,
            verifier_kind="exact_assignment",
            objective_task="constraint_satisfaction",
            search_topology=("delayed_constraint_conflict"
                             if knobs.get("delayed") else "competing_hypotheses"),
            knobs={"n_items": n, "n_bins": m, "delayed": bool(knobs.get("delayed")),
                   "n_constraints": len(order)},
            seed=rng.randrange(2**31))
    raise RuntimeError("assign: no unique-solution instance found")


# Tolerant to separator (":" vs "=") and case in the model's S#/Name pairs.
# Structural validation unchanged: the full slot->name mapping must equal the
# reference exactly.
_PAIR_RE = re.compile(r"(S\d+)\s*[:=]\s*([A-Za-z]+)", re.IGNORECASE)


def check(completion: str, reference: str, knobs: dict | None = None) -> bool:
    def parse(s):
        return {slot.upper(): name.lower()
                for slot, name in _PAIR_RE.findall(s)}
    got = parse(answer_text(completion))
    want = parse(reference)
    return bool(want) and got == want
