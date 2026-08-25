"""Bounded sequence construction with late-interaction constraints.

The solver must ARRANGE N items under positional/relational constraints
accreted until only 1-2 orders survive, optionally within a
transition-cost budget. Unlike assign/certify (verify or refute a
candidate object), construction makes the model choose an order
sequentially: plausible partial orders dead-end late and must be
backtracked — the explore/prune texture the b2 pool lacked (b2 trace
read: committed search mostly absent outside grid/certify).

Knobs:
    n_items   5-8    items to arrange
    budget    bool   transition-cost budget in force (feasibility budget)
    none_prob 0-0.4  fraction unsatisfiable (planted before-cycle)
"""
from __future__ import annotations

import random
import re

from .common import Problem, answer_text, norm_ws

ITEMS = ["Ada", "Bo", "Cy", "Di", "Ed", "Fay", "Gus", "Hal"]

SKINS = [
    {"intro": "A ceremony committee must fix the processional order of "
              "{n} delegates", "pos_noun": "position"},
    {"intro": "A photo studio must fix the lineup order of {n} delegates",
     "pos_noun": "spot"},
    {"intro": "A relay captain must fix the running order of {n} "
              "runners", "pos_noun": "leg"},
]


def _decide(c, pos):
    """True/False if constraint c is decidable from partial pos, else None."""
    k = c[0]
    if k == "before":
        if c[1] in pos and c[2] in pos:
            return pos[c[1]] < pos[c[2]]
        return None
    if k == "at":
        return pos.get(c[1]) == c[2] if c[1] in pos else None
    if k == "notat":
        return pos.get(c[1]) != c[2] if c[1] in pos else None
    if k == "adj":
        if c[1] in pos and c[2] in pos:
            return abs(pos[c[1]] - pos[c[2]]) == 1
        return None
    if k == "notadj":
        if c[1] in pos and c[2] in pos:
            return abs(pos[c[1]] - pos[c[2]]) != 1
        return None
    if k == "parity":
        if c[1] in pos:
            return pos[c[1]] % 2 == c[2]
        return None
    raise ValueError(k)


def _ok(order, cons):
    pos = {name: i + 1 for i, name in enumerate(order)}
    return all(_decide(c, pos) is not False for c in cons)


def _count(items, cons, limit=3):
    cnt = 0
    order = []

    def rec():
        nonlocal cnt
        if cnt >= limit:
            return
        if len(order) == len(items):
            cnt += 1
            return
        for it in items:
            if it in order:
                continue
            order.append(it)
            if _ok(order, cons):
                rec()
            order.pop()

    rec()
    return cnt


def _order_cost(order, costs):
    return sum(costs.get(",".join(sorted((order[i], order[i + 1]))), 0)
               for i in range(len(order) - 1))


def _phrased(c, pos_noun):
    k = c[0]
    if k == "before":
        return f"{c[1]} must stand somewhere before {c[2]}."
    if k == "at":
        return f"{c[1]} stands in {pos_noun} {c[2]}."
    if k == "notat":
        return f"{c[1]} does not stand in {pos_noun} {c[2]}."
    if k == "adj":
        return f"{c[1]} and {c[2]} must be adjacent."
    if k == "notadj":
        return f"{c[1]} and {c[2]} must not be adjacent."
    return f"{c[1]} stands in an {'odd' if c[2] == 1 else 'even'} {pos_noun}."


def make(rng: random.Random, knobs: dict) -> Problem:
    n = knobs["n_items"]
    budget = bool(knobs.get("budget", False))
    want_none = rng.random() < knobs.get("none_prob", 0.2)
    items = ITEMS[:n]

    for _attempt in range(150):
        perm = rng.sample(items, n)

        # solution-consistent constraint pool
        pool = []
        for i, a in enumerate(perm):
            pool.append(("at", a, i + 1))
            for p in range(1, n + 1):
                if p != i + 1:
                    pool.append(("notat", a, p))
            pool.append(("parity", a, (i + 1) % 2))
        for i, a in enumerate(perm):
            for j, b in enumerate(perm):
                if i >= j:
                    continue
                pool.append(("before", a, b))
                if j == i + 1:
                    pool.append(("adj", a, b))
                else:
                    pool.append(("notadj", a, b))

        costs = {}
        B = None
        if budget:
            for i, a in enumerate(items):
                for b in items[i + 1:]:
                    if rng.random() < 0.4:
                        costs[f"{a},{b}"] = rng.randrange(1, 4)
            B = _order_cost(perm, costs)

        if want_none:
            # noise constraints consistent with perm, then plant a
            # before-cycle: provably unsatisfiable
            rng.shuffle(pool)
            cons = [c for c in pool if c[0] not in ("before",)][:n]
            cyc = rng.sample(items, 3)
            cons += [("before", cyc[0], cyc[1]), ("before", cyc[1], cyc[2]),
                     ("before", cyc[2], cyc[0])]
            if _count(items, cons) != 0:
                continue
            ans = "NONE"
        else:
            # force one long before-chain (late-interaction texture),
            # then accrete until near-unique
            cons = [("before", perm[i], perm[i + 1])
                    for i in range(min(3, n - 1))]
            rest = [c for c in pool if c not in cons]
            rng.shuffle(rest)
            for c in rest:
                cons.append(c)
                if _count(items, cons, limit=3) <= 2:
                    break
            if _count(items, cons, limit=3) > 2:
                continue
            # prune redundancies that keep the count unchanged
            for c in list(cons):
                trial = [x for x in cons if x is not c]
                if _count(items, trial, limit=3) <= 2:
                    cons = trial
            if not (n <= len(cons) <= 2 * n + 2):
                continue
            ans = ", ".join(perm)

        skin = rng.choice(SKINS)
        lines = "\n".join(f"  {i+1}. {_phrased(c, skin['pos_noun'])}"
                          for i, c in enumerate(cons))
        prose = (
            skin["intro"].format(n=n) +
            f": {', '.join(items)}. Every condition below must hold.\n\n"
            f"Conditions:\n{lines}"
        )
        if budget:
            cost_lines = ", ".join(
                f"{k.replace(',', '-')}:{v}" for k, v in sorted(costs.items())
                if v)
            prose += (
                f"\n\nTransition friction: moving directly between two "
                f"items costs points — {cost_lines} (unlisted pairs cost "
                f"0). The total friction of the whole order must be at "
                f"most {B} points."
            )

        return Problem(
            family="construct", prose=prose,
            question=("Give an order satisfying every condition"
                      + (f" within the {B}-point budget" if budget else "")
                      + ", as a comma-separated list of all "
                      f"{n} names, or state NONE if no such order exists."),
            answer=ans,
            verifier_kind="constraint_checker",
            objective_task="constraint_satisfaction",
            search_topology="bounded_construction",
            knobs={"n_items": n, "budget": budget, "items": items,
                   "cons": [list(c) for c in cons],
                   "costs": costs or {}, "B": B,
                   "solvable": not want_none},
            seed=rng.randrange(2**31))
    raise RuntimeError("construct: no instance found")


def _parse_order(text, items):
    """Ordered extraction of item names; tolerant to numbering/repeats."""
    t = norm_ws(text)
    names = [it.lower() for it in items]
    ms = []
    for it in names:
        for m in re.finditer(r"\b%s\b" % re.escape(it), t):
            ms.append((m.start(), it))
    ms.sort()
    seq = [x[1] for x in ms]
    n = len(items)
    for cand in ((seq[-n:] if len(seq) >= n else None),
                 (seq[:n] if len(seq) >= n else None),
                 (seq if len(seq) == n else None)):
        if cand and sorted(cand) == sorted(names):
            # restore original casing: constraints/costs use it
            return [items[names.index(x)] for x in cand]
    return None


def check(completion: str, reference: str, knobs: dict | None = None) -> bool:
    text = answer_text(completion)
    if reference == "NONE":
        return "none" in text.lower()
    if not knobs or "cons" not in knobs:
        return False
    got = _parse_order(text, knobs["items"])
    if got is None:
        return False
    if not _ok(got, [tuple(c) for c in knobs["cons"]]):
        return False
    if knobs.get("budget") and knobs.get("B") is not None:
        if _order_cost(got, knobs["costs"]) > knobs["B"]:
            return False
    return True
