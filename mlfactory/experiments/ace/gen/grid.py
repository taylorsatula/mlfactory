"""Zebra-style logic grid: positions x categories, clues until unique.

Reasoning-shaped constraint propagation (generalizes logic_records).
Clues are accreted from a solution-consistent pool until backtracking
proves uniqueness; redundant clues are then pruned so every remaining
clue is load-bearing (cross_source_reconciliation structure).

Knobs:
    n_pos   4-6   shelf positions (categories scale with it)
"""
from __future__ import annotations

import random
import re

from .common import Problem, answer_text

POOLS = {
    "name": ["Ash", "Birch", "Cedar", "Elm", "Holly", "Maple"],
    "year": ["1711", "1712", "1713", "1714", "1715", "1716"],
    "material": ["vellum", "leather", "cloth", "linen", "rag", "palm"],
}


def _clue_vars(cl):
    if cl[0] in ("at", "notat"):
        return [(cl[1], cl[2])]
    return [(cl[1], cl[2]), (cl[3], cl[4])]


def _clue_status(cl, pos):
    """True/False if decidable from partial assignment, None otherwise."""
    vs = _clue_vars(cl)
    if any(v not in pos for v in vs):
        return None
    p = [pos[v] for v in vs]
    k = cl[0]
    if k == "at":
        return p[0] == cl[3]
    if k == "notat":
        return p[0] != cl[3]
    if k == "samepos":
        return p[0] == p[1]
    if k == "immleft":
        return p[0] + 1 == p[1]
    if k == "leftof":
        return p[0] < p[1]
    raise ValueError(k)


def _count(P, cats, clues, limit=2):
    vars_ = [(c, v) for c in cats for v in cats[c]]
    assign = {}
    sols = 0

    def legal(var, p):
        c = var[0]
        for (c2, _), p2 in assign.items():
            if c2 == c and p2 == p:
                return False
        assign[var] = p
        ok = all(_clue_status(cl, assign) is not False for cl in clues)
        del assign[var]
        return ok

    def rec():
        nonlocal sols
        if sols >= limit:
            return
        if len(assign) == len(vars_):
            sols += 1
            return
        best, best_opts = None, None
        for var in vars_:
            if var in assign:
                continue
            opts = [p for p in range(1, P + 1) if legal(var, p)]
            if not opts:
                return
            if best_opts is None or len(opts) < len(best_opts):
                best, best_opts = var, opts
        for p in best_opts:
            assign[best] = p
            rec()
            del assign[best]

    rec()
    return sols


def _phrased(cl):
    k = cl[0]
    if k == "at":
        return f"The {cl[2]} record stands in slot {cl[3]}."
    if k == "notat":
        return f"The {cl[2]} record is not in slot {cl[3]}."
    if k == "samepos":
        return f"The {cl[2]} record and the {cl[4]} record share a slot."
    if k == "immleft":
        return (f"The {cl[2]} record stands immediately left of "
                f"the {cl[4]} record.")
    return (f"The {cl[2]} record stands somewhere left of "
            f"the {cl[4]} record.")


def make(rng: random.Random, knobs: dict) -> Problem:
    P = knobs["n_pos"]
    cats = {c: POOLS[c][:P] for c in POOLS}

    for _attempt in range(200):
        sol = {}
        for c, vals in cats.items():
            perm = list(range(1, P + 1))
            rng.shuffle(perm)
            for v, p in zip(vals, perm):
                sol[(c, v)] = p

        pool = []
        for c, vals in cats.items():
            for v in vals:
                pool.append(("at", c, v, sol[(c, v)]))
                for p in range(1, P + 1):
                    if p != sol[(c, v)]:
                        pool.append(("notat", c, v, p))
        items = [(c, v) for c in cats for v in cats[c]]
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if a[0] == b[0]:
                    continue
                pa, pb = sol[a], sol[b]
                if pa == pb:
                    pool.append(("samepos", *a, *b))
                    pool.append(("samepos", *b, *a))
                elif pa < pb:
                    pool.append(("leftof", *a, *b))
                    if pa + 1 == pb:
                        pool.append(("immleft", *a, *b))
                else:
                    pool.append(("leftof", *b, *a))
                    if pb + 1 == pa:
                        pool.append(("immleft", *b, *a))
        rng.shuffle(pool)

        clues = []
        for cl in pool:
            clues.append(cl)
            if _count(P, cats, clues) == 1:
                break
        if _count(P, cats, clues) != 1:
            continue
        for cl in list(clues):
            rest = [x for x in clues if x != cl]
            if _count(P, cats, rest) == 1:
                clues = rest
        if not (P + 1 <= len(clues) <= 3 * P):
            continue

        clue_lines = "\n".join(f"  {i+1}. {_phrased(c)}"
                               for i, c in enumerate(clues))
        prose = (
            f"{P} bound records stand on a shelf in slots 1 (leftmost) "
            f"through {P}. Each has a distinct label "
            f"({', '.join(cats['name'])}), a distinct accession year "
            f"({', '.join(cats['year'])}), and a distinct binding "
            f"({', '.join(cats['material'])}). The labels are lost; a "
            f"ledger survives with these notes, all true:\n\n{clue_lines}"
        )
        ans = "; ".join(
            f"Slot {p}: " + ", ".join(
                next(v for v in cats[c] if sol[(c, v)] == p) for c in cats)
            for p in range(1, P + 1))
        return Problem(
            family="grid", prose=prose,
            question=f"For each slot 1-{P}, give its label, accession year, "
                     "and binding (format: Slot N: label, year, binding; "
                     "one per slot).",
            answer=ans,
            verifier_kind="truth_table_or_model_check",
            objective_task="constraint_satisfaction",
            search_topology="cross_source_reconciliation",
            knobs={"n_pos": P, "n_clues": len(clues)},
            seed=rng.randrange(2**31))
    raise RuntimeError("grid: no unique-solution instance found")


_SLOT_RE = re.compile(
    r"slot\s*(\d+)\s*[:\-]\s*([A-Za-z]+)\s*,\s*(\d+)\s*,\s*([A-Za-z ]+?)"
    r"(?=;|slot|$)", re.IGNORECASE)


def check(completion: str, reference: str, knobs: dict | None = None) -> bool:
    def parse(s):
        return {m.group(1): (m.group(2).strip().lower(), m.group(3),
                             m.group(4).strip().lower())
                for m in _SLOT_RE.finditer(s)}
    got = parse(answer_text(completion))
    want = parse(reference)
    return bool(want) and got == want
