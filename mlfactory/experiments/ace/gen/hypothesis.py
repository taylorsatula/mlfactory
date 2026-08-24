"""Numeric hypothesis elimination: one disputed record, N candidate
interpretations, and a closing figure that only one hypothesis reconciles.

Reasoning-shaped replacement for the oversized ledger_reconciliation domain
(which buried the solver in 8-record bookkeeping). Here the arithmetic is
small, the structure is pure competing_hypotheses: each rival hypothesis is
refuted by the closing count; the survivor must be identified AND the exact
closing balance derived.

Knobs:
    n_sales    4-6    completed sale records
    n_payouts  1-3    payout records
    spread     >=12   minimum $ gap between hypothesis values
"""
from __future__ import annotations

import random
import re

from .common import Problem, answer_text, money, parse_money


def make(rng: random.Random, knobs: dict) -> Problem:
    ns, npay = knobs["n_sales"], knobs["n_payouts"]
    spread = knobs["spread"]

    for _attempt in range(300):
        opening = rng.choice([10000, 15000, 20000, 25000])
        sales = [rng.randrange(1200, 6500) for _ in range(ns)]
        payouts = [rng.randrange(500, 4000) for _ in range(npay)]
        true_drop = rng.randrange(8000, 22000)
        rivals = set()
        while len(rivals) < 2:
            r = true_drop + rng.choice([-1, 1]) * rng.randrange(spread, 4 * spread) * 100 // 1
            if 4000 < r != true_drop:
                rivals.add(r)
        cands = sorted(rivals | {true_drop})
        true_idx = cands.index(true_drop)

        subtotal = opening + sum(sales) - sum(payouts)
        counted = subtotal - true_drop          # true hypothesis balances
        # rivals must NOT balance
        if any(subtotal - c == counted for c in cands if c != true_drop):
            continue

        sale_lines = "\n".join(
            f"  T{101+i} sale {money(a)} completed"
            for i, a in enumerate(sales))
        pay_lines = "\n".join(
            f"  P{201+i} payout {money(a)} completed"
            for i, a in enumerate(payouts))
        hyp_lines = "\n".join(
            f"  H{i+1}: the drop slip reads {money(c)}"
            for i, c in enumerate(cands))
        prose = (
            f"You are closing Register {rng.randrange(2, 6)}. The deposit "
            f"must balance to the cent. All records for the shift:\n\n"
            f"  Opening float: {money(opening)}\n"
            f"{sale_lines}\n{pay_lines}\n"
            f"  Drop slip D1: illegible\n"
            f"  Counted drawer at close: {money(counted)}\n\n"
            f"Three readings of the drop slip have been proposed:\n"
            f"{hyp_lines}\n\n"
            f"Exactly one reading makes expected cash equal counted cash."
        )
        expected = subtotal - cands[true_idx]
        over_short = counted - expected
        ans = (f"H{true_idx+1} expected={money(expected)} "
               f"over_short={money(over_short)}")
        return Problem(
            family="hypothesis", prose=prose,
            question="Which hypothesis is consistent with every record, and "
                     "what are the exact expected cash and over/short under "
                     "it? (format: H# expected=$X.XX over_short=$Y.YY)",
            answer=ans,
            verifier_kind="hypothesis_elimination",
            objective_task="evidence_reconciliation",
            search_topology="competing_hypotheses",
            knobs={"n_sales": ns, "n_payouts": npay, "spread": spread,
                   "n_hypotheses": len(cands),
                   "true_hypothesis": true_idx + 1},
            seed=rng.randrange(2**31))
    raise RuntimeError("hypothesis: no instance found")


_ANS_RE = re.compile(
    r"h\s*(\d+).*?expected\s*=\s*\$?(-?[\d,]+\.\d{2}).*?"
    r"over_?short\s*=\s*\$?(-?[\d,]+\.\d{2})",
    re.IGNORECASE | re.DOTALL)


def check(completion: str, reference: str, knobs: dict | None = None) -> bool:
    def parse(s):
        m = _ANS_RE.search(s)
        if not m:
            return None
        return (int(m.group(1)),
                parse_money(m.group(2).replace(",", "")),
                parse_money(m.group(3).replace(",", "")))
    got = parse(answer_text(completion))
    want = parse(reference)
    return got is not None and got == want
