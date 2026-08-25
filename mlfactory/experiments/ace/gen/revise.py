"""Two-stage evidence revision: four candidate readings of a torn
transfer slip, a batch-1 record set plus a clerk's soft note that
favors a DECOY, and a batch-2 fragment find that only the true reading
reconciles — including a foreign-account fragment that sums to a rival
reading and must be excluded.

Built for the texture the b2 pool lacked (b2 trace read): an early
provisional commitment the evidence later kills. Batch 1 genuinely
underdetermines — every reading survives it — so the clerk's note is
the only early discriminator, and it points the wrong way.

Knobs:
    n_records  3-6    uncontested shift records (sales + payouts)
    spread     >=12   minimum $ gap between candidate readings
    decoy      bool   clerk's soft note favors a rival reading
"""
from __future__ import annotations

import random
import re

from .common import Problem, answer_text, money, parse_money

SKINS = [
    "You are auditing Register {reg} at the store after close.",
    "You are auditing the till at the cafe after the evening shift.",
    "You are auditing ticket window {reg} after the last show.",
]


def make(rng: random.Random, knobs: dict) -> Problem:
    nrec = knobs["n_records"]
    spread = knobs["spread"]
    decoy = bool(knobs.get("decoy", True))

    for _attempt in range(300):
        opening = rng.choice([10000, 15000, 20000, 25000])
        ns = max(2, nrec - 1)
        npay = nrec - ns
        sales = [rng.randrange(1200, 6500) for _ in range(ns)]
        payouts = [rng.randrange(500, 4000) for _ in range(npay)]
        subtotal = opening + sum(sales) - sum(payouts)

        true_drop = rng.randrange(8000, 22000)
        if true_drop >= subtotal:
            continue   # a drop cannot exceed the drawer's cash
        rivals = set()
        while len(rivals) < 3:
            r = true_drop + rng.choice([-1, 1]) * rng.randrange(
                spread, 4 * spread) * 100
            if 4000 < r != true_drop and all(
                    abs(r - x) >= spread * 100 for x in rivals):
                rivals.add(r)
        cands = sorted(rivals | {true_drop})
        true_idx = cands.index(true_drop)

        # fragments: fragment A is compatible with every reading
        x = rng.randrange(1000, min(cands) - 100)
        y = true_drop - x
        if y <= 0:
            continue
        # foreign fragment sums with x to exactly one RIVAL reading
        rival_idx = rng.choice([i for i in range(4) if i != true_idx])
        z = cands[rival_idx] - x
        if z <= 0 or z == y:
            continue
        # decoy note favors a rival (the attractor)
        fav_idx = rng.choice([i for i in range(4) if i != true_idx])

        # batch 1 alone must not discriminate: all readings consistent
        final = subtotal - true_drop

        sale_lines = "\n".join(
            f"  T{101+i} sale {money(a)} completed"
            for i, a in enumerate(sales))
        pay_lines = "\n".join(
            f"  P{201+i} payout {money(a)} completed"
            for i, a in enumerate(payouts))
        hyp_lines = "\n".join(
            f"  H{i+1}: the drop was {money(c)}" for i, c in enumerate(cands))

        prose = (
            rng.choice(SKINS).format(reg=rng.randrange(2, 6)) +
            " The vault drop was logged, but the drop slip tore and "
            "its amount is disputed. All uncontested records for the "
            "shift:\n\n"
            f"  Opening float: {money(opening)}\n"
            f"{sale_lines}\n{pay_lines}\n\n"
            f"Four readings of the drop have been proposed:\n{hyp_lines}\n\n"
            + (f"The closing clerk's note says the drop 'looked like "
               f"about {money(cands[fav_idx])}'.\n\n" if decoy else "")
            + f"Fragment A of the torn slip survives; it shows "
              f"{money(x)}.\n\n"
            + "Later, behind the drawer, two more papers turn up:\n"
              f"  Fragment B: {money(y)}\n"
              f"  Fragment C: {money(z)} — stamped ACCT 7, a different "
              f"drawer.\n\n"
            + "Exactly one reading of the drop is consistent with all "
              "the evidence from this drawer. Under it, expected cash "
              "= opening + sales - payouts - drop."
        )
        ans = f"H{true_idx+1} final={money(final)}"
        return Problem(
            family="revise", prose=prose,
            question="Which reading survives all the evidence, and what "
                     "is the expected cash under it? (format: H# "
                     "final=$X.XX)",
            answer=ans,
            verifier_kind="hypothesis_elimination",
            objective_task="evidence_reconciliation",
            search_topology="staged_revision",
            knobs={"n_records": nrec, "spread": spread, "decoy": decoy,
                   "n_hypotheses": 4, "true_hypothesis": true_idx + 1,
                   "decoy_hypothesis": fav_idx + 1,
                   "trap_hypothesis": rival_idx + 1},
            seed=rng.randrange(2**31))
    raise RuntimeError("revise: no instance found")


_ANS_RE = re.compile(
    r"h\s*(\d+).*?final\s*[=:]\s*(-?\$?-?[\d,]+\.\d{2})",
    re.IGNORECASE | re.DOTALL)


def check(completion: str, reference: str, knobs: dict | None = None) -> bool:
    def parse(s):
        m = _ANS_RE.search(s)
        if not m:
            return None
        return (int(m.group(1)),
                parse_money(m.group(2).replace(",", "")))
    got = parse(answer_text(completion))
    want = parse(reference)
    return got is not None and got == want
