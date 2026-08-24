#!/usr/bin/env python3
"""Small set of objectively verifiable reasoning problems (local, no downloads).

Deviation note: the brief asked for problems "already available locally". The
only local corpora are (a) legacy ACE advice traces with no gold answers and
(b) coding prompts with no local execution harness — neither supports
objective correctness checking. This module instead defines 9 template
families of multi-step arithmetic word problems with programmatically
computed gold answers: 6 families x 4 instances = 24 training prompts,
3 disjoint holdout families x 4 instances = 12 holdout prompts. All instances
are drawn with fixed seeds; building the set twice yields identical problems.

Problems are deliberately multi-constraint (staged rates, waits, leftover
work, waste, red-herring quantities) so the frozen 9B base is not already
at ceiling — GRPO needs mixed group rewards.

Answer protocol: the model must end its response with "Answer: <number>".
Verification is exact numeric comparison (2-decimal tolerance), no judge.
"""
from __future__ import annotations

import math
import random
import re

INSTRUCTION = ("Think carefully through every constraint. End your response "
               "with a single line of the form 'Answer: <number>'.")


def fill_drain_staged(rng):
    """Start, drain-only, then fill+drain, then a removal. Capacity unused."""
    start = rng.choice([180, 220, 260, 300])
    cap = start + rng.choice([80, 120, 160])          # red herring
    d1 = rng.choice([8, 10, 12])
    t1 = rng.choice([3, 4, 5])
    fill = rng.choice([6, 7, 9])
    d2 = rng.choice([4, 5])
    t2 = rng.choice([2, 3, 4])
    remove = rng.choice([15, 20, 25])
    level = start - d1 * t1 + (fill - d2) * t2 - remove
    assert 0 < level < cap
    return (f"A tank holds at most {cap} liters and currently contains "
            f"{start} liters. For the first {t1} hours it is drained at "
            f"{d1} L/h with no inflow. Then for {t2} hours a pump adds "
            f"{fill} L/h while draining continues at {d2} L/h. Finally "
            f"{remove} liters are drawn off for a sample. How many liters "
            f"remain?", float(level))


def discount_coupon_tax_split(rng):
    """Percent off, then flat coupon, then tax, then split n ways + extra tip."""
    price = rng.choice([180.0, 240.0, 320.0, 400.0])
    disc = rng.choice([10, 15, 20])
    coupon = rng.choice([8.0, 12.0, 15.0])
    tax = rng.choice([6, 8, 10])
    n = rng.choice([3, 4, 5])
    tip = rng.choice([4.0, 6.0, 9.0])                 # paid by one person only
    after_disc = price * (1 - disc / 100)
    after_coupon = after_disc - coupon
    with_tax = after_coupon * (1 + tax / 100)
    per = round(with_tax / n, 2)
    # question: what does EACH of the other (n-1) people pay? (not the tipper)
    return (f"A bill is ${price:.0f} before any reductions. The restaurant "
            f"takes {disc}% off, then applies a ${coupon:.0f} coupon to the "
            f"already-discounted amount, then adds {tax}% tax. The {n} diners "
            f"split the taxed total evenly, except one diner also leaves a "
            f"${tip:.0f} cash tip that is NOT split. How many dollars does "
            f"each of the other diners pay?", per)


def two_leg_with_wait(rng):
    """Average speed over elapsed time including a wait (common trap)."""
    v1, v2 = rng.choice([(48, 72), (36, 54), (40, 60), (45, 75)])
    t1 = rng.choice([2, 3])                            # hours
    t2 = rng.choice([1, 2])
    d1, d2 = v1 * t1, v2 * t2
    wait_min = rng.choice([20, 30, 40, 45])
    elapsed = t1 + wait_min / 60 + t2
    avg = round((d1 + d2) / elapsed, 2)
    rest_stop = rng.choice(["Oakville", "Redfield", "Millbrook"])  # flavor
    return (f"A courier drives {d1} km at {v1} km/h, waits {wait_min} minutes "
            f"at a depot in {rest_stop}, then drives {d2} km at {v2} km/h. "
            f"The depot is {rng.choice([12, 18, 24])} km off the highway "
            f"(already included in the distances above). What is the average "
            f"speed for the whole elapsed time, in km/h?", avg)


def leftover_work(rng):
    """Together for t hours, then one leaves; remaining hours for the other."""
    a, b = rng.choice([(8, 12), (6, 10), (9, 18), (10, 15)])
    # together rate; pick t so some work remains
    together = 1 / a + 1 / b
    t = rng.choice([2, 3])
    remaining = 1 - together * t
    assert 0.05 < remaining < 0.7
    extra = round(remaining * a, 2)                    # Anna finishes alone
    return (f"Anna paints a room alone in {a} hours; Ben paints the same "
            f"room alone in {b} hours. They work together for {t} hours, "
            f"then Ben leaves. The room is {rng.choice([14, 16, 18])} feet "
            f"on a side (not needed for the calculation). How many more "
            f"hours does Anna need to finish the room by herself?", extra)


def fence_with_waste(rng):
    """Perimeter minus gate, 10% waste on fencing, plus posts every p m."""
    L = rng.choice([24, 28, 32])
    W = rng.choice([10, 12, 14])
    gate = rng.choice([2, 3])
    p = rng.choice([4, 5])                             # post spacing
    c_fence = rng.choice([7, 8, 9])
    c_post = rng.choice([11, 13, 15])
    fence_m = 2 * (L + W) - gate
    fence_buy = math.ceil(fence_m * 1.1)               # 10% waste, whole meters
    # posts around the fenced length, including both ends of the gate gap
    n_posts = fence_m // p + 1
    total = fence_buy * c_fence + n_posts * c_post
    return (f"A {L} m by {W} m rectangular yard is fenced except for a "
            f"{gate} m gate. Fencing is sold only in whole meters and "
            f"{10}% extra must be bought for waste. Posts cost ${c_post} "
            f"each and sit every {p} m along the fenced length, including "
            f"a post at each end of the gate opening "
            f"({fence_m}//{p} + 1 posts). Fencing costs ${c_fence} per "
            f"meter bought. What is the total cost in dollars?",
            float(total))


def savings_skip_weeks(rng):
    """Weekly deposit, skip every k-th week, then one withdrawal."""
    start = rng.choice([180, 220, 260])
    dep = rng.choice([18, 22, 25])
    weeks = rng.choice([10, 12, 14])
    k = rng.choice([3, 4])                             # skip every k-th
    spend = rng.choice([40, 55, 70])
    n_skip = weeks // k
    n_dep = weeks - n_skip
    total = start + dep * n_dep - spend
    return (f"Jordan starts with ${start}. For {weeks} consecutive weeks "
            f"they deposit ${dep} each week EXCEPT they skip every {k}th "
            f"week (weeks {k}, {2*k}, ...). After week {weeks} they spend "
            f"${spend}. Their rent is ${rng.choice([900, 1100, 1250])}/month "
            f"(already paid, ignore it). How many dollars remain?",
            float(total))


def recipe_guest_leftover(rng):  # holdout
    """Scale a recipe, buy bags, then an extra guest eats one more serving."""
    serves = rng.choice([4, 5])
    cups_per = rng.choice([2, 3])                      # cups per full recipe
    people = serves * rng.choice([2, 3])
    bag = rng.choice([8, 10])
    extra_guests = rng.choice([1, 2])
    need = cups_per * (people / serves)
    bags = math.ceil(need / bag)
    bought = bags * bag
    # leftover after cooking for `people`, then extra guests eat
    # (cups_per/serves) each from leftover
    leftover_after_cook = bought - need
    leftover_final = leftover_after_cook - extra_guests * (cups_per / serves)
    leftover_final = round(leftover_final, 2)
    assert leftover_final >= 0
    return (f"A recipe for {serves} people uses {cups_per} cups of flour. "
            f"Sam plans for {people} people and buys flour in {bag}-cup "
            f"bags, taking the fewest bags that cover the scaled recipe. "
            f"After cooking, {extra_guests} extra guest(s) arrive and each "
            f"eats one serving's worth of flour from what was left uncooked. "
            f"How many cups of flour remain unused?", leftover_final)


def battery_idle(rng):  # holdout
    """Use, idle drain, charge, then a hard cap at 100 is avoided by design."""
    start = rng.choice([82, 88, 94])
    use_r = rng.choice([11, 13, 14])
    use_h = rng.choice([2, 3])
    idle_r = rng.choice([2, 3])
    idle_h = rng.choice([4, 5])
    chg_r = rng.choice([16, 18, 20])
    chg_h = rng.choice([2, 3])
    level = start - use_r * use_h - idle_r * idle_h + chg_r * chg_h
    assert 5 < level < 100
    return (f"A laptop starts at {start}%. Video playback drains {use_r}% "
            f"per hour for {use_h} hours, then it sits idle draining "
            f"{idle_r}% per hour for {idle_h} hours, then it charges at "
            f"{chg_r}% per hour for {chg_h} hours. The charger is rated "
            f"{rng.choice([45, 65, 90])} W (irrelevant). What is the final "
            f"battery percentage?", float(level))


def tile_waste_remainder(rng):  # holdout
    """Area + waste percent, pack coverage, leftover unused area in last pack."""
    cover = rng.choice([4, 5, 6])
    waste_pct = rng.choice([8, 10, 12])
    rooms = rng.choice([2, 3])
    room_a = rng.choice([18, 20, 24])
    area = rooms * room_a
    need = area * (1 + waste_pct / 100)
    packs = math.ceil(need / cover)
    leftover = round(packs * cover - need, 2)          # unused coverage bought
    price = rng.choice([13, 17, 19])                   # red herring
    return (f"{rooms} identical rooms of {room_a} m² each are tiled. An "
            f"extra {waste_pct}% of the total area is bought for cuts. "
            f"Each pack covers exactly {cover} m² and costs ${price} "
            f"(price not needed). Buying the fewest whole packs that cover "
            f"the wasted-inclusive area, how many square meters of coverage "
            f"are bought but unused?", leftover)


TRAIN_FAMILIES = [
    ("fill_drain_staged", fill_drain_staged),
    ("discount_coupon_tax_split", discount_coupon_tax_split),
    ("two_leg_with_wait", two_leg_with_wait),
    ("leftover_work", leftover_work),
    ("fence_with_waste", fence_with_waste),
    ("savings_skip_weeks", savings_skip_weeks),
]
HOLDOUT_FAMILIES = [
    ("recipe_guest_leftover", recipe_guest_leftover),
    ("battery_idle", battery_idle),
    ("tile_waste_remainder", tile_waste_remainder),
]
INSTANCES_PER_FAMILY = 4


def build_problems():
    """Deterministic problem set: 24 train, 12 holdout (disjoint families)."""
    train, holdout = [], []
    for split, families in (("train", TRAIN_FAMILIES),
                            ("holdout", HOLDOUT_FAMILIES)):
        for fname, fn in families:
            rng = random.Random(f"ace-{split}-{fname}-v2")
            for inst in range(INSTANCES_PER_FAMILY):
                body, gold = fn(rng)
                item = {"id": f"{fname}-{inst}", "family": fname,
                        "split": split, "gold": gold,
                        "prompt": f"{body}\n\n{INSTRUCTION}"}
                (train if split == "train" else holdout).append(item)
    return train, holdout


TRAIN, HOLDOUT = build_problems()

ANSWER_RE = re.compile(r"answer\s*:\s*([^\n]*)", re.IGNORECASE)
NUM_RE = re.compile(r"(-)?\s*\$?([\d,]+(?:\.\d+)?)")


def extract_answer(text: str) -> float | None:
    """Last 'Answer:' line wins; parse the first number in it."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    num = NUM_RE.search(matches[-1])
    if not num:
        return None
    sign = -1.0 if num.group(1) else 1.0
    return sign * float(num.group(2).replace(",", ""))


def verify(text: str, gold: float, tol: float = 0.011) -> bool:
    pred = extract_answer(text)
    if pred is None:
        return False
    return abs(pred - gold) <= tol + 1e-9 * abs(gold)


if __name__ == "__main__":
    for split, items in (("TRAIN", TRAIN), ("HOLDOUT", HOLDOUT)):
        print(f"== {split}: {len(items)} problems ==")
        for it in items:
            print(f"[{it['id']:>28}] gold={it['gold']:>10}  "
                  f"{it['prompt'].splitlines()[0][:80]}")
