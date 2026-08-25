"""Fixed strict extractors for b1 calibration — loosen extraction, not semantics.

Each fixes label/serialization tolerance while keeping exact semantic
validation against the reference. Validated with round-trip (reference must
pass) + per-field negative controls (mutated reference must reject).
Prototyped here, then ported into gen/*.py at source.
"""
from __future__ import annotations

import re

from gen.common import answer_text, norm_ws, parse_money

# ---------------------------------------------------------------- machine
_STATES = ("init", "ready", "active", "paused", "fault", "done")
_STATE_ALT = "|".join(_STATES)


def machine_check(completion: str, reference: str, knobs=None) -> bool:
    g = _machine_fields(answer_text(completion))
    w = _machine_fields(answer_text(reference))
    return all(v is not None for v in g) and g == w


def _machine_fields(ans: str):
    t = norm_ws(ans)
    # final state: labeled or bare state word
    m = re.search(r"(?:final|state|configuration)\s*[=:]?\s*(%s)" % _STATE_ALT, t)
    state = m.group(1) if m else None
    if state is None:
        m = re.search(r"\b(%s)\b" % _STATE_ALT, t)
        state = m.group(1) if m else None
    a = _grab(r"\ba\b", r"(true|false)", t)
    T = _grab(r"\bt\b", r"(true|false)", t)
    n = _grab(r"\bn\b", r"(-?\d+)", t)
    # positional booleans: "ready, false, false, 0, ..." or "fault, a=false, ..."
    if state is not None and a is None and T is None:
        m = re.search(re.escape(state) +
                      r"\s*[,;]\s*(true|false)\s*[,;]\s*(true|false)\s*[,;]\s*(-?\d+)", t)
        if m:
            a, T, n = m.group(1), m.group(2), m.group(3)
    # rejected count
    rej = None
    m = re.search(r"\brejected\b(?:\s*events)?\s*[=:]\s*(\d+)", t)
    if m:
        rej = m.group(1)
    else:
        m = re.search(r"\b(\d+)\s+(?:events\s+)?rejected\b", t)
        if m:
            rej = m.group(1)
    # first rejected index: tolerate index/idx/position labels and concatenations
    first = None
    for pat in (r"first[_ ]?re(?:jected|jection|jects)?[_ ]?(?:index|idx|position|event)?\s*[=:]\s*(\d+)",
                r"first[_ ]?(?:index|idx|position)\s*[=:]\s*(\d+)",
                r"\b(?:index|idx)\s*[=:]\s*(\d+)"):
        m = re.search(pat, t)
        if m:
            first = m.group(1)
            break
    # positional fallback: integers following the n value -> (rejected, first)
    if rej is None or first is None:
        if n is not None:
            mn = re.search(r"\bn\b\s*[=:]\s*-?\d+", t)
            start = mn.end() if mn else 0
            tail_ints = re.findall(r"-?\d+", t[start:])
            # drop any integer that is part of a label word (none expected) — take last two
            if rej is None and first is None and len(tail_ints) >= 2:
                rej, first = tail_ints[-2], tail_ints[-1]
            elif rej is None and len(tail_ints) >= 1:
                rej = tail_ints[0]
            elif first is None and len(tail_ints) >= 1:
                first = tail_ints[-1]
    return (state, a, T, n, rej, first)


def _grab(label, val, t):
    m = re.search(label + r"\s*[=:]\s*" + val, t)
    return m.group(1) if m else None


# ---------------------------------------------------------------- adversary
def adversary_check(completion: str, reference: str, knobs=None) -> bool:
    g_seq, g_tr = _adversary_witness(answer_text(completion))
    w_seq, w_tr = _adversary_witness(answer_text(reference))
    if not g_seq or not w_seq:
        return False
    # shortest witness required: same length as reference
    if len(g_seq) != len(w_seq):
        return False
    # verify against rules when available (authoritative)
    if knobs and "rules" in knobs:
        rules = {(m, cmd): (nm, dn) for m, cmd, nm, dn in knobs["rules"]}
        m, c = 0, 0
        derived = []
        for cmd in g_seq:
            if (m, cmd) not in rules:
                return False
            m, dn = rules[(m, cmd)]
            c += dn
            derived.append(c)
        # witness must drive the counter negative; a stated trace must match
        if c >= 0:
            return False
        # full answer required (sequence AND credit trace, matching derived)
        return bool(g_tr) and g_tr == derived
    # no rules: exact match only
    return g_seq == w_seq and g_tr == w_tr


def _adversary_witness(ans: str):
    # PRESERVE case (rules use A/B/C). Strip label words first so a label
    # starting with a command letter ("CMDS", "Credits") cannot leak its C
    # into the sequence. Sequence = first maximal run of A/B/C tolerating
    # separators; trace = the integer list that follows it.
    t = re.sub(r"\s+", " ", ans.strip())
    t = re.sub(r"(?i)\b(cmds?|commands?|credits?|credit|sequence|seq|trace|values?|result|answer)\b",
               " ", t)
    m = re.search(r"[ABC](?:[\s,]*[ABC])*", t)
    if not m:
        return None, []
    seq = re.sub(r"[^ABC]", "", m.group(0))
    tr = [int(x) for x in re.findall(r"-?\d+", t[m.end():])]
    return (seq or None), tr


# ---------------------------------------------------------------- assign
def assign_check(completion: str, reference: str, knobs=None) -> bool:
    got = _assign_pairs(answer_text(completion))
    want = _assign_pairs(reference)
    return bool(want) and got == want


def _assign_pairs(s: str):
    # S# = or : Name, case-insensitive; normalize slot->upper, name->lower
    return {slot.upper(): name.lower()
            for slot, name in re.findall(r"(?i)\b(s\d+)\s*[:=]\s*([a-z]+)", s)}


# ---------------------------------------------------------------- hypothesis
def hypothesis_check(completion: str, reference: str, knobs=None) -> bool:
    got = _hypothesis_parse(answer_text(completion))
    want = _hypothesis_parse(reference)
    return got is not None and got == want


def _hypothesis_parse(s: str):
    m = re.search(
        r"h\s*(\d+).*?expected\s*[=:]\s*\$?(-?[\d,]+\.\d{2}).*?"
        r"over[_ /]?short\s*[=:]\s*\$?(-?[\d,]+\.\d{2})",
        s, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return (int(m.group(1)),
            parse_money(m.group(2).replace(",", "")),
            parse_money(m.group(3).replace(",", "")))
