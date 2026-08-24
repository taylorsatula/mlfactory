"""Graph coloring certification: produce a valid k-coloring or state NONE.

This is the TRANSFORMED replacement for the enumeration-shaped
graph_networks domain (which asked the model to count spanning trees —
computation, not reasoning). Certification keeps the graph structure and
the deceptive-greedy trap, but the deliverable is a single witness (or
impossibility verdict), which a reasoner can construct and check.

Trap knob: nodes are presented in an order where greedy coloring FAILS
(needs a k+1th color) even though a k-coloring exists — a
deceptive_local_optimum the solver must back out of.

Knobs:
    n_nodes    6-9     graph size
    k          3       colors
    none_prob  0.25    fraction of instances that are uncolorable (NONE)
"""
from __future__ import annotations

import random
import re
from itertools import combinations

from .common import Problem, answer_text

NODES = "ABCDEFGHIJ"
COLORS = ["red", "blue", "green", "yellow"]


def _k_color(n, edges, k):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    color = [None] * n
    order = sorted(range(n), key=lambda x: -len(adj[x]))

    def rec(idx):
        if idx == n:
            return True
        v = order[idx]
        for c in range(k):
            if all(color[u] != c for u in adj[v]):
                color[v] = c
                if rec(idx + 1):
                    return True
                color[v] = None
        return False

    return list(color) if rec(0) else None


def _greedy_fails(n, edges, k, order):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    color = {}
    for v in order:
        used = {color[u] for u in adj[v] if u in color}
        c = 0
        while c in used:
            c += 1
        if c >= k:
            return True
        color[v] = c
    return False


def make(rng: random.Random, knobs: dict) -> Problem:
    n, k = knobs["n_nodes"], knobs["k"]
    want_none = rng.random() < knobs.get("none_prob", 0.25)

    for _attempt in range(400):
        edges = set()
        if want_none:
            # plant K(k+1) so it is definitely not k-colorable
            clique = rng.sample(range(n), k + 1)
            edges.update(tuple(sorted(e)) for e in combinations(clique, 2))
        for u, v in combinations(range(n), 2):
            if rng.random() < 0.35:
                edges.add((u, v))
        edges = sorted(edges)
        if len(edges) < n:          # too sparse is trivial
            continue
        sol = _k_color(n, edges, k)
        if want_none and sol is not None:
            continue
        if not want_none:
            if sol is None:
                continue
            # find a presentation order where greedy fails (trap)
            trap_order = None
            if knobs.get("trap", True):
                for _ in range(60):
                    cand = list(range(n))
                    rng.shuffle(cand)
                    if _greedy_fails(n, edges, k, cand):
                        trap_order = cand
                        break
            if knobs.get("trap", True) and trap_order is None:
                continue

        listed = trap_order if (not want_none and knobs.get("trap", True)
                                and trap_order) else list(range(n))
        edge_str = ", ".join(f"{NODES[u]}-{NODES[v]}" for u, v in edges)
        prose = (
            f"A deployment graph has {n} services, listed in audit order: "
            + ", ".join(NODES[i] for i in listed) + ".\n\n"
            f"Conflict edges (services that must NOT share a channel): "
            f"{edge_str}\n\n"
            f"There are {k} channels available: {', '.join(COLORS[:k])}."
            + ("" if want_none or trap_order is None else
               f"\n\nNote: assigning channels greedily in the listed audit "
               f"order forces a {k+1}th channel, yet the auditor insists a "
               f"{k}-channel assignment exists.")
        )
        if sol is None:
            ans = "NONE"
        else:
            ans = ", ".join(f"{NODES[i]}={COLORS[sol[i]]}" for i in range(n))
        return Problem(
            family="certify", prose=prose,
            question=f"Give a valid {k}-channel assignment as Node=channel "
                     f"pairs covering every service, or state NONE if no "
                     f"valid assignment exists.",
            answer=ans,
            verifier_kind="constraint_checker",
            objective_task="constraint_satisfaction",
            search_topology=("deceptive_local_optimum" if sol is not None
                             else "representation_change"),
            knobs={"n_nodes": n, "k": k, "trap": bool(knobs.get("trap")),
                   "edges": [[u, v] for u, v in edges],
                   "solvable": sol is not None},
            seed=rng.randrange(2**31))
    raise RuntimeError("certify: no instance found")


# Colors are matched case-insensitively (models often write Title Case);
# node letters stay uppercase-only, colors are lowercased after capture.
_PAIR_RE = re.compile(r"([A-Z])\s*=\s*([A-Za-z]+)")


def check(completion: str, reference: str, knobs: dict | None = None) -> bool:
    text = answer_text(completion)
    if reference == "NONE":
        return "none" in text.lower() and not _PAIR_RE.search(text)
    if not knobs or "edges" not in knobs:
        return False
    got = {u: c.lower() for u, c in _PAIR_RE.findall(text)}
    n = knobs["n_nodes"]
    if len(got) != n or set(got) != set(NODES[:n]):
        return False
    palette = set(COLORS[: knobs["k"]])
    if any(c not in palette for c in got.values()):
        return False
    return all(got[NODES[u]] != got[NODES[v]] for u, v in knobs["edges"])
