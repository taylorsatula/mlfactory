#!/usr/bin/env python3
"""Generate the baseline prompt corpus for autoregressive working-state quality study.

This script is deterministic: every prompt ID, text, and answer key is produced from
fixed seeds and versioned generator code.  It does not touch any model or API.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import string
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from mlfactory.core.metrics import MetricsLogger
from mlfactory.core.prompts import render_markdown

VERSION = "2026-07-27-v1"
DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "generation_system_prompt.md"


def system_prompt(extra_instructions: str = "") -> str:
    return render_markdown(DEFAULT_PROMPT_PATH, extra_instructions=extra_instructions)


def uid(prefix: str, idx: int) -> str:
    return f"{prefix}{idx:04d}"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def difficulty_rationale(difficulty: str, category: str, subcategory: str) -> str:
    rationales = {
        "easy": "Solvable in one to three explicit reasoning steps; the procedure is directly inferable from the prompt.",
        "medium": "Requires connecting multiple facts or a short bounded search; the path is structured but not immediate.",
        "hard": "Requires sustained reasoning, non-trivial search, or several interacting constraints near the model's capability boundary.",
    }
    return rationales.get(difficulty, "")


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 1. Multistep mathematics
# ---------------------------------------------------------------------------
def gen_math_arithmetic(rng: random.Random) -> dict:
    """Nested integer expression with mixed operations and parentheses."""
    ops = ["+", "-", "*", "//"]
    depth = rng.randint(3, 5)

    def expr(d: int) -> tuple[str, int]:
        if d == 0:
            n = rng.randint(2, 99)
            return str(n), n
        if rng.random() < 0.35:
            n = rng.randint(2, 99)
            return str(n), n
        left, lv = expr(d - 1)
        right, rv = expr(d - 1)
        op = rng.choice(ops)
        if op == "//":
            if rv == 0:
                rv = 1
            if lv * rv < 0 and lv % rv != 0:
                val = lv // rv + 1  # Python floor division correction
            else:
                val = lv // rv
        elif op == "+":
            val = lv + rv
        elif op == "-":
            val = lv - rv
        else:
            val = lv * rv
        return f"({left} {op} {right})", val

    text, answer = expr(depth)
    prompt = (
        "Compute the exact value of the following integer expression. "
        "Use integer division (//) where indicated; division rounds toward negative infinity.\n\n"
        f"Expression: {text}\n\n"
        "Give the final integer."
    )
    return {
        "category": "multistep_math",
        "subcategory": "arithmetic",
        "difficulty": "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(answer)},
        "metadata": {"expression": text},
    }


def gen_math_diophantine(rng: random.Random) -> dict:
    """Count positive integer solutions to a*x + b*y = c."""
    a = rng.randint(2, 12)
    b = rng.randint(2, 12)
    # choose a target with a small but nonzero number of positive solutions
    target = rng.randint(a + b + 1, 10 * (a + b))
    count = 0
    sols = []
    x = 1
    while a * x < target:
        rem = target - a * x
        if rem > 0 and rem % b == 0:
            y = rem // b
            sols.append((x, y))
            count += 1
        x += 1
    prompt = (
        "How many pairs of positive integers (x, y) satisfy the equation\n\n"
        f"    {a}*x + {b}*y = {target}\n\n"
        "Explain your enumeration strategy and give the final count."
    )
    return {
        "category": "multistep_math",
        "subcategory": "diophantine",
        "difficulty": "hard" if count > 5 else "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(count)},
        "metadata": {"a": a, "b": b, "target": target, "solutions": sols},
    }


def gen_math_modular(rng: random.Random) -> dict:
    """Find smallest positive n such that a^n ≡ b (mod m)."""
    m = rng.choice([23, 29, 31, 37, 41, 43, 47])
    a = rng.randint(2, m - 1)
    # ensure a is coprime to m
    while math.gcd(a, m) != 1:
        a = rng.randint(2, m - 1)
    # pick a target reachable by powers of a
    order = 1
    cur = a % m
    while cur != 1:
        cur = (cur * a) % m
        order += 1
    powers = [(a ** i) % m for i in range(1, order + 1)]
    b = rng.choice(powers)
    n = powers.index(b) + 1
    prompt = (
        "Find the smallest positive integer n such that\n\n"
        f"    {a}^n ≡ {b} (mod {m})\n\n"
        "State n."
    )
    return {
        "category": "multistep_math",
        "subcategory": "modular",
        "difficulty": "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(n)},
        "metadata": {"a": a, "b": b, "m": m, "order": order},
    }


def gen_math_polynomial_root(rng: random.Random) -> dict:
    """Given a small cubic with rational root, ask for the sum of roots."""
    roots = [rng.randint(-6, 6) for _ in range(3)]
    while len(set(roots)) < 3:
        roots = [rng.randint(-6, 6) for _ in range(3)]
    x = sp.Symbol("x")
    poly = sp.expand((x - roots[0]) * (x - roots[1]) * (x - roots[2]))
    expr = sp.expand(poly).as_ordered_terms()
    poly_str = str(poly).replace("**", "^").replace("*", "")
    # manual pretty formatting
    terms = []
    for term in sp.Poly(poly, x).all_terms():
        coeff = int(term[1])
        power = term[0][0]
        if coeff == 0:
            continue
        if power == 0:
            terms.append(f"{coeff:+d}")
        elif power == 1:
            terms.append(f"{coeff:+d}x")
        else:
            terms.append(f"{coeff:+d}x^{power}")
    poly_pretty = "".join(terms).lstrip("+")
    prompt = (
        "Consider the cubic polynomial\n\n"
        f"    P(x) = {poly_pretty}\n\n"
        "Find the sum of its three (possibly repeated) roots."
    )
    return {
        "category": "multistep_math",
        "subcategory": "polynomial",
        "difficulty": "easy",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(sum(roots))},
        "metadata": {"roots": roots, "poly": poly_pretty},
    }


def gen_math_sequence(rng: random.Random) -> dict:
    """Give terms of a linear recurrence and ask for next term."""
    kind = rng.choice(["linear", "quadratic", "fibonacci-like"])
    if kind == "linear":
        a, d = rng.randint(1, 20), rng.randint(2, 15)
        terms = [a + i * d for i in range(6)]
        next_term = a + 6 * d
    elif kind == "quadratic":
        a, b, c = rng.randint(1, 5), rng.randint(-5, 5), rng.randint(0, 10)
        terms = [a * i * i + b * i + c for i in range(6)]
        next_term = a * 36 + b * 6 + c
    else:
        p, q = rng.randint(1, 9), rng.randint(1, 9)
        terms = [p, q]
        for _ in range(4):
            terms.append(terms[-1] + terms[-2])
        next_term = terms[-1] + terms[-2]
    prompt = (
        "The following sequence follows a simple deterministic rule. "
        "Determine the next term and explain the rule.\n\n"
        f"Sequence: {', '.join(map(str, terms))}, ?"
    )
    return {
        "category": "multistep_math",
        "subcategory": "sequence",
        "difficulty": "easy",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(next_term)},
        "metadata": {"terms": terms, "rule": kind},
    }


def gen_math_inequality_count(rng: random.Random) -> dict:
    """Count integer solutions to a system of two inequalities."""
    a, b = rng.randint(2, 8), rng.randint(2, 8)
    c, d = rng.randint(10, 40), rng.randint(10, 40)
    count = 0
    sols = []
    for x in range(-20, 41):
        for y in range(-20, 41):
            if a * x + b * y <= c and x - y <= d and x >= 0 and y >= 0:
                count += 1
                sols.append((x, y))
    prompt = (
        "How many ordered pairs of non-negative integers (x, y) satisfy both inequalities?\n\n"
        f"    {a}*x + {b}*y ≤ {c}\n"
        f"    x - y ≤ {d}\n\n"
        "Give the count."
    )
    return {
        "category": "multistep_math",
        "subcategory": "integer_inequalities",
        "difficulty": "hard" if count > 15 else "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(count)},
        "metadata": {"a": a, "b": b, "c": c, "d": d, "solutions": sols},
    }


MATH_GENERATORS = [
    gen_math_arithmetic,
    gen_math_diophantine,
    gen_math_modular,
    gen_math_polynomial_root,
    gen_math_sequence,
    gen_math_inequality_count,
]


# ---------------------------------------------------------------------------
# 2. Formal logic and constraint satisfaction
# ---------------------------------------------------------------------------
def gen_logic_knights_knaves(rng: random.Random) -> dict:
    """Small Knights and Knaves puzzle with N people and one statement each."""
    names = [rng.choice(["A", "B", "C", "D", "E"]) for _ in range(rng.randint(3, 5))]
    names = list(dict.fromkeys(names))
    if len(names) < 3:
        names = ["A", "B", "C"]
    n = len(names)
    # generate a random consistent assignment
    assignment = [rng.choice([True, False]) for _ in range(n)]
    # build statements of the form "X is a knight" or "Y is a knave"
    statements = []
    for i, name in enumerate(names):
        target = rng.randrange(n)
        claimed = rng.choice([True, False])
        # Statement is true iff assignment[target] == claimed
        # Speaker tells truth iff assignment[i] is True.
        # So we need assignment[i] == (assignment[target] == claimed)
        # If not, flip claimed to make it consistent.
        if (assignment[target] == claimed) != assignment[i]:
            claimed = not claimed
        statements.append((name, names[target], "knight" if claimed else "knave"))
    stmt_texts = [f"{s[0]} says: \"{s[1]} is a {s[2]}.\"" for s in statements]
    answer = ", ".join(
        f"{names[i]}={'knight' if assignment[i] else 'knave'}" for i in range(n)
    )
    prompt = (
        "On an island, every person is either a knight (always tells the truth) "
        "or a knave (always lies). You hear the following statements:\n\n"
        + "\n".join(stmt_texts)
        + "\n\nDetermine the type of each person. "
        "Give your answer as a comma-separated list, e.g., A=knight, B=knave."
    )
    return {
        "category": "formal_logic",
        "subcategory": "knights_knaves",
        "difficulty": "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": answer},
        "metadata": {"names": names, "assignment": assignment, "statements": statements},
    }


def gen_logic_sat_count(rng: random.Random) -> dict:
    """Count satisfying assignments of a small CNF formula."""
    n_vars = rng.randint(3, 5)
    n_clauses = rng.randint(4, 7)
    clauses = []
    for _ in range(n_clauses):
        clause = []
        for _ in range(rng.randint(2, 3)):
            v = rng.randint(1, n_vars)
            neg = rng.choice([True, False])
            clause.append((v, neg))
        clauses.append(clause)
    count = 0
    models = []
    for bits in itertools.product([False, True], repeat=n_vars):
        ok = True
        for clause in clauses:
            if not any((bits[v - 1] if not neg else not bits[v - 1]) for v, neg in clause):
                ok = False
                break
        if ok:
            count += 1
            models.append(bits)
    # pretty clauses
    def lit(v, neg):
        return f"¬x{v}" if neg else f"x{v}"

    clause_strs = ["(" + " ∨ ".join(lit(v, neg) for v, neg in c) + ")" for c in clauses]
    formula = " ∧ ".join(clause_strs)
    prompt = (
        "A Boolean formula in conjunctive normal form is given by\n\n"
        f"    {formula}\n\n"
        "How many assignments of the variables make the formula true?"
    )
    return {
        "category": "formal_logic",
        "subcategory": "sat_count",
        "difficulty": "hard" if n_vars >= 5 else "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(count)},
        "metadata": {"n_vars": n_vars, "clauses": clauses, "models": models},
    }


def gen_logic_syllogism(rng: random.Random) -> dict:
    """Simple categorical syllogism with quantifiers."""
    categories = rng.sample(["mammals", "birds", "reptiles", "fish", "insects", "plants", "fungi", "bacteria"], 3)
    a, b, c = categories
    # All A are B; some B are C; conclusion: some A are C?  Not valid in general, but here decide via premise choice.
    valid = rng.choice([True, False])
    if valid:
        prompt = (
            f"Premise 1: All {a} are {b}.\n"
            f"Premise 2: Some {b} are {c}.\n\n"
            "Assuming the premises are true, is the conclusion \"Some {a} are {c}\" necessarily true? "
            "Answer exactly Yes or No, then briefly justify."
        )
        answer = "No"  # actually invalid; this is a deliberate false-lead structure
    else:
        prompt = (
            f"Premise 1: All {a} are {b}.\n"
            f"Premise 2: All {b} are {c}.\n\n"
            "Assuming the premises are true, is the conclusion \"All {a} are {c}\" necessarily true? "
            "Answer exactly Yes or No, then briefly justify."
        )
        answer = "Yes"
    prompt = prompt.format(a=a, b=b, c=c)
    return {
        "category": "formal_logic",
        "subcategory": "syllogism",
        "difficulty": "easy",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": answer},
        "metadata": {"valid": valid, "categories": categories},
    }


LOGIC_GENERATORS = [gen_logic_knights_knaves, gen_logic_sat_count, gen_logic_syllogism]


# ---------------------------------------------------------------------------
# 3. Combinatorial search and planning
# ---------------------------------------------------------------------------
def gen_search_grid_paths(rng: random.Random) -> dict:
    """Count monotone lattice paths in a small grid with blocked cells."""
    w, h = rng.randint(4, 6), rng.randint(4, 6)
    blocked = set()
    num_blocks = rng.randint(2, 5)
    while len(blocked) < num_blocks:
        x = rng.randint(1, w - 1)
        y = rng.randint(1, h - 1)
        blocked.add((x, y))
    # dynamic programming count
    dp = [[0] * (h + 1) for _ in range(w + 1)]
    dp[0][0] = 1
    for x in range(w + 1):
        for y in range(h + 1):
            if (x, y) in blocked:
                dp[x][y] = 0
                continue
            if x > 0:
                dp[x][y] += dp[x - 1][y]
            if y > 0:
                dp[x][y] += dp[x][y - 1]
    answer = dp[w][h]
    block_lines = [f"({x},{y})" for x, y in sorted(blocked)]
    prompt = (
        "You start at (0,0) and want to reach "
        f"({w},{h}) moving only right (+1,0) or up (0,+1). "
        "The following grid cells are blocked and cannot be entered:\n\n"
        + ", ".join(block_lines)
        + "\n\nHow many valid monotone paths exist?"
    )
    return {
        "category": "combinatorial_search",
        "subcategory": "grid_paths",
        "difficulty": "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(answer)},
        "metadata": {"w": w, "h": h, "blocked": sorted(blocked)},
    }


def gen_search_subset_sum(rng: random.Random) -> dict:
    """Ask whether a subset of numbers sums to a target."""
    n = rng.randint(6, 10)
    nums = sorted(rng.sample(range(1, 50), n))
    # choose a target that is achievable
    subset_mask = rng.randint(1, (1 << n) - 1)
    target = sum(nums[i] for i in range(n) if (subset_mask >> i) & 1)
    # sometimes perturb to make impossible
    if rng.random() < 0.3:
        target += rng.choice([1, 3, 7])
        # verify no subset sums to target
        ok = True
        for mask in range(1 << n):
            s = sum(nums[i] for i in range(n) if (mask >> i) & 1)
            if s == target:
                ok = False
                break
        if not ok:
            target += 1
    answer = "Yes" if any(
        sum(nums[i] for i in range(n) if (mask >> i) & 1) == target
        for mask in range(1 << n)
    ) else "No"
    prompt = (
        "Given the set of integers\n\n"
        f"    {nums}\n\n"
        f"is there a subset whose elements sum to exactly {target}? "
        "Answer Yes or No and briefly justify."
    )
    return {
        "category": "combinatorial_search",
        "subcategory": "subset_sum",
        "difficulty": "hard" if n >= 8 else "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": answer},
        "metadata": {"nums": nums, "target": target},
    }


def gen_search_permutation_inversion(rng: random.Random) -> dict:
    """Given permutation, count inversions."""
    n = rng.randint(6, 10)
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    prompt = (
        "Count the number of inversions in the permutation\n\n"
        f"    {perm}\n\n"
        "An inversion is a pair (i, j) with i < j and perm[i] > perm[j]."
    )
    return {
        "category": "combinatorial_search",
        "subcategory": "inversions",
        "difficulty": "easy",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": str(inv)},
        "metadata": {"permutation": perm},
    }


SEARCH_GENERATORS = [gen_search_grid_paths, gen_search_subset_sum, gen_search_permutation_inversion]


# ---------------------------------------------------------------------------
# 4. Algorithm design and complexity reasoning
# ---------------------------------------------------------------------------
def gen_algorithm_design(rng: random.Random) -> dict:
    """Ask for an algorithmic approach; verifier is a human-checked rubric."""
    problems = [
        {
            "name": "range_mode",
            "text": (
                "You are given an array A of n integers and q queries. "
                "Each query gives two indices l and r and asks for the most frequent value "
                "in A[l..r] (the mode). Design an algorithm that answers all queries efficiently. "
                "State the preprocessing time, query time, and space complexity."
            ),
            "rubric": "Expected: O(n log n) preprocessing or sqrt decomposition; query time sublinear or O(log n); space O(n log n).",
        },
        {
            "name": "dynamic_median",
            "text": (
                "Design a data structure that supports inserting integers one at a time and, after each insertion, "
                "reports the median of all elements seen so far. State the per-operation time complexity and explain why."
            ),
            "rubric": "Expected: two heaps (max-heap for lower half, min-heap for upper half), O(log n) per insertion, O(1) median.",
        },
        {
            "name": "k_largest_stream",
            "text": (
                "Integers arrive in a stream. At any time, a query may ask for the k largest elements seen so far. "
                "Design a streaming algorithm and state the time to process each new element and the time to answer a query."
            ),
            "rubric": "Expected: min-heap of size k; O(log k) per element; query O(k) or O(k log k).",
        },
        {
            "name": "interval_coverage",
            "text": (
                "Given a set of intervals on the line, find the size of the smallest subset whose union equals the union of all intervals. "
                "Describe an efficient algorithm and prove its correctness idea."
            ),
            "rubric": "Expected: greedy by earliest finishing point after sorting by start; O(n log n).",
        },
    ]
    p = rng.choice(problems)
    return {
        "category": "algorithm_design",
        "subcategory": p["name"],
        "difficulty": "hard",
        "prompt_text": p["text"],
        "verifier": {"type": "rubric", "answer": p["rubric"]},
        "metadata": {"problem": p["name"]},
    }


ALGO_GENERATORS = [gen_algorithm_design]


# ---------------------------------------------------------------------------
# 5. Code generation
# ---------------------------------------------------------------------------
def gen_code_function(rng: random.Random) -> dict:
    """Ask to implement a Python function with provided tests."""
    tasks = [
        {
            "name": "rotate_matrix",
            "spec": (
                "Write a Python function `rotate_90_clockwise(matrix)` that rotates a square 2-D list "
                "90 degrees clockwise in-place and returns the modified matrix."
            ),
            "tests": [
                "assert rotate_90_clockwise([[1,2],[3,4]]) == [[3,1],[4,2]]",
                "assert rotate_90_clockwise([[1]]) == [[1]]",
            ],
            "difficulty": "medium",
        },
        {
            "name": "merge_sorted_lists",
            "spec": (
                "Write a Python function `merge_sorted(a, b)` that merges two sorted lists of integers into a single sorted list. "
                "Do not use built-in sort."
            ),
            "tests": [
                "assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]",
                "assert merge_sorted([], [1,2]) == [1,2]",
            ],
            "difficulty": "easy",
        },
        {
            "name": "first_missing_positive",
            "spec": (
                "Write a Python function `first_missing_positive(nums)` that returns the smallest positive integer "
                "greater than 0 that does not appear in the list `nums`. Optimize for O(n) time and O(1) extra space."
            ),
            "tests": [
                "assert first_missing_positive([3,4,-1,1]) == 2",
                "assert first_missing_positive([1,2,0]) == 3",
                "assert first_missing_positive([7,8,9,11,12]) == 1",
            ],
            "difficulty": "hard",
        },
        {
            "name": "longest_balanced_parentheses",
            "spec": (
                "Write a Python function `longest_balanced(s)` that returns the length of the longest contiguous "
                "substring of parentheses that is balanced, where '(' has value +1 and ')' has value -1."
            ),
            "tests": [
                "assert longest_balanced('(()') == 2",
                "assert longest_balanced(')()())') == 4",
                "assert longest_balanced('') == 0",
            ],
            "difficulty": "hard",
        },
    ]
    t = rng.choice(tasks)
    test_block = "\n".join(t["tests"])
    prompt = (
        f"{t['spec']}\n\n"
        f"Your function will be checked against these tests:\n\n```python\n{test_block}\n```\n\n"
        "Provide only the function definition."
    )
    return {
        "category": "code_generation",
        "subcategory": t["name"],
        "difficulty": t["difficulty"],
        "prompt_text": prompt,
        "verifier": {"type": "python_tests", "tests": t["tests"]},
        "metadata": {"task": t["name"]},
    }


CODE_GENERATORS = [gen_code_function]


# ---------------------------------------------------------------------------
# 6. Code debugging and test diagnosis
# ---------------------------------------------------------------------------
def gen_code_debug(rng: random.Random) -> dict:
    """Present a buggy snippet and ask for the fix."""
    snippets = [
        {
            "name": "binary_search_off_by_one",
            "buggy": (
                "def binary_search(arr, x):\n"
                "    lo, hi = 0, len(arr)\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] < x:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid\n"
                "    return arr[lo] == x\n"
            ),
            "issue": "Returns IndexError when x is greater than all elements because lo can equal len(arr).",
            "fix": "Return lo < len(arr) and arr[lo] == x.",
        },
        {
            "name": "fib_memo_missing_return",
            "buggy": (
                "def fib(n, memo={}):\n"
                "    if n in memo:\n"
                "        memo[n]\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n"
            ),
            "issue": "The memoized branch does not return the cached value, and the final recursive result is not returned.",
            "fix": "Use 'return memo[n]' in the cached branch and 'return memo[n]' after computing it.",
        },
    ]
    s = rng.choice(snippets)
    prompt = (
        "The following Python function is intended to work correctly but contains one or more bugs. "
        "Identify the bug(s) and provide a corrected implementation.\n\n"
        f"```python\n{s['buggy']}```\n\n"
        "Explain the bug and give the fixed code."
    )
    return {
        "category": "code_debugging",
        "subcategory": s["name"],
        "difficulty": "medium",
        "prompt_text": prompt,
        "verifier": {"type": "rubric", "answer": s["fix"]},
        "metadata": {"bug": s["issue"]},
    }


DEBUG_GENERATORS = [gen_code_debug]


# ---------------------------------------------------------------------------
# 7. Causal and scientific reasoning
# ---------------------------------------------------------------------------
def gen_causal_scientific(rng: random.Random) -> dict:
    """Simple causal reasoning from supplied facts."""
    scenarios = [
        {
            "name": "circuit_series",
            "text": (
                "A 12 V battery is connected in series with a 4 Ω resistor and an 8 Ω resistor. "
                "What is the current flowing through the circuit? Give your answer in amperes."
            ),
            "answer": "1.0",
        },
        {
            "name": "ecosystem_trophic",
            "text": (
                "In a simple ecosystem, grass is eaten by rabbits, and rabbits are eaten by foxes. "
                "If a disease suddenly removes most of the fox population, what is the most likely short-term effect on the rabbit population? "
                "Choose one: increase, decrease, or stay the same, and explain the causal chain."
            ),
            "answer": "increase",
        },
        {
            "name": "gas_law",
            "text": (
                "A sealed container holds a fixed amount of ideal gas at 300 K and 1 atm. "
                "The temperature is raised to 600 K while the volume remains constant. "
                "What is the new pressure? Give your answer in atm."
            ),
            "answer": "2",
        },
    ]
    s = rng.choice(scenarios)
    return {
        "category": "causal_scientific",
        "subcategory": s["name"],
        "difficulty": "easy",
        "prompt_text": s["text"],
        "verifier": {"type": "exact", "answer": s["answer"]},
        "metadata": {"scenario": s["name"]},
    }


CAUSAL_GENERATORS = [gen_causal_scientific]


# ---------------------------------------------------------------------------
# 8. Stateful simulations and rule-based systems
# ---------------------------------------------------------------------------
def gen_simulation_rule(rng: random.Random) -> dict:
    """One-dimensional cellular automaton (rule 110/30/90) after n steps."""
    rule = rng.choice([30, 90, 110, 184])
    steps = rng.randint(5, 12)
    width = rng.randint(12, 20)
    init = [0] * width
    mid = width // 2
    init[mid] = 1

    def next_state(state):
        new = [0] * len(state)
        for i in range(len(state)):
            l = state[(i - 1) % len(state)]
            c = state[i]
            r = state[(i + 1) % len(state)]
            pattern = (l << 2) | (c << 1) | r
            new[i] = 1 if (rule >> pattern) & 1 else 0
        return new

    state = init[:]
    for _ in range(steps):
        state = next_state(state)
    final = "".join("#" if x else "." for x in state)
    prompt = (
        "You are simulating a one-dimensional cellular automaton on a cyclic tape of width "
        f"{width}. The rule is elementary rule {rule}: the next state of a cell is determined by "
        "the 3-bit pattern formed by its left neighbor, itself, and its right neighbor. "
        "Initially every cell is 0 except the center cell, which is 1. "
        f"Run the automaton for exactly {steps} steps.\n\n"
        "Represent the final tape as a string of '#' (cell value 1) and '.' (cell value 0). "
        "Give only the final string."
    )
    return {
        "category": "stateful_simulation",
        "subcategory": "elementary_ca",
        "difficulty": "medium" if steps > 8 else "easy",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": final},
        "metadata": {"rule": rule, "steps": steps, "width": width, "final": final},
    }


def gen_simulation_register(rng: random.Random) -> dict:
    """Tiny register-machine program; ask for final register values."""
    regs = {"A": 0, "B": 0, "C": 0}
    program = []
    n = rng.randint(4, 8)
    reg_names = list(regs.keys())
    for _ in range(n):
        op = rng.choice(["INC", "DEC", "MOV", "ADD"])
        r1 = rng.choice(reg_names)
        if op == "INC":
            program.append(("INC", r1))
            regs[r1] += 1
        elif op == "DEC":
            program.append(("DEC", r1))
            regs[r1] -= 1
        elif op == "MOV":
            r2 = rng.choice(reg_names)
            program.append(("MOV", r1, r2))
            regs[r2] = regs[r1]
        else:
            r2 = rng.choice(reg_names)
            program.append(("ADD", r1, r2))
            regs[r2] = regs[r2] + regs[r1]
    lines = []
    for instr in program:
        if len(instr) == 2:
            lines.append(f"{instr[0]} {instr[1]}")
        else:
            lines.append(f"{instr[0]} {instr[1]} {instr[2]}")
    final = ", ".join(f"{k}={v}" for k, v in regs.items())
    prompt = (
        "A simple register machine has three registers A, B, C, all initially 0. "
        "The supported instructions are:\n"
        "  INC R   (R := R + 1)\n"
        "  DEC R   (R := R - 1)\n"
        "  MOV R S (S := R)\n"
        "  ADD R S (S := S + R)\n\n"
        "Execute the following program and report the final values of A, B, and C.\n\n"
        + "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
        + "\n\nGive the final state as A=..., B=..., C=..."
    )
    return {
        "category": "stateful_simulation",
        "subcategory": "register_machine",
        "difficulty": "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": final},
        "metadata": {"program": program, "final": regs},
    }


SIM_GENERATORS = [gen_simulation_rule, gen_simulation_register]


# ---------------------------------------------------------------------------
# 9. Adversarially underspecified problems
# ---------------------------------------------------------------------------
def gen_underspecified(rng: random.Random) -> dict:
    """Missing information; correct answer requires stating assumptions."""
    items = [
        {
            "name": "average_speed",
            "text": (
                "A cyclist rides from town A to town B at 12 km/h and returns at 18 km/h. "
                "What is the cyclist's average speed for the entire round trip?"
            ),
            "answer": "14.4",
            "note": "Assumes equal distances each way.",
        },
        {
            "name": "dilution_volume",
            "text": (
                "You have a 10% salt solution and want to make 500 mL of a 4% salt solution. "
                "How much water should you add?"
            ),
            "answer": "300 mL of the 10% solution plus 200 mL water (or equivalent; needs assumption about final volume)",
            "note": "Ambiguous wording; needs assumption.",
        },
    ]
    item = rng.choice(items)
    return {
        "category": "underspecified",
        "subcategory": item["name"],
        "difficulty": "medium",
        "prompt_text": item["text"],
        "verifier": {"type": "rubric", "answer": item["answer"], "note": item["note"]},
        "metadata": {"name": item["name"]},
    }


UNDER_GENERATORS = [gen_underspecified]


# ---------------------------------------------------------------------------
# 10. False leads and interacting constraints
# ---------------------------------------------------------------------------
def gen_false_leads(rng: random.Random) -> dict:
    """Puzzle with extra irrelevant quantities."""
    # Relative speed problem with red herring.
    train_length = rng.randint(100, 200)
    platform_length = rng.randint(200, 400)
    speed = rng.randint(20, 60)
    time = (train_length + platform_length) / speed
    answer = f"{time:.2f}"
    prompt = (
        "A train of length "
        f"{train_length} m is traveling at a constant speed of {speed} m/s. "
        "It completely passes a platform of length "
        f"{platform_length} m. "
        "A passenger inside the train notes that the journey from his home station to the next major city "
        f"covers {rng.randint(50,300)} km and usually takes {rng.randint(1,4)} hours. "
        "How many seconds does it take for the entire train to pass the platform? "
        "Ignore the irrelevant information."
    )
    return {
        "category": "false_leads_constraints",
        "subcategory": "red_herring",
        "difficulty": "easy",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": answer},
        "metadata": {"train": train_length, "platform": platform_length, "speed": speed},
    }


def gen_interacting_constraints(rng: random.Random) -> dict:
    """Small arithmetic cryptarithm (no leading zeros)."""
    # Use a simple 2-letter mapping a*b = cd? Let's use SEND+MORE=MONEY style too complex.
    # Instead: distinct digits A,B,C with A+B=C and constraints.
    letters = rng.sample(["A", "B", "C", "D"], 4)
    # build constraints: A+B=C, A-B=D (so C = A+B, D = A-B), all digits distinct, A> B
    solutions = []
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            c = a + b
            d = a - b
            if 0 <= c <= 9 and 0 <= d <= 9 and len({a, b, c, d}) == 4:
                solutions.append({letters[0]: a, letters[1]: b, letters[2]: c, letters[3]: d})
    sol = rng.choice(solutions)
    prompt = (
        "Solve the digit puzzle. Each letter stands for a distinct digit 0-9.\n\n"
        f"    {letters[0]} + {letters[1]} = {letters[2]}\n"
        f"    {letters[0]} - {letters[1]} = {letters[3]}\n\n"
        "Give the value of each letter, e.g., A=1, B=2, ..."
    )
    answer = ", ".join(f"{k}={v}" for k, v in sol.items())
    return {
        "category": "false_leads_constraints",
        "subcategory": "digit_puzzle",
        "difficulty": "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": answer},
        "metadata": {"solution": sol},
    }


FALSE_GENERATORS = [gen_false_leads, gen_interacting_constraints]


# ---------------------------------------------------------------------------
# 11. Long-input tasks
# ---------------------------------------------------------------------------
def gen_long_input(rng: random.Random) -> dict:
    """Multiple statements; answer requires combining evidence."""
    n = rng.randint(8, 15)
    people = [f"P{i}" for i in range(1, n + 1)]
    # Random friendship graph
    friends = {p: set() for p in people}
    for i, a in enumerate(people):
        for b in people[i + 1 :]:
            if rng.random() < 0.3:
                friends[a].add(b)
                friends[b].add(a)
    # choose a target person and build statements
    target = rng.choice(people)
    statements = []
    clues = []
    for p in people:
        if p == target:
            continue
        if target in friends[p]:
            clue = f"{p} says: \"{target} is my friend.\""
        else:
            clue = f"{p} says: \"I do not know {target}.\""
        clues.append(clue)
    rng.shuffle(clues)
    # ask for number of friends of target
    answer = str(len(friends[target]))
    prompt = (
        "You are given a list of statements about friendships in a group. "
        "Determine how many friends the person "
        f"{target} has.\n\n"
        + "\n".join(clues)
        + "\n\nGive the integer count."
    )
    return {
        "category": "long_input",
        "subcategory": "friendship_count",
        "difficulty": "medium",
        "prompt_text": prompt,
        "verifier": {"type": "exact", "answer": answer},
        "metadata": {"target": target, "friends": {k: sorted(v) for k, v in friends.items()}},
    }


LONG_GENERATORS = [gen_long_input]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
CATEGORY_GENERATORS = {
    "multistep_math": (MATH_GENERATORS, 70),
    "formal_logic": (LOGIC_GENERATORS, 40),
    "combinatorial_search": (SEARCH_GENERATORS, 40),
    "algorithm_design": (ALGO_GENERATORS, 25),
    "code_generation": (CODE_GENERATORS, 55),
    "code_debugging": (DEBUG_GENERATORS, 25),
    "causal_scientific": (CAUSAL_GENERATORS, 25),
    "stateful_simulation": (SIM_GENERATORS, 35),
    "underspecified": (UNDER_GENERATORS, 25),
    "false_leads_constraints": (FALSE_GENERATORS, 30),
    "long_input": (LONG_GENERATORS, 30),
}


def generate_corpus(total_target: int = 400, extra_instructions: str = "") -> list[dict]:
    records: list[dict] = []
    base_seed = 20260727
    prompt_text = system_prompt(extra_instructions=extra_instructions)
    prompt_hash = sha256(prompt_text)
    # First pass: generate per-category quotas, then pad if needed.
    for cat, (gens, quota) in CATEGORY_GENERATORS.items():
        rng = random.Random(base_seed + hash(cat) % 2**31)
        for i in range(quota):
            item = rng.choice(gens)(rng)
            item["prompt_id"] = uid("p", len(records))
            item["category"] = cat
            item["construction_method"] = "programmatic"
            item["generator_version"] = VERSION
            item["provenance"] = "original_generator"
            item["system_prompt"] = prompt_text
            item["sampling_profile"] = "coding" if cat in ("code_generation", "code_debugging") else "general"
            item["difficulty_rationale"] = difficulty_rationale(item["difficulty"], cat, item.get("subcategory", ""))
            item["prompt_hash"] = sha256(item["prompt_text"])
            item["system_prompt_hash"] = prompt_hash
            # verifier reference depends on type; will point to the verifier artifact
            if item["verifier"]["type"] == "python_tests":
                item["verifier_reference"] = f"verifiers/{item['prompt_id']}_tests.py"
            else:
                item["verifier_reference"] = "verifiers.jsonl"
            records.append(item)
    # deterministic shuffle
    shuffle_rng = random.Random(base_seed)
    shuffle_rng.shuffle(records)
    for i, r in enumerate(records):
        r["prompt_id"] = uid("p", i)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", default=None, help="legacy alias; ignored when --run-dir is given")
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--extra-instructions", default="", help="Injectable content inserted into the system prompt")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsLogger(run_dir, echo=True)

    records = generate_corpus(args.target, extra_instructions=args.extra_instructions)
    prompts_path = out_dir / "prompts.jsonl"
    write_jsonl_atomic(prompts_path, records)

    # Save a verifier map for easy lookup
    verifiers = {r["prompt_id"]: r["verifier"] for r in records}
    (out_dir / "verifiers.jsonl").write_text(
        "\n".join(json.dumps({"prompt_id": k, **v}, ensure_ascii=False) for k, v in verifiers.items()) + "\n",
        encoding="utf-8",
    )

    # Save verifier code separately for python test tasks
    verifier_dir = out_dir / "verifiers"
    verifier_dir.mkdir(exist_ok=True)
    for r in records:
        if r["verifier"]["type"] == "python_tests":
            (verifier_dir / f"{r['prompt_id']}_tests.py").write_text(
                f"# tests for {r['prompt_id']}\n" + "\n".join(r["verifier"]["tests"]) + "\n",
                encoding="utf-8",
            )

    # manifest summary
    counts = Counter(r["category"] for r in records)
    print(f"Wrote {len(records)} prompts to {prompts_path}")
    print("Category counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
