"""Robust b1 re-score: qwen38 as answer-equivalence judge vs solver-built reference.

Ground truth is exact by construction (make()'s internal solver). The judge
only decides whether the completion EXPRESSES that answer — a normalization
task, not a truth task. Deterministic fixed check() is a cross-check; the
union covers multi-valid-answer families (certify alternate colorings,
adversary alternative witnesses) where single-reference comparison undercounts.

Judge: temp 0, reasoning disabled, YES/NO. Verdicts saved immutably.
Writes: scratch/judge_b1_verdicts.jsonl (one row per completion).
"""
import json, collections, sys, time, urllib.request
sys.path.insert(0, 'scratch')
import fixed_verifiers as FV
from gen.common import answer_text
from gen import grid, certify

DET = {'machine': FV.machine_check, 'adversary': FV.adversary_check,
       'assign': FV.assign_check, 'hypothesis': FV.hypothesis_check,
       'grid': grid.check, 'certify': certify.check}

URL = "http://127.0.0.1:3090/v1/chat/completions"

PROMPT = """You are a strict answer-equivalence checker.

The problem asked for a specific final answer. You are given the REFERENCE (correct) answer and a CANDIDATE's final answer extracted from its response.

REFERENCE (correct) answer:
{ref}

CANDIDATE's final answer:
{cand}

Decide whether the candidate states the SAME final result as the reference.
Rules:
- Ignore formatting, capitalization, whitespace, separators (= vs : vs , vs ;), field-label wording, and field ordering.
- Compare the underlying VALUES only. Every value in the reference must be present in the candidate with the same value.
- If any value differs, or the candidate is missing part of the answer, or is garbled / truncated / merely restating the problem, answer NO.
- If the reference is "NONE", the candidate must state that no valid solution exists.
- Do NOT re-solve the problem. Only compare the candidate's stated final answer to the reference.

Answer with exactly one word: YES or NO."""


def judge(ref, cand):
    body = {"model": "Qwen3.8 27B MTP",
            "messages": [{"role": "user", "content": PROMPT.format(ref=ref, cand=cand)}],
            "temperature": 0.0, "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            txt = out["choices"][0]["message"].get("content", "").strip().upper()
            tok = out.get("usage", {}).get("completion_tokens")
            word = "".join(ch for ch in txt if ch.isalpha())
            return ("YES" if word.startswith("YES") else
                    "NO" if word.startswith("NO") else "ERR:" + txt[:30], tok)
        except Exception as e:
            if attempt == 2:
                return "ERR:" + type(e).__name__, None
            time.sleep(2)


def main():
    cands = {json.loads(l)['provenance']['proposal_id']: json.loads(l)
             for l in open('data/acegen_probe_b1.jsonl')}
    rolls = [json.loads(l) for l in open('data/probe_rollouts_b1.jsonl')]
    out = open('scratch/judge_b1_verdicts.jsonl', 'w')
    n = 0; t0 = time.time()
    for r in rolls:
        pid = r['proposal_id']; c = cands[pid]; fam = c['domain']
        ref = c['problem']['reference_answer']; knobs = c.get('knobs', {})
        cand = answer_text(r['completion'])
        # cap to last 2000 chars: a real answer line is always short; a huge
        # tail only occurs for truncated loops (no 'Answer:') whose tail is
        # pure repetition -> correctly judged NO. Avoids 32k-token judge prompts.
        cand_j = cand[-2000:]
        jv, tok = judge(ref, cand_j)
        dv = bool(DET[fam](r['completion'], ref, knobs))
        row = {"proposal_id": pid, "sample_i": r['sample_i'], "family": fam,
               "judge": jv, "det": dv, "truncated": bool(r.get('truncated')),
               "cand_answer": cand[:300]}
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
        n += 1
        if n % 24 == 0:
            print(f"  {n}/{len(rolls)}  elapsed {time.time()-t0:.0f}s", flush=True)
    out.close()
    print(f"DONE {n} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
