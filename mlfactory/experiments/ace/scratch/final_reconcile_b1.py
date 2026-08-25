import json, collections, sys
from gen import machine, adversary, assign, hypothesis, grid, certify
from gen.common import answer_text
DET = {'machine': machine.check, 'adversary': adversary.check,
       'assign': assign.check, 'hypothesis': hypothesis.check,
       'grid': grid.check, 'certify': certify.check}

cands = {json.loads(l)['provenance']['proposal_id']: json.loads(l)
         for l in open('data/acegen_probe_b1.jsonl')}
rolls = [json.loads(l) for l in open('data/probe_rollouts_b1.jsonl')]
verdicts = {(v['proposal_id'], v['sample_i']): v
            for v in (json.loads(l) for l in open('scratch/judge_b1_verdicts.jsonl'))}
by = collections.defaultdict(list)
for r in rolls:
    by[r['proposal_id']].append(r)

# ---- final reconciliation: judge vs det(final, tightened) ----
print("=== judge vs det(final) agreement per family ===")
print(f"{'fam':11} {'agree':>6} {'jYes/dNo':>9} {'jNo/dYes':>9}  det-final")
for fam in ['machine', 'adversary', 'assign', 'certify', 'grid', 'hypothesis']:
    agree = jydn = jndy = dn = 0
    for r in rolls:
        c = cands[r['proposal_id']]
        if c['domain'] != fam:
            continue
        v = verdicts[(r['proposal_id'], r['sample_i'])]
        jy = v['judge'] == 'YES'
        d = bool(DET[fam](r['completion'], c['problem']['reference_answer'], c.get('knobs', {})))
        if jy == d:
            agree += 1
        elif jy and not d:
            jydn += 1
        else:
            jndy += 1
    pids = [p for p in by if cands[p]['domain'] == fam]
    dettot = sum(1 for p in pids for r in by[p]
                 if DET[fam](r['completion'], cands[p]['problem']['reference_answer'], cands[p].get('knobs', {})))
    print(f"{fam:11} {agree:>6} {jydn:>9} {jndy:>9}  {dettot}/64")

# ---- failure modes among WRONG (det-final) ----
def looped(comp):
    tail = comp[-3000:]
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    if len(lines) < 8:
        return False
    c = collections.Counter(lines)
    topn = c.most_common(1)[0][1]
    return topn >= 6 and len(c.most_common(1)[0][0]) < 80

print("\n=== failure modes among WRONG (det-final), per family ===")
print(f"{'fam':11} {'loop-noanswer':>14} {'trunc-other':>12} {'finished-wrong':>14} {'wrong-total':>11}")
for fam in ['machine', 'adversary', 'assign', 'certify', 'grid', 'hypothesis']:
    lc = to = fw = 0
    for r in rolls:
        c = cands[r['proposal_id']]
        if c['domain'] != fam:
            continue
        if DET[fam](r['completion'], c['problem']['reference_answer'], c.get('knobs', {})):
            continue
        tr = bool(r.get('truncated'))
        at = answer_text(r['completion'])
        has_answer = bool(__import__('re').search(r'(?i)answer\s*:', r['completion']))
        if tr and looped(r['completion']):
            lc += 1
        elif tr:
            to += 1
        else:
            fw += 1
    print(f"{fam:11} {lc:>14} {to:>12} {fw:>14} {lc+to+fw:>11}")

# truncated-but-correct (emitted answer then looped/truncated)
tc = sum(1 for r in rolls
         if r.get('truncated') and DET[cands[r['proposal_id']]['domain']](
             r['completion'], cands[r['proposal_id']]['problem']['reference_answer'],
             cands[r['proposal_id']].get('knobs', {})))
tottrunc = sum(1 for r in rolls if r.get('truncated'))
print(f"\ntruncated-but-CORRECT: {tc}/{tottrunc} truncated rows "
      f"(emitted answer then looped/capped)")

# ---- authoritative band table ----
print("\n=== authoritative per-prompt bands (det-final) ===")
for fam in ['adversary', 'machine', 'assign', 'certify', 'grid', 'hypothesis']:
    pids = sorted(p for p in by if cands[p]['domain'] == fam)
    bands = []
    for pid in pids:
        c = cands[pid]
        n = sum(1 for r in by[pid]
                if DET[fam](r['completion'], c['problem']['reference_answer'], c.get('knobs', {})))
        band = 'DEAD-HARD' if n == 0 else ('DEAD-EASY' if n == 8 else 'LIVE')
        bands.append(f"p{pid}:{n}/8{'' if band=='LIVE' else '('+band+')'}")
    print(f"  {fam:11} " + "  ".join(bands))
