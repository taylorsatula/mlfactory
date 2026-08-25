import json, collections, sys
sys.path.insert(0, 'scratch')
import fixed_verifiers as FV
from gen import grid, certify
from gen.common import answer_text

DET = {'machine': FV.machine_check, 'adversary': FV.adversary_check,
       'assign': FV.assign_check, 'hypothesis': FV.hypothesis_check,
       'grid': grid.check, 'certify': certify.check}

cands = {json.loads(l)['provenance']['proposal_id']: json.loads(l)
         for l in open('data/acegen_probe_b1.jsonl')}
rolls = [json.loads(l) for l in open('data/probe_rollouts_b1.jsonl')]
verdicts = [json.loads(l) for l in open('scratch/judge_b1_verdicts.jsonl')]
jmap = {(v['proposal_id'], v['sample_i']): v for v in verdicts}

# judge errors?
errs = [v for v in verdicts if v['judge'].startswith('ERR')]
print(f"judge rows: {len(verdicts)}  ERR rows: {len(errs)}")
for v in errs[:10]:
    print("   ERR", v['proposal_id'], v['sample_i'], v['family'], v['judge'])

# reconcile per family
fam_stats = collections.defaultdict(lambda: collections.Counter())
disagree = []
for r in rolls:
    pid = r['proposal_id']; c = cands[pid]; fam = c['domain']
    ref = c['problem']['reference_answer']; knobs = c.get('knobs', {})
    v = jmap[(pid, r['sample_i'])]
    jyes = v['judge'] == 'YES'
    det = bool(DET[fam](r['completion'], ref, knobs))
    union = jyes or det
    fam_stats[fam]['judge'] += jyes
    fam_stats[fam]['det'] += det
    fam_stats[fam]['union'] += union
    fam_stats[fam]['n'] += 1
    if jyes != det:
        disagree.append((pid, r['sample_i'], fam, v['judge'], det,
                         r.get('truncated'), v['cand_answer'][:120], ref[:60]))

print("\n=== per-family correct counts: judge / det / union ===")
print(f"{'fam':11} {'judge':>6} {'det':>6} {'union':>6} {'n':>4}")
for fam in ['machine', 'adversary', 'assign', 'certify', 'grid', 'hypothesis']:
    s = fam_stats[fam]
    print(f"{fam:11} {s['judge']:>3}/{s['n']:<3} {s['det']:>3}/{s['n']:<3} {s['union']:>3}/{s['n']:<3}")

print(f"\n=== judge vs det disagreements: {len(disagree)} ===")
for d in disagree:
    pid, si, fam, jv, det, trunc, cand, ref = d
    kind = 'judge-YES/det-NO' if jv == 'YES' else 'judge-NO/det-YES'
    print(f"  pid{pid} s{si} {fam:11} [{kind}] trunc={int(bool(trunc))}")
    print(f"      ref : {ref!r}")
    print(f"      cand: {cand!r}")
