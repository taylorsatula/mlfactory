import json, collections, sys
sys.path.insert(0, 'scratch')
import fixed_verifiers as FV
from gen.common import answer_text

cands = {json.loads(l)['provenance']['proposal_id']: json.loads(l)
         for l in open('data/acegen_probe_b1.jsonl')}
rolls = [json.loads(l) for l in open('data/probe_rollouts_b1.jsonl')]
by_pid = collections.defaultdict(list)
for r in rolls:
    by_pid[r['proposal_id']].append(r)

# adversary after trace-extraction fix + stated-trace verification
apids = sorted(p for p in by_pid if cands[p]['domain'] == 'adversary')
tot = 0; cross = 0; crossex = []
print("=== adversary per-prompt (trace-checked) ===")
for pid in apids:
    c = cands[pid]; ref = c['problem']['reference_answer']; knobs = c.get('knobs', {})
    n = 0
    for r in sorted(by_pid[pid], key=lambda x: x['sample_i']):
        seq, tr = FV._adversary_witness(answer_text(r['completion']))
        ok = FV.adversary_check(r['completion'], ref, knobs)
        n += ok
        if ok and tr:
            rules = {(m, cmd): (nm, dn) for m, cmd, nm, dn in knobs['rules']}
            m, c2 = 0, 0; der = []
            for cmd in seq:
                m, dn = rules[(m, cmd)]; c2 += dn; der.append(c2)
            if tr != der:
                print(f"  !! pid{pid} s{r['sample_i']} accepted but stated {tr} != derived {der}")
    tot += n; print(f"  pid {pid}: {n}/8")
    for r in by_pid[pid]:
        if FV.adversary_check(r['completion'], ref, knobs):
            for opid in apids:
                if opid == pid:
                    continue
                oc = cands[opid]
                if FV.adversary_check(r['completion'], oc['problem']['reference_answer'], oc.get('knobs', {})):
                    cross += 1; crossex.append((pid, r['sample_i'], opid))
print(f"adversary total {tot}/64 | cross-accepts now {cross} {crossex}")

# machine per-field mutation negative control on FINAL extractor
print("\n=== machine per-field mutation negative control (final extractor) ===")
ALT = {'init': 'done', 'ready': 'active', 'active': 'ready', 'paused': 'fault', 'fault': 'ready', 'done': 'init'}
mpids = sorted(p for p in by_pid if cands[p]['domain'] == 'machine')
fp = 0; tt = 0
for pid in mpids:
    c = cands[pid]; ref = c['problem']['reference_answer']; knobs = c.get('knobs', {})
    w = FV._machine_fields(answer_text(ref))
    corr = [r for r in by_pid[pid] if FV.machine_check(r['completion'], ref, knobs)]
    if not corr:
        continue
    comp = corr[0]['completion']
    st, a, T, n, rej, first = w
    muts = {'state': ALT[st], 'a': ('true' if a == 'false' else 'false'),
            't': ('true' if T == 'false' else 'false'), 'n': str(int(n) + 1),
            'rejected': str(int(rej) + 1), 'first': str(int(first) + 1)}
    for field, nv in muts.items():
        mw = {'state': st, 'a': a, 't': T, 'n': n, 'rejected': rej, 'first': first}
        mw[field] = nv
        mref = (f"final={mw['state']} A={mw['a']} T={mw['t']} n={mw['n']} "
                f"rejected={mw['rejected']} first_rejected={mw['first']}")
        tt += 1
        if FV.machine_check(comp, mref, knobs):
            fp += 1; print(f"  !! FALSE POS pid{pid} field {field}")
print(f"  machine per-field negative controls: {tt - fp}/{tt} rejected")
