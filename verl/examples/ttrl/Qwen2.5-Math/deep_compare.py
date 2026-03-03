import json, math

NEW = "eval_results_aime_full_entropy/aime_eval_full_entropy_20260212_033954.json"
OLD = "eval_results_aime_full_entropy/aime_eval_kl_concentration_full.json"

with open(NEW) as f:
    new = json.load(f)
with open(OLD) as f:
    old = json.load(f)

eps = 1e-10

print("=" * 60)
print("Data Overview")
print("=" * 60)
for d, label in [(new, "NEW(standardized)"), (old, "OLD(raw)")]:
    nc = sum(1 for p in d['results'] for r in p['responses'] if r.get('is_correct'))
    nt = sum(len(p['responses']) for p in d['results'])
    mixed = sum(1 for p in d['results']
                if any(r.get('is_correct') for r in p['responses'])
                and not all(r.get('is_correct') for r in p['responses']))
    print(f"  {label}: {nt} resp, {nc} correct ({nc/nt*100:.1f}%), mixed={mixed}")

print()
print("=" * 60)
print("Sigma distribution")
print("=" * 60)

def get_all_sigmas(data_dict):
    sigs = []
    for p in data_dict['results']:
        for r in p['responses']:
            st = r['entropy_analysis'].get('step_transitions', [])
            for t in st:
                ks = t['kl_forward'] + t['kl_reverse']
                s = 2.0*t['js_divergence']/ks if ks>eps else 1.0
                sigs.append(max(0.0, min(1.0, s)))
    return sigs

for d, label in [(old, "OLD"), (new, "NEW")]:
    s = get_all_sigmas(d)
    m = sum(s)/len(s)
    std = (sum((x-m)**2 for x in s)/len(s))**0.5
    ss = sorted(s)
    n = len(ss)
    print(f"  {label}: mean={m:.4f} std={std:.4f} P5={ss[int(n*0.05)]:.4f} P50={ss[n//2]:.4f} P95={ss[int(n*0.95)]:.4f}")

print()
print("=" * 60)
print("Correct vs Wrong: mean(sigma)")
print("=" * 60)
for d, label in [(old, "OLD"), (new, "NEW")]:
    cs, ws = [], []
    for p in d['results']:
        for r in p['responses']:
            st = r['entropy_analysis'].get('step_transitions', [])
            if len(st) < 2: continue
            sigs = []
            for t in st:
                ks = t['kl_forward'] + t['kl_reverse']
                s = 2.0*t['js_divergence']/ks if ks>eps else 1.0
                sigs.append(max(0.0, min(1.0, s)))
            ms = sum(sigs)/len(sigs)
            if r.get('is_correct'): cs.append(ms)
            else: ws.append(ms)
    if cs and ws:
        mc, mw = sum(cs)/len(cs), sum(ws)/len(ws)
        sc = (sum((x-mc)**2 for x in cs)/len(cs))**0.5
        sw = (sum((x-mw)**2 for x in ws)/len(ws))**0.5
        sp = ((sc**2+sw**2)/2)**0.5
        cd = (mc-mw)/sp if sp>0 else 0
        print(f"  {label}: correct={mc:.6f}(n={len(cs)}) wrong={mw:.6f}(n={len(ws)}) gap={mc-mw:.6f} d={cd:+.4f}")
