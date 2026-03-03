"""
Exhaustive search for unsupervised reward signals derived from RevKL(boxed||window).
Key insight: correct responses have RevKL ~1.9 vs incorrect ~5.9 (3x gap, most stable signal).
"""

import json, re, math, numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ============================================================
# Load & extract
# ============================================================
base_dir = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy"
files = [f"{base_dir}/aime_eval_full_entropy_20260207_092147.json",
         f"{base_dir}/aime_eval_full_entropy_20260207_090427.json"]

all_resp = []
for fp in files:
    with open(fp) as f: data = json.load(f)
    tag = fp.split('_')[-1].replace('.json','')
    for res in data['results']:
        for r in res['responses']:
            toks, ents = [], []
            for s in r['entropy_analysis']['steps']:
                toks.extend(s['tokens']); ents.extend(s['token_entropies'])
            if toks:
                all_resp.append({'tag':tag,'pid':res['id'],'correct':r['is_correct'],
                                 'tokens':toks,'ent':np.array(ents,dtype=np.float32)})

def boxed_idx(tokens):
    txt=""; pos=[]
    for i,t in enumerate(tokens): s=len(txt); txt+=t; pos.append((s,len(txt)))
    idx=set()
    for m in re.finditer(r'\\boxed\{',txt):
        d,p=1,m.end()
        while p<len(txt) and d>0:
            if txt[p]=='{':d+=1
            elif txt[p]=='}':d-=1
            p+=1
        for i,(ts,te) in enumerate(pos):
            if ts<p and te>m.start(): idx.add(i)
    return sorted(idx)

def kl_js(p,q,nb=50,eps=1e-10):
    if len(p)<3 or len(q)<3: return np.nan,np.nan,np.nan
    a=np.concatenate([p,q]); lo,hi=a.min(),a.max()
    if hi-lo<1e-6: return 0.,0.,0.
    bins=np.linspace(lo-.01,hi+.01,nb+1)
    ph,_=np.histogram(p,bins=bins,density=True); qh,_=np.histogram(q,bins=bins,density=True)
    ph=(ph+eps); ph/=ph.sum(); qh=(qh+eps); qh/=qh.sum()
    mh=.5*(ph+qh)
    fkl=float(np.sum(ph*np.log(ph/qh)))
    rkl=float(np.sum(qh*np.log(qh/ph)))
    js=float(.5*np.sum(ph*np.log(ph/mh))+.5*np.sum(qh*np.log(qh/mh)))
    return fkl,rkl,js

# ============================================================
# Per-response feature extraction
# ============================================================
WINDOW = 15

records = []
for resp in all_resp:
    ent = resp['ent']; n = len(ent)
    bi = boxed_idx(resp['tokens'])
    has_boxed = len(bi) > 0

    if not has_boxed or n < WINDOW + len(bi):
        records.append({**{k:resp[k] for k in ['tag','pid','correct']},
                        'has_boxed':False, **{f:np.nan for f in
                        ['fkl','rkl','js','ratio','asym','rkl_early','rkl_late',
                         'rkl_slope','rkl_cv','rkl_min','rkl_max','rkl_final_drop',
                         'neg_rkl','inv_rkl','log_rkl','rkl_stability',
                         'boxed_h','reason_h','all_h','boxed_zero','n_tokens',
                         'fkl_late','rkl_median','rkl_iqr']}})
        continue

    boxed_ent = ent[bi]; bs = set(bi)
    nb_idx = [i for i in range(n) if i not in bs]
    nb_ent = ent[nb_idx]

    # Full response KL
    fkl, rkl, js = kl_js(nb_ent, boxed_ent)

    # Sliding window RevKL trajectory
    rkl_traj, fkl_traj = [], []
    step = max(1, (n - WINDOW) // 30)
    for start in range(0, n - WINDOW + 1, step):
        w = ent[start:start+WINDOW]
        f_, r_, _ = kl_js(w, boxed_ent, nb=30)
        if not np.isnan(r_):
            rkl_traj.append(r_)
            fkl_traj.append(f_)

    if len(rkl_traj) < 4:
        rkl_traj = [rkl] * 4
        fkl_traj = [fkl] * 4

    rkl_arr = np.array(rkl_traj)
    fkl_arr = np.array(fkl_traj)
    q1 = len(rkl_arr) // 4

    rkl_early = float(np.mean(rkl_arr[:q1])) if q1 > 0 else rkl_arr[0]
    rkl_late = float(np.mean(rkl_arr[-q1:])) if q1 > 0 else rkl_arr[-1]
    fkl_late = float(np.mean(fkl_arr[-q1:])) if q1 > 0 else fkl_arr[-1]

    # RevKL trajectory statistics
    rkl_slope = (rkl_late - rkl_early) / (rkl_early + 0.01)
    rkl_cv = float(np.std(rkl_arr) / (np.mean(rkl_arr) + 0.01))
    rkl_min = float(np.min(rkl_arr))
    rkl_max = float(np.max(rkl_arr))
    rkl_final_drop = (rkl_early - rkl_late) / (rkl_early + 0.01)
    rkl_median = float(np.median(rkl_arr))
    rkl_iqr = float(np.percentile(rkl_arr, 75) - np.percentile(rkl_arr, 25))
    rkl_stability = 1.0 / (1.0 + rkl_cv)

    records.append({
        'tag': resp['tag'], 'pid': resp['pid'], 'correct': resp['correct'],
        'has_boxed': True, 'n_tokens': n,
        'fkl': fkl, 'rkl': rkl, 'js': js,
        'ratio': fkl / (rkl + 0.01),
        'asym': fkl - rkl,
        'rkl_early': rkl_early, 'rkl_late': rkl_late,
        'rkl_slope': rkl_slope, 'rkl_cv': rkl_cv,
        'rkl_min': rkl_min, 'rkl_max': rkl_max,
        'rkl_final_drop': rkl_final_drop,
        'rkl_median': rkl_median, 'rkl_iqr': rkl_iqr,
        'rkl_stability': rkl_stability,
        'neg_rkl': -rkl,
        'inv_rkl': 1.0 / (rkl + 0.1),
        'log_rkl': -math.log(rkl + 0.1),
        'fkl_late': fkl_late,
        'boxed_h': float(np.mean(boxed_ent)),
        'reason_h': float(np.mean(nb_ent)),
        'all_h': float(np.mean(ent)),
        'boxed_zero': float(np.mean(boxed_ent == 0)),
    })

wb = [r for r in records if r['has_boxed']]
print(f"With boxed: {len(wb)}, Correct: {sum(r['correct'] for r in wb)}, "
      f"Incorrect: {sum(not r['correct'] for r in wb)}")

# ============================================================
# Evaluate all candidates
# ============================================================
def evaluate(name, vals_c, vals_ic, data, val_fn, direction='higher'):
    c = np.array([v for v in vals_c if not np.isnan(v)])
    ic = np.array([v for v in vals_ic if not np.isnan(v)])
    if len(c) < 3 or len(ic) < 3:
        return None
    pooled = np.sqrt((c.var()*len(c)+ic.var()*len(ic))/(len(c)+len(ic)))
    d = (c.mean()-ic.mean())/(pooled+1e-10) if direction=='higher' else (ic.mean()-c.mean())/(pooled+1e-10)
    alt = 'greater' if direction=='higher' else 'less'
    u,p = stats.mannwhitneyu(c,ic,alternative=alt)
    auc = u/(len(c)*len(ic))

    pg = defaultdict(lambda:{'c':[],'ic':[]})
    for r in data:
        v = val_fn(r)
        if np.isnan(v): continue
        k = (r['tag'],r['pid'])
        if r['correct']: pg[k]['c'].append(v)
        else: pg[k]['ic'].append(v)
    conc,tot,pw,pt = 0,0,0,0
    for k,g in pg.items():
        if g['c'] and g['ic']:
            pt+=1
            cm,im = np.mean(g['c']),np.mean(g['ic'])
            if direction=='higher':
                if cm>im: pw+=1
            else:
                if cm<im: pw+=1
            for cv in g['c']:
                for iv in g['ic']:
                    tot+=1
                    if direction=='higher':
                        if cv>iv: conc+=1
                    else:
                        if cv<iv: conc+=1
    concordance = conc/tot if tot else 0
    wr = pw/pt if pt else 0
    return {'name':name,'d':d,'auc':auc,'conc':concordance,'wr':wr,
            'c_mean':c.mean(),'ic_mean':ic.mean(),'c_med':np.median(c),'ic_med':np.median(ic),
            'pw':pw,'pt':pt,'u_p':p,'n_c':len(c),'n_ic':len(ic)}

candidates = [
    # --- Pure RevKL transforms ---
    ("A1: -RevKL",                   'neg_rkl',        'higher'),
    ("A2: 1/(RevKL+0.1)",           'inv_rkl',        'higher'),
    ("A3: -log(RevKL+0.1)",         'log_rkl',        'higher'),
    ("A4: -RevKL_min (best window)", 'rkl_min',        'lower'),
    ("A5: -RevKL_median",           'rkl_median',      'lower'),
    ("A6: RevKL stability 1/(1+CV)", 'rkl_stability',  'higher'),
    ("A7: -RevKL_late (final quarter)", 'rkl_late',    'lower'),
    # --- RevKL trajectory shape ---
    ("B1: RevKL convergence (early-late)/early", 'rkl_final_drop', 'higher'),
    ("B2: -RevKL slope (late-early)/early", 'rkl_slope', 'lower'),
    ("B3: -RevKL IQR (consistency)",  'rkl_iqr',       'lower'),
    ("B4: -RevKL CV",                'rkl_cv',          'lower'),
    # --- RevKL combinations ---
    ("C1: FwdKL/RevKL (ratio)",      'ratio',          'higher'),
    ("C2: FwdKL-RevKL (asymmetry)",  'asym',           'higher'),
    # --- Baselines ---
    ("D1: -H_boxed",                 'boxed_h',        'lower'),
    ("D2: -H_all",                   'all_h',          'lower'),
    ("D3: boxed_zero_ratio",         'boxed_zero',     'higher'),
]

# Add computed composite candidates
composite_fns = [
    ("E1: -RevKL * stability",
     lambda r: -r['rkl'] * r['rkl_stability'] if not any(np.isnan(r[k]) for k in ['rkl','rkl_stability']) else np.nan,
     'higher'),
    ("E2: -log(RevKL) * (1-CV)",
     lambda r: -math.log(r['rkl']+0.1) * (1-r['rkl_cv']) if not any(np.isnan(r[k]) for k in ['rkl','rkl_cv']) else np.nan,
     'higher'),
    ("E3: 1/(RevKL+0.1) * convergence",
     lambda r: 1/(r['rkl']+0.1) * max(0, r['rkl_final_drop']) if not any(np.isnan(r[k]) for k in ['rkl','rkl_final_drop']) else np.nan,
     'higher'),
    ("E4: exp(-RevKL/5)",
     lambda r: math.exp(-r['rkl']/5) if not np.isnan(r['rkl']) else np.nan,
     'higher'),
    ("E5: exp(-RevKL/3)",
     lambda r: math.exp(-r['rkl']/3) if not np.isnan(r['rkl']) else np.nan,
     'higher'),
    ("E6: exp(-RevKL/10)",
     lambda r: math.exp(-r['rkl']/10) if not np.isnan(r['rkl']) else np.nan,
     'higher'),
    ("E7: sigmoid(3-RevKL)",
     lambda r: 1/(1+math.exp(-(3-r['rkl']))) if not np.isnan(r['rkl']) else np.nan,
     'higher'),
    ("E8: sigmoid(5-RevKL)",
     lambda r: 1/(1+math.exp(-(5-r['rkl']))) if not np.isnan(r['rkl']) else np.nan,
     'higher'),
    ("E9: max(0, 1-RevKL/10) (linear clip)",
     lambda r: max(0, 1-r['rkl']/10) if not np.isnan(r['rkl']) else np.nan,
     'higher'),
    ("E10: -RevKL_late / (RevKL_early+0.1)",
     lambda r: -r['rkl_late']/(r['rkl_early']+0.1) if not any(np.isnan(r[k]) for k in ['rkl_late','rkl_early']) else np.nan,
     'higher'),
    ("F1: FwdKL_late / (RevKL+0.1)",
     lambda r: r['fkl_late']/(r['rkl']+0.1) if not any(np.isnan(r[k]) for k in ['fkl_late','rkl']) else np.nan,
     'higher'),
    ("F2: -RevKL * (FwdKL/RevKL)^0.5",
     lambda r: -r['rkl'] * math.sqrt(r['fkl']/(r['rkl']+0.01)) if not any(np.isnan(r[k]) for k in ['rkl','fkl']) else np.nan,
     'higher'),
    ("F3: -sqrt(RevKL)",
     lambda r: -math.sqrt(r['rkl']) if not np.isnan(r['rkl']) else np.nan,
     'higher'),
    ("F4: -(RevKL - RevKL_min)",
     lambda r: -(r['rkl'] - r['rkl_min']) if not any(np.isnan(r[k]) for k in ['rkl','rkl_min']) else np.nan,
     'higher'),
]

print("\n" + "=" * 95)
print("COMPREHENSIVE RevKL-BASED REWARD CANDIDATE EVALUATION")
print("=" * 95)
print(f"{'Candidate':<40s} | {'d':>7s} {'AUC':>6s} {'Conc':>7s} {'WR':>6s} | {'C_mean':>9s} {'IC_mean':>9s}")
print("-" * 95)

all_results = []

for name, key, direction in candidates:
    vc = [r[key] for r in wb if r['correct'] and not np.isnan(r[key])]
    vic = [r[key] for r in wb if not r['correct'] and not np.isnan(r[key])]
    res = evaluate(name, vc, vic, wb, lambda r, k=key: r[k], direction)
    if res:
        all_results.append(res)
        print(f"{name:<40s} | {res['d']:>+7.3f} {res['auc']:>6.3f} {res['conc']:>6.1%} {res['wr']:>5.0%} | "
              f"{res['c_mean']:>9.4f} {res['ic_mean']:>9.4f}")

for name, fn, direction in composite_fns:
    vc = [fn(r) for r in wb if r['correct'] and not np.isnan(fn(r))]
    vic = [fn(r) for r in wb if not r['correct'] and not np.isnan(fn(r))]
    res = evaluate(name, vc, vic, wb, fn, direction)
    if res:
        all_results.append(res)
        print(f"{name:<40s} | {res['d']:>+7.3f} {res['auc']:>6.3f} {res['conc']:>6.1%} {res['wr']:>5.0%} | "
              f"{res['c_mean']:>9.4f} {res['ic_mean']:>9.4f}")

# ============================================================
# Rank and summarize
# ============================================================
print("\n" + "=" * 95)
print("TOP 10 BY CONCORDANCE (most important for GRPO)")
print("=" * 95)
by_conc = sorted(all_results, key=lambda x: x['conc'], reverse=True)
for i, r in enumerate(by_conc[:10]):
    print(f"  #{i+1} {r['name']:<38s} Conc={r['conc']:>6.1%}  d={r['d']:>+6.3f}  WR={r['wr']:>5.0%}  AUC={r['auc']:.3f}")

print("\n" + "=" * 95)
print("TOP 10 BY COHEN'S D (population-level separation)")
print("=" * 95)
by_d = sorted(all_results, key=lambda x: x['d'], reverse=True)
for i, r in enumerate(by_d[:10]):
    print(f"  #{i+1} {r['name']:<38s} d={r['d']:>+6.3f}  Conc={r['conc']:>6.1%}  WR={r['wr']:>5.0%}")

print("\n" + "=" * 95)
print("TOP 10 BY PROMPT WIN RATE")
print("=" * 95)
by_wr = sorted(all_results, key=lambda x: (x['wr'], x['conc']), reverse=True)
for i, r in enumerate(by_wr[:10]):
    print(f"  #{i+1} {r['name']:<38s} WR={r['wr']:>5.0%}  Conc={r['conc']:>6.1%}  d={r['d']:>+6.3f}")

# ============================================================
# Detailed stats for top candidates
# ============================================================
top_names = set()
for lst in [by_conc[:5], by_d[:5], by_wr[:5]]:
    for r in lst: top_names.add(r['name'])

print("\n" + "=" * 95)
print("DETAILED STATS FOR TOP CANDIDATES")
print("=" * 95)
for r in all_results:
    if r['name'] not in top_names: continue
    print(f"\n  {r['name']}")
    print(f"    Correct:   mean={r['c_mean']:.4f}, median={r['c_med']:.4f}, n={r['n_c']}")
    print(f"    Incorrect: mean={r['ic_mean']:.4f}, median={r['ic_med']:.4f}, n={r['n_ic']}")
    print(f"    Cohen's d={r['d']:+.4f}, AUC={r['auc']:.4f}, p={r['u_p']:.2e}")
    print(f"    Concordance={r['conc']:.1%}, WinRate={r['wr']:.0%} ({r['pw']}/{r['pt']})")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('RevKL-Based Unsupervised Reward: Candidate Comparison', fontsize=16, fontweight='bold')

# Sort all by concordance
sorted_res = sorted(all_results, key=lambda x: x['conc'], reverse=True)
names = [r['name'].split(':')[1].strip()[:30] for r in sorted_res]
concs = [r['conc'] for r in sorted_res]
ds = [r['d'] for r in sorted_res]
wrs = [r['wr'] for r in sorted_res]

# 1. Concordance bar
ax = axes[0][0]
colors = ['green' if c > 0.6 else 'gold' if c > 0.5 else 'red' for c in concs]
ax.barh(range(len(names)), concs, color=colors, alpha=0.7)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random 50%')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Concordance'); ax.set_title('Concordance (GRPO ranking)')
ax.legend(fontsize=8); ax.invert_yaxis()

# 2. Cohen's d bar
ax = axes[0][1]
colors = ['green' if d > 0.5 else 'gold' if d > 0 else 'red' for d in ds]
ax.barh(range(len(names)), ds, color=colors, alpha=0.7)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.3)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("Cohen's d"); ax.set_title("Cohen's d (effect size)")
ax.invert_yaxis()

# 3. Win rate bar
ax = axes[0][2]
colors = ['green' if w > 0.7 else 'gold' if w > 0.5 else 'red' for w in wrs]
ax.barh(range(len(names)), wrs, color=colors, alpha=0.7)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Prompt Win Rate'); ax.set_title('Prompt-Level Win Rate')
ax.invert_yaxis()

# 4-6. Distribution plots for top 3 by concordance
for idx, r in enumerate(by_conc[:3]):
    ax = axes[1][idx]
    fn = None
    for n2, k2, d2 in candidates:
        if n2 == r['name']:
            fn = lambda rec, k=k2: rec[k]; break
    if fn is None:
        for n2, fn2, d2 in composite_fns:
            if n2 == r['name']:
                fn = fn2; break
    if fn is None: continue
    vc = [fn(rec) for rec in wb if rec['correct'] and not np.isnan(fn(rec))]
    vic = [fn(rec) for rec in wb if not rec['correct'] and not np.isnan(fn(rec))]
    lo = min(min(vc), min(vic)); hi = max(np.percentile(vc, 95), np.percentile(vic, 95))
    bins = np.linspace(lo, hi, 35)
    ax.hist(vc, bins=bins, alpha=0.5, color='green', density=True, label=f'Correct (n={len(vc)})')
    ax.hist(vic, bins=bins, alpha=0.5, color='red', density=True, label=f'Incorrect (n={len(vic)})')
    ax.axvline(np.mean(vc), color='green', linestyle='--', linewidth=2)
    ax.axvline(np.mean(vic), color='red', linestyle='--', linewidth=2)
    ax.set_title(f"{r['name']}\nConc={r['conc']:.1%}, d={r['d']:+.3f}, WR={r['wr']:.0%}", fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{base_dir}/revkl_reward_candidates.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: revkl_reward_candidates.png")
