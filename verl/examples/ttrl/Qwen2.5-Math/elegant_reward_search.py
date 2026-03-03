"""
Parameter-free unsupervised reward candidates.
No sliding window, no bin count, no hyperparameters.
Each metric takes two sets of scalars (reasoning entropies, boxed entropies)
and returns a single number.
"""

import json, re, math, numpy as np
from collections import defaultdict
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ============================================================
# Load
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

# ============================================================
# Parameter-free distribution distance metrics
# ============================================================

def wasserstein_1d(a, b):
    """W1 Earth Mover's Distance. Closed-form for 1D: integral |F_a - F_b| dx."""
    return float(sp_stats.wasserstein_distance(a, b))

def energy_distance(a, b):
    """Energy distance = 2*E|X-Y| - E|X-X'| - E|Y-Y'|."""
    return float(sp_stats.energy_distance(a, b))

def ks_statistic(a, b):
    """Kolmogorov-Smirnov: max |F_a(x) - F_b(x)|."""
    stat, _ = sp_stats.ks_2samp(a, b)
    return float(stat)

def cvm_statistic(a, b):
    """Cramer-von Mises: integral (F_a - F_b)^2 dx. More sensitive than KS."""
    res = sp_stats.cramervonmises_2samp(a, b)
    return float(res.statistic)

def mean_diff(a, b):
    """Simple |mean(a) - mean(b)|."""
    return abs(float(np.mean(a)) - float(np.mean(b)))

def median_diff(a, b):
    """Simple |median(a) - median(b)|."""
    return abs(float(np.median(a)) - float(np.median(b)))

def quantile_divergence(a, b, qs=[0.25, 0.5, 0.75, 0.9]):
    """Sum of squared quantile differences."""
    return float(sum((np.quantile(a, q) - np.quantile(b, q))**2 for q in qs))

def tail_ratio(a, b):
    """Fraction of reasoning tokens with entropy > max(boxed entropy)."""
    if len(b) == 0: return 0.0
    threshold = np.max(b)
    return float(np.mean(a > threshold))

def zero_ratio_diff(a, b):
    """Difference in zero-entropy fraction."""
    return float(np.mean(b == 0) - np.mean(a == 0))

def boxed_coverage(reason_ent, boxed_ent):
    """
    What fraction of reasoning tokens fall within the boxed entropy range?
    Higher = reasoning is "contained" within boxed distribution support.
    """
    if len(boxed_ent) < 2: return 0.0
    lo, hi = np.min(boxed_ent), np.max(boxed_ent)
    return float(np.mean((reason_ent >= lo) & (reason_ent <= hi)))

def mmd_rbf(a, b):
    """
    Maximum Mean Discrepancy with RBF kernel.
    Bandwidth = median heuristic (parameter-free).
    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    a = a.reshape(-1, 1).astype(np.float64)
    b = b.reshape(-1, 1).astype(np.float64)
    ab = np.vstack([a, b])
    dists = cdist(ab, ab, 'sqeuclidean')
    median_dist = np.median(dists[dists > 0])
    if median_dist < 1e-10: return 0.0
    gamma = 1.0 / (2 * median_dist)
    K = np.exp(-gamma * dists)
    na, nb = len(a), len(b)
    Kxx = K[:na, :na]; Kyy = K[na:, na:]; Kxy = K[:na, na:]
    mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return float(max(0, mmd2))

def per_token_nll(reason_ent, boxed_ent):
    """
    Average negative log-likelihood of each reasoning token's entropy
    under the boxed distribution (KDE, bandwidth=Silverman).
    Lower = reasoning entropies are more "typical" of boxed distribution.
    """
    if len(boxed_ent) < 2: return 10.0
    try:
        kde = sp_stats.gaussian_kde(boxed_ent, bw_method='silverman')
        log_probs = kde.logpdf(reason_ent)
        return float(-np.mean(log_probs))
    except Exception:
        return 10.0

def per_token_nll_scott(reason_ent, boxed_ent):
    """Same but with Scott bandwidth."""
    if len(boxed_ent) < 2: return 10.0
    try:
        kde = sp_stats.gaussian_kde(boxed_ent, bw_method='scott')
        log_probs = kde.logpdf(reason_ent)
        return float(-np.mean(log_probs))
    except Exception:
        return 10.0

# ============================================================
# Compute per-response features
# ============================================================
records = []
for resp in all_resp:
    ent = resp['ent']; n = len(ent)
    bi = boxed_idx(resp['tokens'])
    if not bi:
        records.append({**{k:resp[k] for k in ['tag','pid','correct']},
                        'has_boxed': False})
        continue

    boxed_ent = ent[bi]
    bs = set(bi)
    reason_ent = ent[[i for i in range(n) if i not in bs]]

    if len(reason_ent) < 5 or len(boxed_ent) < 2:
        records.append({**{k:resp[k] for k in ['tag','pid','correct']}, 'has_boxed': False})
        continue

    rec = {
        'tag': resp['tag'], 'pid': resp['pid'], 'correct': resp['correct'],
        'has_boxed': True, 'n_tokens': n, 'n_boxed': len(bi),
    }

    # --- Parameter-free metrics (the point of this script) ---
    rec['wasserstein'] = wasserstein_1d(reason_ent, boxed_ent)
    rec['energy_dist'] = energy_distance(reason_ent, boxed_ent)
    rec['ks_stat'] = ks_statistic(reason_ent, boxed_ent)
    rec['cvm_stat'] = cvm_statistic(reason_ent, boxed_ent)
    rec['mean_diff'] = mean_diff(reason_ent, boxed_ent)
    rec['median_diff'] = median_diff(reason_ent, boxed_ent)
    rec['quantile_div'] = quantile_divergence(reason_ent, boxed_ent)
    rec['tail_ratio'] = tail_ratio(reason_ent, boxed_ent)
    rec['zero_diff'] = zero_ratio_diff(reason_ent, boxed_ent)
    rec['boxed_coverage'] = boxed_coverage(reason_ent, boxed_ent)
    rec['mmd'] = mmd_rbf(reason_ent, boxed_ent)
    rec['nll_silverman'] = per_token_nll(reason_ent, boxed_ent)
    rec['nll_scott'] = per_token_nll_scott(reason_ent, boxed_ent)

    # --- Derived elegant metrics ---
    rec['neg_wasserstein'] = -rec['wasserstein']
    rec['neg_energy'] = -rec['energy_dist']
    rec['neg_ks'] = -rec['ks_stat']
    rec['neg_nll'] = -rec['nll_silverman']
    rec['neg_mmd'] = -rec['mmd']
    rec['exp_neg_w'] = math.exp(-rec['wasserstein']) if rec['wasserstein'] < 50 else 0.0
    rec['inv_w'] = 1.0 / (rec['wasserstein'] + 0.01)
    rec['neg_mean_diff'] = -rec['mean_diff']
    rec['neg_cvm'] = -rec['cvm_stat']

    # Simple baselines
    rec['neg_h_all'] = -float(np.mean(ent))
    rec['neg_h_boxed'] = -float(np.mean(boxed_ent))
    rec['boxed_zero'] = float(np.mean(boxed_ent == 0))

    records.append(rec)

wb = [r for r in records if r.get('has_boxed', False)]
print(f"With boxed: {len(wb)}, Correct: {sum(r['correct'] for r in wb)}, "
      f"Incorrect: {sum(not r['correct'] for r in wb)}")

# ============================================================
# Evaluate
# ============================================================
def evaluate(name, data, val_fn, direction='higher'):
    vc = [val_fn(r) for r in data if r['correct']]
    vic = [val_fn(r) for r in data if not r['correct']]
    vc = [v for v in vc if v is not None and not np.isnan(v) and not np.isinf(v)]
    vic = [v for v in vic if v is not None and not np.isnan(v) and not np.isinf(v)]
    if len(vc) < 3 or len(vic) < 3: return None
    c, ic = np.array(vc), np.array(vic)
    pooled = np.sqrt((c.var()*len(c)+ic.var()*len(ic))/(len(c)+len(ic)))
    d = (c.mean()-ic.mean())/(pooled+1e-10) if direction=='higher' else (ic.mean()-c.mean())/(pooled+1e-10)
    alt = 'greater' if direction=='higher' else 'less'
    u,p = sp_stats.mannwhitneyu(c,ic,alternative=alt)
    auc = u/(len(c)*len(ic))

    pg = defaultdict(lambda:{'c':[],'ic':[]})
    for r in data:
        v = val_fn(r)
        if v is None or np.isnan(v) or np.isinf(v): continue
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
    return {'name':name,'d':d,'auc':auc,
            'conc':conc/tot if tot else 0, 'wr':pw/pt if pt else 0,
            'c_mean':c.mean(),'ic_mean':ic.mean(),
            'c_med':np.median(c),'ic_med':np.median(ic),
            'pw':pw,'pt':pt,'p':p}

candidates = [
    # --- Parameter-free distribution distances (negated: lower dist = better) ---
    ("W1: -Wasserstein (EMD)",         lambda r: r['neg_wasserstein'],    'higher'),
    ("W2: exp(-Wasserstein)",          lambda r: r['exp_neg_w'],          'higher'),
    ("W3: 1/(Wasserstein+0.01)",       lambda r: r['inv_w'],              'higher'),
    ("E1: -Energy distance",           lambda r: r['neg_energy'],         'higher'),
    ("K1: -KS statistic",             lambda r: r['neg_ks'],              'higher'),
    ("K2: -Cramer-von-Mises",         lambda r: r['neg_cvm'],             'higher'),
    ("M1: -MMD (RBF, median heuristic)", lambda r: r['neg_mmd'],          'higher'),
    # --- Per-token likelihood ---
    ("N1: -NLL_silverman (KDE)",       lambda r: r['neg_nll'],            'higher'),
    ("N2: -NLL_scott (KDE)",           lambda r: -r['nll_scott'],         'higher'),
    # --- Simple statistics ---
    ("S1: -|mean_diff|",              lambda r: r['neg_mean_diff'],       'higher'),
    ("S2: -|median_diff|",            lambda r: -r['median_diff'],        'higher'),
    ("S3: -quantile_div",             lambda r: -r['quantile_div'],       'higher'),
    ("S4: -tail_ratio",               lambda r: -r['tail_ratio'],         'higher'),
    ("S5: boxed_coverage",            lambda r: r['boxed_coverage'],      'higher'),
    ("S6: zero_ratio_diff (boxed-reason)", lambda r: r['zero_diff'],      'higher'),
    # --- Baselines ---
    ("B1: -H_all",                     lambda r: r['neg_h_all'],          'higher'),
    ("B2: -H_boxed",                   lambda r: r['neg_h_boxed'],        'higher'),
    ("B3: boxed_zero_ratio",           lambda r: r['boxed_zero'],         'higher'),
]

print("\n" + "=" * 95)
print("PARAMETER-FREE REWARD CANDIDATES")
print("=" * 95)
print(f"{'Candidate':<42s} | {'d':>7s} {'AUC':>6s} {'Conc':>7s} {'WR':>6s} | {'C':>9s} {'IC':>9s} | params")
print("-" * 95)

all_results = []
for name, fn, direction in candidates:
    res = evaluate(name, wb, fn, direction)
    if res:
        all_results.append(res)
        param_free = "NONE" if name.startswith(('W','E','K','M','S')) else ("kde_bw" if name.startswith('N') else "bins" if 'RevKL' in name else "NONE")
        print(f"{name:<42s} | {res['d']:>+7.3f} {res['auc']:>6.3f} {res['conc']:>6.1%} {res['wr']:>5.0%} | "
              f"{res['c_mean']:>9.4f} {res['ic_mean']:>9.4f} | {param_free}")

# ============================================================
# Rankings
# ============================================================
print("\n" + "=" * 95)
print("RANKING BY CONCORDANCE")
print("=" * 95)
by_conc = sorted(all_results, key=lambda x: x['conc'], reverse=True)
for i, r in enumerate(by_conc):
    marker = " ***" if r['conc'] > 0.8 else " **" if r['conc'] > 0.7 else ""
    print(f"  #{i+1:>2d} {r['name']:<40s} Conc={r['conc']:>6.1%}  d={r['d']:>+6.3f}  WR={r['wr']:>5.0%}{marker}")

print("\n" + "=" * 95)
print("RANKING BY COHEN'S D")
print("=" * 95)
by_d = sorted(all_results, key=lambda x: x['d'], reverse=True)
for i, r in enumerate(by_d):
    print(f"  #{i+1:>2d} {r['name']:<40s} d={r['d']:>+6.3f}  Conc={r['conc']:>6.1%}  WR={r['wr']:>5.0%}")

# Detailed stats
print("\n" + "=" * 95)
print("DETAILED STATS FOR TOP 5")
print("=" * 95)
for r in by_conc[:5]:
    print(f"\n  {r['name']}")
    print(f"    Correct:   mean={r['c_mean']:.6f}, median={r['c_med']:.6f}")
    print(f"    Incorrect: mean={r['ic_mean']:.6f}, median={r['ic_med']:.6f}")
    print(f"    d={r['d']:+.4f}, AUC={r['auc']:.4f}, p={r['p']:.2e}")
    print(f"    Concordance={r['conc']:.1%}, WinRate={r['wr']:.0%} ({r['pw']}/{r['pt']})")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('Parameter-Free Unsupervised Reward Candidates', fontsize=16, fontweight='bold')

# Bar charts
sorted_res = sorted(all_results, key=lambda x: x['conc'], reverse=True)
names = [r['name'].split(':')[1].strip()[:32] for r in sorted_res]
concs = [r['conc'] for r in sorted_res]
ds = [r['d'] for r in sorted_res]
wrs = [r['wr'] for r in sorted_res]

ax = axes[0][0]
colors = ['green' if c > 0.8 else 'gold' if c > 0.6 else 'red' for c in concs]
ax.barh(range(len(names)), concs, color=colors, alpha=0.7)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Concordance'); ax.set_title('Concordance'); ax.invert_yaxis()

ax = axes[0][1]
colors = ['green' if d > 0.5 else 'gold' if d > 0 else 'red' for d in ds]
ax.barh(range(len(names)), ds, color=colors, alpha=0.7)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("Cohen's d"); ax.set_title("Cohen's d"); ax.invert_yaxis()

ax = axes[0][2]
colors = ['green' if w > 0.7 else 'gold' if w > 0.5 else 'red' for w in wrs]
ax.barh(range(len(names)), wrs, color=colors, alpha=0.7)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Win Rate'); ax.set_title('Prompt Win Rate'); ax.invert_yaxis()

# Distribution plots for top 3
for idx, r in enumerate(by_conc[:3]):
    ax = axes[1][idx]
    fn = None
    for n2, fn2, d2 in candidates:
        if n2 == r['name']: fn = fn2; break
    if fn is None: continue
    vc = [fn(rec) for rec in wb if rec['correct'] and not np.isnan(fn(rec)) and not np.isinf(fn(rec))]
    vic = [fn(rec) for rec in wb if not rec['correct'] and not np.isnan(fn(rec)) and not np.isinf(fn(rec))]
    lo = min(min(vc), min(vic)); hi = min(max(np.percentile(vc,97), np.percentile(vic,97)), lo+10*(np.median(vc+vic)-lo+0.1))
    bins = np.linspace(lo, hi, 35)
    ax.hist(vc, bins=bins, alpha=0.5, color='green', density=True, label=f'Correct (n={len(vc)})')
    ax.hist(vic, bins=bins, alpha=0.5, color='red', density=True, label=f'Incorrect (n={len(vic)})')
    ax.axvline(np.mean(vc), color='green', linestyle='--', linewidth=2)
    ax.axvline(np.mean(vic), color='red', linestyle='--', linewidth=2)
    ax.set_title(f"{r['name']}\nConc={r['conc']:.1%}, d={r['d']:+.3f}", fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{base_dir}/elegant_reward_candidates.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: elegant_reward_candidates.png")
