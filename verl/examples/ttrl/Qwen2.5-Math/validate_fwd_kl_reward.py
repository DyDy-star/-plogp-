"""
Validate Forward KL(token||boxed) as unsupervised reward signal.
Per-response scalar: KL(P_reasoning || P_boxed)
"""

import json
import re
import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ============================================================
# 1. Load
# ============================================================
base_dir = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy"
files = [
    f"{base_dir}/aime_eval_full_entropy_20260207_092147.json",
    f"{base_dir}/aime_eval_full_entropy_20260207_090427.json",
]

all_responses = []
for fpath in files:
    with open(fpath) as f:
        data = json.load(f)
    tag = fpath.split('_')[-1].replace('.json', '')
    for result in data['results']:
        for resp in result['responses']:
            tokens = []
            entropies = []
            for step in resp['entropy_analysis']['steps']:
                tokens.extend(step['tokens'])
                entropies.extend(step['token_entropies'])
            if len(tokens) == 0:
                continue
            all_responses.append({
                'tag': tag,
                'prompt_id': result['id'],
                'is_correct': resp['is_correct'],
                'tokens': tokens,
                'entropies': np.array(entropies, dtype=np.float32),
                'response_text': resp['response'],
            })

# ============================================================
# 2. Identify boxed tokens
# ============================================================
def find_boxed_token_indices(tokens):
    text = ""
    pos = []
    for i, t in enumerate(tokens):
        s = len(text)
        text += t
        e = len(text)
        pos.append((s, e))
    boxed_idx = set()
    answer_idx = set()
    for m in re.finditer(r'\\boxed\{', text):
        depth, p = 1, m.end()
        while p < len(text) and depth > 0:
            if text[p] == '{': depth += 1
            elif text[p] == '}': depth -= 1
            p += 1
        for i, (ts, te) in enumerate(pos):
            if ts < p and te > m.start():
                boxed_idx.add(i)
            if ts < p - 1 and te > m.end():
                answer_idx.add(i)
    return sorted(boxed_idx), sorted(answer_idx)

# ============================================================
# 3. Compute per-response KL/JS reward candidates
# ============================================================
def kl_from_histograms(p_samples, q_samples, n_bins=80, eps=1e-10):
    if len(p_samples) < 3 or len(q_samples) < 3:
        return np.nan, np.nan, np.nan
    all_s = np.concatenate([p_samples, q_samples])
    lo, hi = all_s.min(), all_s.max()
    if hi - lo < 1e-6:
        return 0.0, 0.0, 0.0
    bins = np.linspace(lo - 0.01, hi + 0.01, n_bins + 1)
    p_h, _ = np.histogram(p_samples, bins=bins, density=True)
    q_h, _ = np.histogram(q_samples, bins=bins, density=True)
    p_h = (p_h + eps); p_h /= p_h.sum()
    q_h = (q_h + eps); q_h /= q_h.sum()
    m_h = 0.5 * (p_h + q_h)
    fwd_kl = float(np.sum(p_h * np.log(p_h / q_h)))
    rev_kl = float(np.sum(q_h * np.log(q_h / p_h)))
    js = float(0.5 * np.sum(p_h * np.log(p_h / m_h)) + 0.5 * np.sum(q_h * np.log(q_h / m_h)))
    return fwd_kl, rev_kl, js

records = []
for resp in all_responses:
    ent = resp['entropies']
    n = len(ent)
    boxed_idx, answer_idx = find_boxed_token_indices(resp['tokens'])
    has_boxed = len(boxed_idx) > 0

    if has_boxed:
        boxed_set = set(boxed_idx)
        nonboxed_idx = [i for i in range(n) if i not in boxed_set]
        boxed_ent = ent[boxed_idx]
        nonboxed_ent = ent[nonboxed_idx] if nonboxed_idx else ent

        fwd_kl, rev_kl, js_div = kl_from_histograms(nonboxed_ent, boxed_ent)

        # Also compute phased FwdKL: early vs late reasoning tokens
        mid = len(nonboxed_idx) // 2
        early_idx = nonboxed_idx[:mid]
        late_idx = nonboxed_idx[mid:]
        fwd_kl_early, _, _ = kl_from_histograms(ent[early_idx], boxed_ent) if early_idx else (np.nan, np.nan, np.nan)
        fwd_kl_late, _, _ = kl_from_histograms(ent[late_idx], boxed_ent) if late_idx else (np.nan, np.nan, np.nan)

        # Convergence ratio = early_fwd_kl / late_fwd_kl (higher = more convergent)
        convergence = fwd_kl_early / fwd_kl_late if fwd_kl_late > 1e-6 else np.nan

        # KL asymmetry = fwd_kl - rev_kl (positive = reasoning more spread than answer)
        kl_asym = fwd_kl - rev_kl if not np.isnan(fwd_kl) else np.nan

        # Boxed entropy stats
        boxed_mean_h = float(np.mean(boxed_ent))
        boxed_zero_ratio = float(np.mean(boxed_ent == 0))

        # Mean reasoning entropy
        reason_mean_h = float(np.mean(nonboxed_ent))

        # Answer entropy
        answer_ent = ent[list(answer_idx)] if answer_idx else boxed_ent
        answer_mean_h = float(np.mean(answer_ent))
    else:
        fwd_kl = rev_kl = js_div = np.nan
        fwd_kl_early = fwd_kl_late = convergence = kl_asym = np.nan
        boxed_mean_h = boxed_zero_ratio = np.nan
        reason_mean_h = float(np.mean(ent))
        answer_mean_h = np.nan

    records.append({
        'tag': resp['tag'],
        'prompt_id': resp['prompt_id'],
        'is_correct': resp['is_correct'],
        'has_boxed': has_boxed,
        'n_tokens': n,
        # Primary candidates
        'fwd_kl': fwd_kl,          # KL(reasoning || boxed)
        'rev_kl': rev_kl,          # KL(boxed || reasoning)
        'js_div': js_div,          # JS(reasoning, boxed)
        'kl_asym': kl_asym,        # fwd_kl - rev_kl
        'convergence': convergence, # fwd_kl_early / fwd_kl_late
        'fwd_kl_early': fwd_kl_early,
        'fwd_kl_late': fwd_kl_late,
        # Baselines
        'boxed_mean_h': boxed_mean_h,
        'boxed_zero_ratio': boxed_zero_ratio,
        'reason_mean_h': reason_mean_h,
        'answer_mean_h': answer_mean_h,
        'all_mean_h': float(np.mean(ent)),
    })

# ============================================================
# 4. Discriminative power analysis
# ============================================================
with_boxed = [r for r in records if r['has_boxed']]
no_boxed = [r for r in records if not r['has_boxed']]

print("=" * 85)
print("FORWARD KL(token||boxed) AS UNSUPERVISED REWARD - DISCRIMINATIVE POWER")
print("=" * 85)
print(f"Total: {len(records)}, With boxed: {len(with_boxed)}, Without: {len(no_boxed)}")
print(f"Correct with boxed: {sum(1 for r in with_boxed if r['is_correct'])}")
print(f"Incorrect with boxed: {sum(1 for r in with_boxed if not r['is_correct'])}")

candidates = [
    ("R1: FwdKL(reason||boxed)",     'fwd_kl',          'higher_better'),
    ("R2: KL Asymmetry (Fwd-Rev)",   'kl_asym',         'higher_better'),
    ("R3: JS(reason, boxed)",        'js_div',           'higher_better'),
    ("R4: RevKL(boxed||reason)",     'rev_kl',           'higher_better'),
    ("R5: Convergence (early/late)", 'convergence',      'higher_better'),
    ("R6: FwdKL_early",             'fwd_kl_early',      'higher_better'),
    ("R7: FwdKL_late",              'fwd_kl_late',       'higher_better'),
    ("R8: -H_boxed (baseline)",      'boxed_mean_h',     'lower_better'),
    ("R9: -H_answer (baseline)",     'answer_mean_h',    'lower_better'),
    ("R10: -H_all (baseline)",       'all_mean_h',       'lower_better'),
    ("R11: boxed_zero_ratio",        'boxed_zero_ratio', 'higher_better'),
    ("R12: H_reason - H_boxed",      None,               'higher_better'),
]

def evaluate_candidate(name, key, direction, data):
    if key is not None:
        vals_c = [r[key] for r in data if r['is_correct'] and not np.isnan(r[key])]
        vals_ic = [r[key] for r in data if not r['is_correct'] and not np.isnan(r[key])]
    else:
        vals_c = [r['reason_mean_h'] - r['boxed_mean_h'] for r in data if r['is_correct'] and not np.isnan(r['boxed_mean_h'])]
        vals_ic = [r['reason_mean_h'] - r['boxed_mean_h'] for r in data if not r['is_correct'] and not np.isnan(r['boxed_mean_h'])]

    if len(vals_c) < 3 or len(vals_ic) < 3:
        return None

    c, ic = np.array(vals_c), np.array(vals_ic)

    # Cohen's d
    pooled = np.sqrt((c.var() * len(c) + ic.var() * len(ic)) / (len(c) + len(ic)))
    if direction == 'higher_better':
        d = (c.mean() - ic.mean()) / (pooled + 1e-10)
    else:
        d = (ic.mean() - c.mean()) / (pooled + 1e-10)

    # AUC via Mann-Whitney
    alt = 'greater' if direction == 'higher_better' else 'less'
    u_stat, u_p = stats.mannwhitneyu(c, ic, alternative=alt)
    auc = u_stat / (len(c) * len(ic))

    # Per-prompt concordance (GRPO-style)
    concordant, total_pairs, n_prompts = 0, 0, 0
    prompt_groups = defaultdict(lambda: {'c': [], 'ic': []})
    for r in data:
        if key is not None:
            v = r[key]
        else:
            v = r['reason_mean_h'] - r['boxed_mean_h'] if not np.isnan(r.get('boxed_mean_h', np.nan)) else np.nan
        if np.isnan(v):
            continue
        k = (r['tag'], r['prompt_id'])
        if r['is_correct']:
            prompt_groups[k]['c'].append(v)
        else:
            prompt_groups[k]['ic'].append(v)

    prompt_wins = 0
    prompt_total = 0
    for k, g in prompt_groups.items():
        if g['c'] and g['ic']:
            n_prompts += 1
            for cv in g['c']:
                for iv in g['ic']:
                    total_pairs += 1
                    if direction == 'higher_better':
                        if cv > iv: concordant += 1
                    else:
                        if cv < iv: concordant += 1
            c_m = np.mean(g['c'])
            ic_m = np.mean(g['ic'])
            prompt_total += 1
            if direction == 'higher_better':
                if c_m > ic_m: prompt_wins += 1
            else:
                if c_m < ic_m: prompt_wins += 1

    concordance = concordant / total_pairs if total_pairs > 0 else 0
    win_rate = prompt_wins / prompt_total if prompt_total > 0 else 0

    return {
        'name': name,
        'c_mean': c.mean(), 'c_std': c.std(), 'c_median': np.median(c),
        'ic_mean': ic.mean(), 'ic_std': ic.std(), 'ic_median': np.median(ic),
        'cohens_d': d,
        'auc': auc,
        'u_p': u_p,
        'concordance': concordance,
        'n_pairs': total_pairs,
        'n_prompts': n_prompts,
        'win_rate': win_rate,
        'prompt_wins': prompt_wins,
        'prompt_total': prompt_total,
    }

print("\n" + "-" * 85)
print(f"{'Candidate':<32s} | {'Cohen d':>8s} {'AUC':>6s} {'Concord':>8s} {'WinRate':>8s} | {'C_mean':>8s} {'IC_mean':>8s}")
print("-" * 85)

results = {}
for name, key, direction in candidates:
    res = evaluate_candidate(name, key, direction, with_boxed)
    if res is None:
        print(f"{name:<32s} | insufficient data")
        continue
    results[name] = res
    print(f"{name:<32s} | {res['cohens_d']:>8.3f} {res['auc']:>6.3f} {res['concordance']:>7.1%} {res['win_rate']:>7.1%} | "
          f"{res['c_mean']:>8.4f} {res['ic_mean']:>8.4f}")

# ============================================================
# 5. Detailed stats for top candidates
# ============================================================
print("\n" + "=" * 85)
print("DETAILED STATS FOR TOP CANDIDATES")
print("=" * 85)

for name in ["R1: FwdKL(reason||boxed)", "R2: KL Asymmetry (Fwd-Rev)",
             "R5: Convergence (early/late)", "R8: -H_boxed (baseline)"]:
    if name not in results:
        continue
    r = results[name]
    print(f"\n--- {name} ---")
    print(f"  Correct:   n={sum(1 for x in with_boxed if x['is_correct'])}, "
          f"mean={r['c_mean']:.4f}, std={r['c_std']:.4f}, median={r['c_median']:.4f}")
    print(f"  Incorrect: n={sum(1 for x in with_boxed if not x['is_correct'])}, "
          f"mean={r['ic_mean']:.4f}, std={r['ic_std']:.4f}, median={r['ic_median']:.4f}")
    print(f"  Cohen's d:     {r['cohens_d']:.4f}")
    print(f"  AUC:           {r['auc']:.4f}")
    print(f"  Mann-Whitney p:{r['u_p']:.2e}")
    print(f"  Concordance:   {r['concordance']:.1%} ({int(r['concordance']*r['n_pairs'])}/{r['n_pairs']} pairs)")
    print(f"  Prompt WinRate:{r['win_rate']:.1%} ({r['prompt_wins']}/{r['prompt_total']} prompts)")

# ============================================================
# 6. Combined metric: FwdKL * Convergence
# ============================================================
print("\n" + "=" * 85)
print("COMPOSITE REWARD CANDIDATES")
print("=" * 85)

composites = [
    ("C1: FwdKL * Convergence",
     lambda r: r['fwd_kl'] * r['convergence'] if not np.isnan(r['fwd_kl']) and not np.isnan(r['convergence']) else np.nan),
    ("C2: FwdKL * (1+KLAsym)",
     lambda r: r['fwd_kl'] * (1 + max(0, r['kl_asym'])) if not np.isnan(r['fwd_kl']) and not np.isnan(r['kl_asym']) else np.nan),
    ("C3: JS * Convergence",
     lambda r: r['js_div'] * r['convergence'] if not np.isnan(r['js_div']) and not np.isnan(r['convergence']) else np.nan),
    ("C4: FwdKL - RevKL + JS",
     lambda r: r['fwd_kl'] - r['rev_kl'] + r['js_div'] if not any(np.isnan(r[k]) for k in ['fwd_kl','rev_kl','js_div']) else np.nan),
    ("C5: log(FwdKL+1) * Conv",
     lambda r: np.log(r['fwd_kl'] + 1) * r['convergence'] if not np.isnan(r['fwd_kl']) and not np.isnan(r['convergence']) else np.nan),
    ("C6: FwdKL / (RevKL+0.1)",
     lambda r: r['fwd_kl'] / (r['rev_kl'] + 0.1) if not np.isnan(r['fwd_kl']) and not np.isnan(r['rev_kl']) else np.nan),
]

print(f"\n{'Composite':<32s} | {'Cohen d':>8s} {'AUC':>6s} {'Concord':>8s} {'WinRate':>8s} | {'C_mean':>8s} {'IC_mean':>8s}")
print("-" * 85)

for name, fn in composites:
    vals_c = [fn(r) for r in with_boxed if r['is_correct'] and not np.isnan(fn(r))]
    vals_ic = [fn(r) for r in with_boxed if not r['is_correct'] and not np.isnan(fn(r))]
    if len(vals_c) < 3 or len(vals_ic) < 3:
        print(f"{name:<32s} | insufficient data")
        continue
    c, ic = np.array(vals_c), np.array(vals_ic)
    pooled = np.sqrt((c.var()*len(c) + ic.var()*len(ic))/(len(c)+len(ic)))
    d = (c.mean() - ic.mean()) / (pooled + 1e-10)
    u_stat, _ = stats.mannwhitneyu(c, ic, alternative='greater')
    auc = u_stat / (len(c) * len(ic))

    # Concordance
    conc, tot = 0, 0
    prompt_groups = defaultdict(lambda: {'c': [], 'ic': []})
    for r in with_boxed:
        v = fn(r)
        if np.isnan(v): continue
        k = (r['tag'], r['prompt_id'])
        if r['is_correct']: prompt_groups[k]['c'].append(v)
        else: prompt_groups[k]['ic'].append(v)
    pw, pt = 0, 0
    for k, g in prompt_groups.items():
        if g['c'] and g['ic']:
            pt += 1
            if np.mean(g['c']) > np.mean(g['ic']): pw += 1
            for cv in g['c']:
                for iv in g['ic']:
                    tot += 1
                    if cv > iv: conc += 1
    concordance = conc / tot if tot > 0 else 0
    wr = pw / pt if pt > 0 else 0

    print(f"{name:<32s} | {d:>8.3f} {auc:>6.3f} {concordance:>7.1%} {wr:>7.1%} | {c.mean():>8.4f} {ic.mean():>8.4f}")

# ============================================================
# 7. Per-file (checkpoint) analysis
# ============================================================
print("\n" + "=" * 85)
print("PER-CHECKPOINT DISCRIMINATIVE POWER")
print("=" * 85)

for tag_label in ["092147", "090427"]:
    subset = [r for r in with_boxed if r['tag'] == tag_label]
    nc = sum(1 for r in subset if r['is_correct'])
    nic = sum(1 for r in subset if not r['is_correct'])
    print(f"\n--- Checkpoint {tag_label} (correct={nc}, incorrect={nic}) ---")

    for name, key, direction in candidates[:7]:
        res = evaluate_candidate(name, key, direction, subset)
        if res is None:
            print(f"  {name:<30s} | insufficient data")
            continue
        print(f"  {name:<30s} | d={res['cohens_d']:>7.3f} AUC={res['auc']:>5.3f} "
              f"Conc={res['concordance']:>6.1%} WR={res['win_rate']:>6.1%} "
              f"C={res['c_mean']:>7.4f} IC={res['ic_mean']:>7.4f}")

# ============================================================
# 8. Sensitivity: what if we include responses without boxed?
# ============================================================
print("\n" + "=" * 85)
print("COVERAGE ANALYSIS: Including responses WITHOUT \\boxed{}")
print("=" * 85)

for penalty_name, penalty_val in [("penalty=0", 0.0), ("penalty=-0.5", -0.5), ("penalty=min(with_boxed)", None)]:
    if penalty_val is None:
        all_fwd_kl = [r['fwd_kl'] for r in with_boxed if not np.isnan(r['fwd_kl'])]
        penalty_val = min(all_fwd_kl) if all_fwd_kl else 0.0

    extended = []
    for r in records:
        if r['has_boxed'] and not np.isnan(r['fwd_kl']):
            extended.append({'val': r['fwd_kl'], 'is_correct': r['is_correct'],
                             'tag': r['tag'], 'prompt_id': r['prompt_id']})
        else:
            extended.append({'val': penalty_val, 'is_correct': r['is_correct'],
                             'tag': r['tag'], 'prompt_id': r['prompt_id']})

    vals_c = [r['val'] for r in extended if r['is_correct']]
    vals_ic = [r['val'] for r in extended if not r['is_correct']]
    c, ic = np.array(vals_c), np.array(vals_ic)
    pooled = np.sqrt((c.var()*len(c) + ic.var()*len(ic))/(len(c)+len(ic)))
    d = (c.mean() - ic.mean()) / (pooled + 1e-10)
    u_stat, _ = stats.mannwhitneyu(c, ic, alternative='greater')
    auc = u_stat / (len(c) * len(ic))

    conc, tot, pw, pt = 0, 0, 0, 0
    pg = defaultdict(lambda: {'c':[],'ic':[]})
    for r in extended:
        k = (r['tag'], r['prompt_id'])
        if r['is_correct']: pg[k]['c'].append(r['val'])
        else: pg[k]['ic'].append(r['val'])
    for k, g in pg.items():
        if g['c'] and g['ic']:
            pt += 1
            if np.mean(g['c']) > np.mean(g['ic']): pw += 1
            for cv in g['c']:
                for iv in g['ic']:
                    tot += 1
                    if cv > iv: conc += 1

    concordance = conc/tot if tot>0 else 0
    wr = pw/pt if pt>0 else 0
    print(f"  {penalty_name:<25s} | d={d:.3f} AUC={auc:.3f} Conc={concordance:.1%} WR={wr:.1%} "
          f"(n_c={len(c)}, n_ic={len(ic)})")

# ============================================================
# 9. Visualization
# ============================================================
fig = plt.figure(figsize=(22, 18))
gs = matplotlib.gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle('FwdKL(reasoning||boxed) as Unsupervised Reward: Discriminative Power', fontsize=16, fontweight='bold')

# 9a: Distribution comparison for top candidates
top_metrics = [
    ("FwdKL(reason||boxed)", 'fwd_kl', 'higher'),
    ("KL Asymmetry", 'kl_asym', 'higher'),
    ("Convergence (early/late)", 'convergence', 'higher'),
]

for idx, (label, key, _) in enumerate(top_metrics):
    ax = fig.add_subplot(gs[0, idx])
    c_vals = [r[key] for r in with_boxed if r['is_correct'] and not np.isnan(r[key])]
    ic_vals = [r[key] for r in with_boxed if not r['is_correct'] and not np.isnan(r[key])]
    bins = np.linspace(min(min(c_vals), min(ic_vals)),
                       np.percentile(c_vals + ic_vals, 95), 40)
    ax.hist(c_vals, bins=bins, alpha=0.5, color='green', density=True, label=f'Correct (n={len(c_vals)})')
    ax.hist(ic_vals, bins=bins, alpha=0.5, color='red', density=True, label=f'Incorrect (n={len(ic_vals)})')
    ax.axvline(np.mean(c_vals), color='green', linestyle='--', linewidth=2)
    ax.axvline(np.mean(ic_vals), color='red', linestyle='--', linewidth=2)
    r = results.get(f"R{idx+1}: {label.split('(')[0].strip()}", results.get([k for k in results if label[:8] in k][0], None)) if results else None
    title = f"{label}\n"
    if key in ['fwd_kl']:
        res = results.get("R1: FwdKL(reason||boxed)")
        if res: title += f"d={res['cohens_d']:.3f}, Conc={res['concordance']:.1%}"
    elif key == 'kl_asym':
        res = results.get("R2: KL Asymmetry (Fwd-Rev)")
        if res: title += f"d={res['cohens_d']:.3f}, Conc={res['concordance']:.1%}"
    elif key == 'convergence':
        res = results.get("R5: Convergence (early/late)")
        if res: title += f"d={res['cohens_d']:.3f}, Conc={res['concordance']:.1%}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel('Density')

# 9b: Bar chart of all Cohen's d
ax = fig.add_subplot(gs[1, 0])
names = [n for n in results]
d_vals = [results[n]['cohens_d'] for n in names]
colors = ['green' if d > 0.5 else 'orange' if d > 0.2 else 'red' for d in d_vals]
bars = ax.barh(range(len(names)), d_vals, color=colors, alpha=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.split(':')[1].strip()[:25] for n in names], fontsize=8)
ax.set_xlabel("Cohen's d")
ax.set_title("Cohen's d (all candidates)", fontsize=11)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='d=0.5 (medium)')
ax.axvline(0.8, color='gray', linestyle=':', alpha=0.5, label='d=0.8 (large)')
ax.legend(fontsize=8)
for bar, d in zip(bars, d_vals):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{d:.3f}', va='center', fontsize=8)

# 9c: Bar chart of Concordance
ax = fig.add_subplot(gs[1, 1])
conc_vals = [results[n]['concordance'] for n in names]
colors = ['green' if c > 0.6 else 'orange' if c > 0.5 else 'red' for c in conc_vals]
bars = ax.barh(range(len(names)), conc_vals, color=colors, alpha=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.split(':')[1].strip()[:25] for n in names], fontsize=8)
ax.set_xlabel('Concordance Rate')
ax.set_title('Concordance (GRPO pairwise)', fontsize=11)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
ax.legend(fontsize=8)
for bar, c in zip(bars, conc_vals):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{c:.1%}', va='center', fontsize=8)

# 9d: Bar chart of Prompt WinRate
ax = fig.add_subplot(gs[1, 2])
wr_vals = [results[n]['win_rate'] for n in names]
colors = ['green' if w > 0.7 else 'orange' if w > 0.5 else 'red' for w in wr_vals]
bars = ax.barh(range(len(names)), wr_vals, color=colors, alpha=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.split(':')[1].strip()[:25] for n in names], fontsize=8)
ax.set_xlabel('Prompt Win Rate')
ax.set_title('Prompt-Level Win Rate', fontsize=11)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
for bar, w in zip(bars, wr_vals):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.0%}', va='center', fontsize=8)

# 9e: Scatter plot: FwdKL vs correctness
ax = fig.add_subplot(gs[2, 0])
c_fwd = [r['fwd_kl'] for r in with_boxed if r['is_correct'] and not np.isnan(r['fwd_kl'])]
ic_fwd = [r['fwd_kl'] for r in with_boxed if not r['is_correct'] and not np.isnan(r['fwd_kl'])]
ax.scatter(range(len(c_fwd)), sorted(c_fwd), s=20, color='green', alpha=0.6, label=f'Correct (n={len(c_fwd)})')
ax.scatter(range(len(ic_fwd)), sorted(ic_fwd), s=8, color='red', alpha=0.3, label=f'Incorrect (n={len(ic_fwd)})')
ax.set_xlabel('Rank')
ax.set_ylabel('FwdKL(reason||boxed)')
ax.set_title('Sorted FwdKL Values', fontsize=11)
ax.legend(fontsize=9)

# 9f: Per-prompt boxplot
ax = fig.add_subplot(gs[2, 1:])
prompt_data = defaultdict(lambda: {'c': [], 'ic': []})
for r in with_boxed:
    if np.isnan(r['fwd_kl']): continue
    k = f"{r['tag']}_{r['prompt_id']}"
    if r['is_correct']: prompt_data[k]['c'].append(r['fwd_kl'])
    else: prompt_data[k]['ic'].append(r['fwd_kl'])

valid_prompts = {k: v for k, v in prompt_data.items() if v['c'] and v['ic']}
prompt_names = sorted(valid_prompts.keys())
positions = np.arange(len(prompt_names))
for i, pname in enumerate(prompt_names):
    g = valid_prompts[pname]
    c_vals = g['c']
    ic_vals = g['ic']
    ax.scatter([i - 0.15] * len(c_vals), c_vals, s=40, color='green', alpha=0.7, zorder=3)
    ax.scatter([i + 0.15] * len(ic_vals), ic_vals, s=15, color='red', alpha=0.4, zorder=3)
    ax.plot([i - 0.15], [np.mean(c_vals)], 'g^', markersize=12, zorder=4)
    ax.plot([i + 0.15], [np.mean(ic_vals)], 'rv', markersize=12, zorder=4)
ax.set_xticks(positions)
ax.set_xticklabels(prompt_names, rotation=45, fontsize=8, ha='right')
ax.set_ylabel('FwdKL(reason||boxed)')
ax.set_title('Per-Prompt: Correct (green) vs Incorrect (red)', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.savefig(f"{base_dir}/fwd_kl_reward_validation.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: fwd_kl_reward_validation.png")

print("\n" + "=" * 85)
print("FINAL VERDICT")
print("=" * 85)
r1 = results.get("R1: FwdKL(reason||boxed)", {})
r8 = results.get("R8: -H_boxed (baseline)", {})
r10 = results.get("R10: -H_all (baseline)", {})
print(f"""
FwdKL(reasoning||boxed) as unsupervised reward:
  Cohen's d:       {r1.get('cohens_d', 0):.3f}
  AUC:             {r1.get('auc', 0):.3f}
  Concordance:     {r1.get('concordance', 0):.1%}
  Prompt Win Rate: {r1.get('win_rate', 0):.1%}
  Coverage:        {len(with_boxed)}/{len(records)} = {len(with_boxed)/len(records)*100:.1f}%

vs -H_boxed baseline:
  Cohen's d:       {r8.get('cohens_d', 0):.3f}
  Concordance:     {r8.get('concordance', 0):.1%}
  
vs -H_all baseline:
  Cohen's d:       {r10.get('cohens_d', 0):.3f}
  Concordance:     {r10.get('concordance', 0):.1%}
""")
