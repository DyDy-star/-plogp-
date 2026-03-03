"""
Per-response analysis: at each token position (sliding window),
compute Forward KL and Reverse KL to that response's own \\boxed{} region.
Reveal the trajectory pattern for correct vs incorrect responses.
"""

import json, re, numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
            tokens, entropies = [], []
            for step in resp['entropy_analysis']['steps']:
                tokens.extend(step['tokens'])
                entropies.extend(step['token_entropies'])
            if not tokens:
                continue
            all_responses.append({
                'tag': tag, 'prompt_id': result['id'],
                'is_correct': resp['is_correct'],
                'tokens': tokens,
                'entropies': np.array(entropies, dtype=np.float32),
            })

def find_boxed_indices(tokens):
    text = ""
    pos = []
    for i, t in enumerate(tokens):
        s = len(text); text += t; pos.append((s, len(text)))
    idx = set()
    for m in re.finditer(r'\\boxed\{', text):
        depth, p = 1, m.end()
        while p < len(text) and depth > 0:
            if text[p] == '{': depth += 1
            elif text[p] == '}': depth -= 1
            p += 1
        for i, (ts, te) in enumerate(pos):
            if ts < p and te > m.start():
                idx.add(i)
    return sorted(idx)

# ============================================================
# 2. Per-response sliding window KL to boxed
# ============================================================
WINDOW = 15
N_NORM_BINS = 40

def kl_js_from_samples(p, q, n_bins=50, eps=1e-10):
    if len(p) < 3 or len(q) < 3:
        return np.nan, np.nan, np.nan
    all_s = np.concatenate([p, q])
    lo, hi = all_s.min(), all_s.max()
    if hi - lo < 1e-6:
        return 0.0, 0.0, 0.0
    bins = np.linspace(lo - 0.01, hi + 0.01, n_bins + 1)
    ph, _ = np.histogram(p, bins=bins, density=True)
    qh, _ = np.histogram(q, bins=bins, density=True)
    ph = (ph + eps); ph /= ph.sum()
    qh = (qh + eps); qh /= qh.sum()
    fkl = float(np.sum(ph * np.log(ph / qh)))
    rkl = float(np.sum(qh * np.log(qh / ph)))
    mh = 0.5 * (ph + qh)
    js = float(0.5 * np.sum(ph * np.log(ph / mh)) + 0.5 * np.sum(qh * np.log(qh / mh)))
    return fkl, rkl, js

# Collect per-response normalized trajectories
correct_trajs = {'fkl': defaultdict(list), 'rkl': defaultdict(list),
                 'js': defaultdict(list), 'asym': defaultdict(list),
                 'mean_h': defaultdict(list), 'ratio': defaultdict(list)}
incorrect_trajs = {'fkl': defaultdict(list), 'rkl': defaultdict(list),
                   'js': defaultdict(list), 'asym': defaultdict(list),
                   'mean_h': defaultdict(list), 'ratio': defaultdict(list)}

# Also collect individual trajectories for plotting examples
correct_examples = []
incorrect_examples = []

n_processed = 0
for resp in all_responses:
    ent = resp['entropies']
    n = len(ent)
    boxed_idx = find_boxed_indices(resp['tokens'])
    if not boxed_idx or n < WINDOW + len(boxed_idx):
        continue
    n_processed += 1

    boxed_ent = ent[boxed_idx]
    target = correct_trajs if resp['is_correct'] else incorrect_trajs

    resp_fkl, resp_rkl, resp_js, resp_asym, resp_mh, resp_ratio = [], [], [], [], [], []
    resp_pos = []

    for start in range(0, n - WINDOW + 1, max(1, (n - WINDOW) // N_NORM_BINS)):
        window_ent = ent[start:start + WINDOW]
        center = (start + WINDOW / 2) / n
        pos_bin = min(int(center * N_NORM_BINS), N_NORM_BINS - 1)

        fkl, rkl, js = kl_js_from_samples(window_ent, boxed_ent, n_bins=30)
        if np.isnan(fkl):
            continue

        asym = fkl - rkl
        ratio = fkl / (rkl + 0.01)
        mh = float(np.mean(window_ent))

        target['fkl'][pos_bin].append(fkl)
        target['rkl'][pos_bin].append(rkl)
        target['js'][pos_bin].append(js)
        target['asym'][pos_bin].append(asym)
        target['mean_h'][pos_bin].append(mh)
        target['ratio'][pos_bin].append(ratio)

        resp_fkl.append(fkl)
        resp_rkl.append(rkl)
        resp_js.append(js)
        resp_asym.append(asym)
        resp_mh.append(mh)
        resp_ratio.append(ratio)
        resp_pos.append(center)

    example = {
        'pos': resp_pos, 'fkl': resp_fkl, 'rkl': resp_rkl,
        'js': resp_js, 'asym': resp_asym, 'mean_h': resp_mh,
        'ratio': resp_ratio, 'prompt_id': resp['prompt_id'], 'tag': resp['tag'],
    }
    if resp['is_correct'] and len(correct_examples) < 15:
        correct_examples.append(example)
    elif not resp['is_correct'] and len(incorrect_examples) < 15:
        incorrect_examples.append(example)

print(f"Processed {n_processed} responses with boxed")
print(f"Correct examples: {len(correct_examples)}, Incorrect: {len(incorrect_examples)}")

# ============================================================
# 3. Compute averaged trajectories
# ============================================================
x_pos = np.arange(N_NORM_BINS) / N_NORM_BINS * 100

def avg_traj(traj_dict, key):
    return [np.mean(traj_dict[key].get(i, [np.nan])) for i in range(N_NORM_BINS)]

def std_traj(traj_dict, key):
    vals = [traj_dict[key].get(i, []) for i in range(N_NORM_BINS)]
    return [np.std(v) if len(v) > 1 else 0 for v in vals]

def median_traj(traj_dict, key):
    return [np.median(traj_dict[key].get(i, [np.nan])) for i in range(N_NORM_BINS)]

# ============================================================
# 4. Print numerical summary
# ============================================================
print("\n" + "=" * 90)
print("PER-RESPONSE TOKEN→BOXED KL TRAJECTORY: AVERAGED ACROSS RESPONSES")
print("=" * 90)

phases = [
    ("Early 0-25%",  0, N_NORM_BINS // 4),
    ("Mid 25-50%",   N_NORM_BINS // 4, N_NORM_BINS // 2),
    ("Late 50-75%",  N_NORM_BINS // 2, 3 * N_NORM_BINS // 4),
    ("Final 75-100%", 3 * N_NORM_BINS // 4, N_NORM_BINS),
]

for metric_name, metric_key in [
    ("Forward KL(window||boxed)", "fkl"),
    ("Reverse KL(boxed||window)", "rkl"),
    ("JS(window, boxed)", "js"),
    ("KL Asymmetry (Fwd-Rev)", "asym"),
    ("FwdKL / (RevKL+0.01)", "ratio"),
    ("Window Mean Entropy", "mean_h"),
]:
    c_avg = avg_traj(correct_trajs, metric_key)
    ic_avg = avg_traj(incorrect_trajs, metric_key)
    print(f"\n--- {metric_name} ---")
    print(f"{'Phase':<16s} | {'Correct':>10s} {'Incorrect':>10s} {'Delta':>10s} {'C/IC Ratio':>10s}")
    print("-" * 65)
    for name, s, e in phases:
        cm = np.nanmean(c_avg[s:e])
        im = np.nanmean(ic_avg[s:e])
        delta = cm - im
        ratio = cm / im if abs(im) > 1e-6 else float('inf')
        print(f"{name:<16s} | {cm:>10.4f} {im:>10.4f} {delta:>+10.4f} {ratio:>10.3f}")

    # Convergence: early-to-final drop
    c_early = np.nanmean(c_avg[:N_NORM_BINS // 4])
    c_final = np.nanmean(c_avg[3 * N_NORM_BINS // 4:])
    ic_early = np.nanmean(ic_avg[:N_NORM_BINS // 4])
    ic_final = np.nanmean(ic_avg[3 * N_NORM_BINS // 4:])
    c_drop = (c_early - c_final) / (c_early + 1e-6) * 100
    ic_drop = (ic_early - ic_final) / (ic_early + 1e-6) * 100
    print(f"  Convergence:  Correct drops {c_drop:+.1f}%, Incorrect drops {ic_drop:+.1f}%")

# ============================================================
# 5. Plotting
# ============================================================

# ----- Figure 1: Averaged trajectories with confidence bands -----
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('Per-Response Sliding Window KL to Own \\\\boxed{}: Averaged Trajectories',
             fontsize=16, fontweight='bold')

for idx, (title, key, ylabel) in enumerate([
    ("Forward KL(window||boxed)", "fkl", "Forward KL"),
    ("Reverse KL(boxed||window)", "rkl", "Reverse KL"),
    ("JS(window, boxed)", "js", "JS Divergence"),
    ("KL Asymmetry (Fwd - Rev)", "asym", "Fwd KL - Rev KL"),
    ("FwdKL / (RevKL + 0.01)", "ratio", "KL Ratio"),
    ("Window Mean Entropy", "mean_h", "Mean Entropy"),
]):
    ax = axes[idx // 3][idx % 3]
    c_avg = avg_traj(correct_trajs, key)
    c_std = std_traj(correct_trajs, key)
    ic_avg = avg_traj(incorrect_trajs, key)
    ic_std = std_traj(incorrect_trajs, key)

    c_avg, c_std = np.array(c_avg), np.array(c_std)
    ic_avg, ic_std = np.array(ic_avg), np.array(ic_std)

    ax.plot(x_pos, c_avg, 'g-', linewidth=2.5, label='Correct (mean)', zorder=3)
    ax.fill_between(x_pos, c_avg - 0.5 * c_std, c_avg + 0.5 * c_std,
                    alpha=0.15, color='green')
    ax.plot(x_pos, ic_avg, 'r-', linewidth=2.5, label='Incorrect (mean)', zorder=3)
    ax.fill_between(x_pos, ic_avg - 0.5 * ic_std, ic_avg + 0.5 * ic_std,
                    alpha=0.15, color='red')

    c_med = median_traj(correct_trajs, key)
    ic_med = median_traj(incorrect_trajs, key)
    ax.plot(x_pos, c_med, 'g--', linewidth=1, alpha=0.7, label='Correct (median)')
    ax.plot(x_pos, ic_med, 'r--', linewidth=1, alpha=0.7, label='Incorrect (median)')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Position in Response (%)')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvspan(85, 100, alpha=0.08, color='gold')

plt.tight_layout()
plt.savefig(f"{base_dir}/per_response_kl_trajectories.png", dpi=150, bbox_inches='tight')
print("\nSaved: per_response_kl_trajectories.png")

# ----- Figure 2: Individual response trajectories -----
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('Individual Response FwdKL/RevKL Trajectories to Own \\\\boxed{}',
             fontsize=16, fontweight='bold')

# Correct examples: Forward KL
ax = axes[0][0]
for ex in correct_examples:
    ax.plot(np.array(ex['pos']) * 100, ex['fkl'], '-', alpha=0.5, linewidth=1)
ax.set_title('CORRECT: Forward KL(window||boxed)', fontsize=11)
ax.set_xlabel('Position (%)'); ax.set_ylabel('Forward KL')
ax.grid(True, alpha=0.3)

# Correct: Reverse KL
ax = axes[0][1]
for ex in correct_examples:
    ax.plot(np.array(ex['pos']) * 100, ex['rkl'], '-', alpha=0.5, linewidth=1)
ax.set_title('CORRECT: Reverse KL(boxed||window)', fontsize=11)
ax.set_xlabel('Position (%)'); ax.set_ylabel('Reverse KL')
ax.grid(True, alpha=0.3)

# Correct: Asymmetry
ax = axes[0][2]
for ex in correct_examples:
    ax.plot(np.array(ex['pos']) * 100, ex['asym'], '-', alpha=0.5, linewidth=1)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('CORRECT: KL Asymmetry (Fwd-Rev)', fontsize=11)
ax.set_xlabel('Position (%)'); ax.set_ylabel('Fwd - Rev')
ax.grid(True, alpha=0.3)

# Incorrect examples
ax = axes[1][0]
for ex in incorrect_examples:
    ax.plot(np.array(ex['pos']) * 100, ex['fkl'], '-', alpha=0.3, linewidth=1)
ax.set_title('INCORRECT: Forward KL(window||boxed)', fontsize=11)
ax.set_xlabel('Position (%)'); ax.set_ylabel('Forward KL')
ax.grid(True, alpha=0.3)

ax = axes[1][1]
for ex in incorrect_examples:
    ax.plot(np.array(ex['pos']) * 100, ex['rkl'], '-', alpha=0.3, linewidth=1)
ax.set_title('INCORRECT: Reverse KL(boxed||window)', fontsize=11)
ax.set_xlabel('Position (%)'); ax.set_ylabel('Reverse KL')
ax.grid(True, alpha=0.3)

ax = axes[1][2]
for ex in incorrect_examples:
    ax.plot(np.array(ex['pos']) * 100, ex['asym'], '-', alpha=0.3, linewidth=1)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('INCORRECT: KL Asymmetry (Fwd-Rev)', fontsize=11)
ax.set_xlabel('Position (%)'); ax.set_ylabel('Fwd - Rev')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{base_dir}/per_response_individual_trajectories.png", dpi=150, bbox_inches='tight')
print("Saved: per_response_individual_trajectories.png")

# ----- Figure 3: Delta (Correct - Incorrect) trajectory -----
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.suptitle('Correct - Incorrect Difference in Per-Response KL Trajectories',
             fontsize=14, fontweight='bold')

for idx, (title, key) in enumerate([
    ("Forward KL Difference", "fkl"),
    ("Reverse KL Difference", "rkl"),
    ("KL Ratio Difference", "ratio"),
]):
    ax = axes[idx]
    c_avg = np.array(avg_traj(correct_trajs, key))
    ic_avg = np.array(avg_traj(incorrect_trajs, key))
    delta = c_avg - ic_avg
    colors = ['green' if d < 0 else 'red' for d in delta]
    ax.bar(x_pos, delta, width=2.5, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Position (%)')
    ax.set_ylabel('Correct - Incorrect')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{base_dir}/per_response_kl_delta.png", dpi=150, bbox_inches='tight')
print("Saved: per_response_kl_delta.png")

# ----- Figure 4: FwdKL vs RevKL scatter (phase space) -----
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('FwdKL vs RevKL Phase Space at Different Response Positions',
             fontsize=14, fontweight='bold')

for idx, (name, s, e) in enumerate(phases):
    ax = axes[idx]
    c_fkl = [v for i in range(s, e) for v in correct_trajs['fkl'].get(i, [])]
    c_rkl = [v for i in range(s, e) for v in correct_trajs['rkl'].get(i, [])]
    ic_fkl = [v for i in range(s, e) for v in incorrect_trajs['fkl'].get(i, [])]
    ic_rkl = [v for i in range(s, e) for v in incorrect_trajs['rkl'].get(i, [])]

    ax.scatter(ic_fkl, ic_rkl, s=8, alpha=0.15, color='red', label='Incorrect')
    ax.scatter(c_fkl, c_rkl, s=25, alpha=0.5, color='green', label='Correct')

    lim = max(np.percentile(c_fkl + ic_fkl, 95), np.percentile(c_rkl + ic_rkl, 95)) if c_fkl and ic_fkl else 10
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='Fwd=Rev')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Forward KL')
    ax.set_ylabel('Reverse KL')
    ax.set_title(name, fontsize=12)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f"{base_dir}/per_response_fkl_vs_rkl_phase.png", dpi=150, bbox_inches='tight')
print("Saved: per_response_fkl_vs_rkl_phase.png")

# ============================================================
# 6. Key pattern summary
# ============================================================
print("\n" + "=" * 90)
print("KEY PATTERNS SUMMARY")
print("=" * 90)

c_fkl = avg_traj(correct_trajs, 'fkl')
ic_fkl = avg_traj(incorrect_trajs, 'fkl')
c_rkl = avg_traj(correct_trajs, 'rkl')
ic_rkl = avg_traj(incorrect_trajs, 'rkl')
c_asym = avg_traj(correct_trajs, 'asym')
ic_asym = avg_traj(incorrect_trajs, 'asym')
c_ratio = avg_traj(correct_trajs, 'ratio')
ic_ratio = avg_traj(incorrect_trajs, 'ratio')

print(f"""
1. FORWARD KL(window || boxed) - Reasoning divergence from answer:
   Correct:   Early={np.nanmean(c_fkl[:10]):.3f} → Final={np.nanmean(c_fkl[-5:]):.3f}  (drop {(np.nanmean(c_fkl[:10])-np.nanmean(c_fkl[-5:]))/np.nanmean(c_fkl[:10])*100:.0f}%)
   Incorrect: Early={np.nanmean(ic_fkl[:10]):.3f} → Final={np.nanmean(ic_fkl[-5:]):.3f}  (drop {(np.nanmean(ic_fkl[:10])-np.nanmean(ic_fkl[-5:]))/np.nanmean(ic_fkl[:10])*100:.0f}%)
   => Correct drops MORE: correct reasoning converges faster to own boxed.

2. REVERSE KL(boxed || window) - Answer pattern coverage over reasoning:
   Correct:   Early={np.nanmean(c_rkl[:10]):.3f} → Final={np.nanmean(c_rkl[-5:]):.3f}
   Incorrect: Early={np.nanmean(ic_rkl[:10]):.3f} → Final={np.nanmean(ic_rkl[-5:]):.3f}
   => Correct has LOWER RevKL throughout (boxed can "explain" the reasoning better).

3. KL ASYMMETRY (Fwd - Rev):
   Correct:   All-phase average = {np.nanmean(c_asym):.3f}  (Fwd {'>' if np.nanmean(c_asym) > 0 else '<'} Rev)
   Incorrect: All-phase average = {np.nanmean(ic_asym):.3f}  (Fwd {'>' if np.nanmean(ic_asym) > 0 else '<'} Rev)
   Difference: Correct asym - Incorrect asym = {np.nanmean(c_asym) - np.nanmean(ic_asym):+.3f}

4. KL RATIO (Fwd / Rev):
   Correct:   Early={np.nanmean(c_ratio[:10]):.3f} → Final={np.nanmean(c_ratio[-5:]):.3f}
   Incorrect: Early={np.nanmean(ic_ratio[:10]):.3f} → Final={np.nanmean(ic_ratio[-5:]):.3f}

5. CORRECT vs INCORRECT - which has higher per-response FwdKL?
   Correct overall mean FwdKL:   {np.nanmean(c_fkl):.3f}
   Incorrect overall mean FwdKL: {np.nanmean(ic_fkl):.3f}
   => {'Correct' if np.nanmean(c_fkl) < np.nanmean(ic_fkl) else 'Incorrect'} has LOWER FwdKL 
      (its reasoning is CLOSER to its own boxed region in distribution shape)
""")
