"""
Token-vs-Boxed KL/JS Divergence Analysis
For each token position in the response, compute the forward KL, reverse KL, and JS
divergence between that position's entropy distribution and the final \\boxed{} region's
entropy distribution. Separate analysis for correct vs incorrect samples.
"""

import json
import re
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ============================================================
# 1. Load data
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
            })

print(f"Total responses with tokens: {len(all_responses)}")
print(f"Correct: {sum(1 for r in all_responses if r['is_correct'])}")
print(f"Incorrect: {sum(1 for r in all_responses if not r['is_correct'])}")

# ============================================================
# 2. Identify \\boxed{} token indices
# ============================================================
def find_boxed_token_indices(tokens):
    text_so_far = ""
    token_positions = []
    for i, tok in enumerate(tokens):
        start = len(text_so_far)
        text_so_far += tok
        end = len(text_so_far)
        token_positions.append((start, end))
    full_text = text_so_far
    all_boxed_indices = set()
    answer_content_indices = set()

    for match in re.finditer(r'\\boxed\{', full_text):
        brace_start = match.end() - 1
        depth = 1
        pos = match.end()
        while pos < len(full_text) and depth > 0:
            if full_text[pos] == '{':
                depth += 1
            elif full_text[pos] == '}':
                depth -= 1
            pos += 1
        brace_end = pos
        boxed_region = (match.start(), brace_end)
        content_region = (match.end(), brace_end - 1)
        for i, (ts, te) in enumerate(token_positions):
            if ts < boxed_region[1] and te > boxed_region[0]:
                all_boxed_indices.add(i)
            if ts < content_region[1] and te > content_region[0]:
                answer_content_indices.add(i)

    return sorted(all_boxed_indices), sorted(answer_content_indices)

# ============================================================
# 3. KL/JS computation utilities
# ============================================================
def entropy_to_hist(samples, bins, eps=1e-10):
    hist, _ = np.histogram(samples, bins=bins, density=True)
    hist = hist + eps
    return hist / hist.sum()

def compute_kl_js(p_samples, q_samples, n_bins=100):
    """Compute KL(P||Q), KL(Q||P), JS(P,Q) from samples."""
    if len(p_samples) < 3 or len(q_samples) < 3:
        return np.nan, np.nan, np.nan

    all_s = np.concatenate([p_samples, q_samples])
    lo, hi = np.min(all_s), np.max(all_s)
    if hi - lo < 1e-6:
        return 0.0, 0.0, 0.0
    bins = np.linspace(lo - 0.01, hi + 0.01, n_bins + 1)

    p = entropy_to_hist(p_samples, bins)
    q = entropy_to_hist(q_samples, bins)
    m = 0.5 * (p + q)

    fwd_kl = float(np.sum(p * np.log(p / q)))
    rev_kl = float(np.sum(q * np.log(q / p)))
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    return fwd_kl, rev_kl, js

# ============================================================
# 4. Collect per-response data: each token's entropy + boxed entropy
# ============================================================
N_POS_BINS = 50  # normalized position bins

# Structures: pos_bin -> list of entropies
correct_pos_ent = defaultdict(list)
incorrect_pos_ent = defaultdict(list)

# Boxed region entropies (the reference distribution)
correct_boxed_ent = []
incorrect_boxed_ent = []

# Offset from boxed start
OFFSET_RANGE = 80
correct_offset_ent = defaultdict(list)
incorrect_offset_ent = defaultdict(list)

# Per-response: individual token entropy vs that response's boxed mean
correct_token_ent_series = []  # list of (normalized_positions, entropies, boxed_mean)
incorrect_token_ent_series = []

n_with_boxed = 0
n_without_boxed = 0

for resp in all_responses:
    ent = resp['entropies']
    n = len(ent)
    boxed_idx, answer_idx = find_boxed_token_indices(resp['tokens'])

    if not boxed_idx:
        n_without_boxed += 1
        continue
    n_with_boxed += 1

    is_correct = resp['is_correct']
    boxed_ent_vals = ent[boxed_idx]
    boxed_set = set(boxed_idx)

    target_pos = correct_pos_ent if is_correct else incorrect_pos_ent
    target_boxed = correct_boxed_ent if is_correct else incorrect_boxed_ent
    target_offset = correct_offset_ent if is_correct else incorrect_offset_ent

    target_boxed.extend(boxed_ent_vals.tolist())

    # Position-normalized
    for i in range(n):
        pos_bin = min(int(i / n * N_POS_BINS), N_POS_BINS - 1)
        target_pos[pos_bin].append(float(ent[i]))

    # Offset from boxed start
    boxed_start = boxed_idx[0]
    for offset in range(-OFFSET_RANGE, OFFSET_RANGE + 1):
        idx = boxed_start + offset
        if 0 <= idx < n:
            target_offset[offset].append(float(ent[idx]))

    # Per-response series
    series_data = {
        'positions': np.arange(n) / n,
        'entropies': ent,
        'boxed_mean': float(np.mean(boxed_ent_vals)),
        'boxed_indices': boxed_idx,
    }
    if is_correct:
        correct_token_ent_series.append(series_data)
    else:
        incorrect_token_ent_series.append(series_data)

print(f"\nWith \\boxed{{}}: {n_with_boxed}, Without: {n_without_boxed}")
print(f"Correct with boxed: {len(correct_token_ent_series)}")
print(f"Incorrect with boxed: {len(incorrect_token_ent_series)}")
print(f"Correct boxed entropy samples: {len(correct_boxed_ent)}")
print(f"Incorrect boxed entropy samples: {len(incorrect_boxed_ent)}")

# ============================================================
# 5. Compute KL/JS: each position bin vs boxed region
# ============================================================
print("\n" + "=" * 80)
print("POSITION-NORMALIZED: Token(pos) vs Boxed Region KL/JS Divergence")
print("=" * 80)

correct_boxed_arr = np.array(correct_boxed_ent)
incorrect_boxed_arr = np.array(incorrect_boxed_ent)

# For correct responses
c_fwd_kl, c_rev_kl, c_js = [], [], []
for pos in range(N_POS_BINS):
    vals = correct_pos_ent.get(pos, [])
    if len(vals) >= 5 and len(correct_boxed_ent) >= 5:
        fkl, rkl, jsd = compute_kl_js(np.array(vals), correct_boxed_arr)
    else:
        fkl, rkl, jsd = np.nan, np.nan, np.nan
    c_fwd_kl.append(fkl)
    c_rev_kl.append(rkl)
    c_js.append(jsd)

# For incorrect responses
ic_fwd_kl, ic_rev_kl, ic_js = [], [], []
for pos in range(N_POS_BINS):
    vals = incorrect_pos_ent.get(pos, [])
    if len(vals) >= 5 and len(incorrect_boxed_ent) >= 5:
        fkl, rkl, jsd = compute_kl_js(np.array(vals), incorrect_boxed_arr)
    else:
        fkl, rkl, jsd = np.nan, np.nan, np.nan
    ic_fwd_kl.append(fkl)
    ic_rev_kl.append(rkl)
    ic_js.append(jsd)

# Print summary
pos_pct = np.arange(N_POS_BINS) / N_POS_BINS * 100
print(f"\n{'Pos%':>6s} | {'C_FwdKL':>8s} {'C_RevKL':>8s} {'C_JS':>8s} | {'IC_FwdKL':>8s} {'IC_RevKL':>8s} {'IC_JS':>8s}")
print("-" * 75)
for i in range(0, N_POS_BINS, 5):
    print(f"{pos_pct[i]:>5.0f}% | {c_fwd_kl[i]:>8.4f} {c_rev_kl[i]:>8.4f} {c_js[i]:>8.4f} | "
          f"{ic_fwd_kl[i]:>8.4f} {ic_rev_kl[i]:>8.4f} {ic_js[i]:>8.4f}")

# ============================================================
# 6. Compute KL/JS: each offset from boxed vs boxed region
# ============================================================
print("\n" + "=" * 80)
print("OFFSET FROM \\boxed{}: Token(offset) vs Boxed Region KL/JS Divergence")
print("=" * 80)

offsets_valid = []
c_off_fkl, c_off_rkl, c_off_js = [], [], []
ic_off_fkl, ic_off_rkl, ic_off_js = [], [], []
c_off_mean, ic_off_mean = [], []

for offset in range(-OFFSET_RANGE, OFFSET_RANGE + 1):
    c_vals = correct_offset_ent.get(offset, [])
    ic_vals = incorrect_offset_ent.get(offset, [])

    if len(c_vals) < 3 or len(ic_vals) < 3:
        continue

    offsets_valid.append(offset)

    fkl, rkl, jsd = compute_kl_js(np.array(c_vals), correct_boxed_arr)
    c_off_fkl.append(fkl)
    c_off_rkl.append(rkl)
    c_off_js.append(jsd)
    c_off_mean.append(np.mean(c_vals))

    fkl, rkl, jsd = compute_kl_js(np.array(ic_vals), incorrect_boxed_arr)
    ic_off_fkl.append(fkl)
    ic_off_rkl.append(rkl)
    ic_off_js.append(jsd)
    ic_off_mean.append(np.mean(ic_vals))

print(f"\nValid offsets: {len(offsets_valid)} (from {min(offsets_valid)} to {max(offsets_valid)})")

# Print key offsets
print(f"\n{'Offset':>7s} | {'C_FwdKL':>8s} {'C_RevKL':>8s} {'C_JS':>8s} {'C_H':>6s} | "
      f"{'IC_FwdKL':>8s} {'IC_RevKL':>8s} {'IC_JS':>8s} {'IC_H':>6s}")
print("-" * 90)
key_offsets = [-50, -40, -30, -20, -10, -5, -3, -1, 0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
for off in key_offsets:
    if off in offsets_valid:
        idx = offsets_valid.index(off)
        print(f"{off:>+7d} | {c_off_fkl[idx]:>8.4f} {c_off_rkl[idx]:>8.4f} {c_off_js[idx]:>8.4f} {c_off_mean[idx]:>6.3f} | "
              f"{ic_off_fkl[idx]:>8.4f} {ic_off_rkl[idx]:>8.4f} {ic_off_js[idx]:>8.4f} {ic_off_mean[idx]:>6.3f}")

# ============================================================
# 7. KL/JS between Correct-token-at-pos and Incorrect-token-at-pos
#    (the divergence between correct vs incorrect at each position)
# ============================================================
print("\n" + "=" * 80)
print("CORRECT vs INCORRECT at each position (divergence between the two groups)")
print("=" * 80)

ci_fwd_kl, ci_rev_kl, ci_js = [], [], []
ci_c_mean, ci_ic_mean = [], []
for pos in range(N_POS_BINS):
    c_vals = np.array(correct_pos_ent.get(pos, []))
    ic_vals = np.array(incorrect_pos_ent.get(pos, []))
    if len(c_vals) >= 5 and len(ic_vals) >= 5:
        fkl, rkl, jsd = compute_kl_js(c_vals, ic_vals)
        ci_fwd_kl.append(fkl)
        ci_rev_kl.append(rkl)
        ci_js.append(jsd)
        ci_c_mean.append(np.mean(c_vals))
        ci_ic_mean.append(np.mean(ic_vals))
    else:
        ci_fwd_kl.append(np.nan)
        ci_rev_kl.append(np.nan)
        ci_js.append(np.nan)
        ci_c_mean.append(np.nan)
        ci_ic_mean.append(np.nan)

# Same for offset
ci_off_fkl, ci_off_rkl, ci_off_js = [], [], []
ci_offsets_valid = []
ci_off_c_mean, ci_off_ic_mean = [], []
for offset in range(-OFFSET_RANGE, OFFSET_RANGE + 1):
    c_vals = np.array(correct_offset_ent.get(offset, []))
    ic_vals = np.array(incorrect_offset_ent.get(offset, []))
    if len(c_vals) >= 3 and len(ic_vals) >= 3:
        fkl, rkl, jsd = compute_kl_js(c_vals, ic_vals)
        ci_offsets_valid.append(offset)
        ci_off_fkl.append(fkl)
        ci_off_rkl.append(rkl)
        ci_off_js.append(jsd)
        ci_off_c_mean.append(np.mean(c_vals))
        ci_off_ic_mean.append(np.mean(ic_vals))

# ============================================================
# 8. Per-response: rolling KL toward boxed (trajectory)
# ============================================================
print("\n" + "=" * 80)
print("PER-RESPONSE ROLLING DIVERGENCE TOWARD \\boxed{}")
print("=" * 80)

WINDOW = 20  # rolling window size

def compute_rolling_kl_to_boxed(series_list, n_bins=50):
    """For each response, compute rolling KL from sliding window to boxed region."""
    all_trajectories_fkl = defaultdict(list)
    all_trajectories_rkl = defaultdict(list)
    all_trajectories_js = defaultdict(list)
    all_trajectories_mean = defaultdict(list)

    for s in series_list:
        ent = s['entropies']
        n = len(ent)
        boxed_ent = ent[s['boxed_indices']]
        if len(boxed_ent) < 2 or n < WINDOW:
            continue

        for start in range(0, n - WINDOW + 1, max(1, n // n_bins)):
            window_ent = ent[start:start + WINDOW]
            pos_norm = (start + WINDOW // 2) / n
            pos_bin = min(int(pos_norm * n_bins), n_bins - 1)

            fkl, rkl, jsd = compute_kl_js(window_ent, boxed_ent, n_bins=30)
            if not np.isnan(fkl):
                all_trajectories_fkl[pos_bin].append(fkl)
                all_trajectories_rkl[pos_bin].append(rkl)
                all_trajectories_js[pos_bin].append(jsd)
                all_trajectories_mean[pos_bin].append(np.mean(window_ent))

    result_fkl, result_rkl, result_js, result_mean = [], [], [], []
    for pos in range(n_bins):
        if all_trajectories_fkl[pos]:
            result_fkl.append(np.mean(all_trajectories_fkl[pos]))
            result_rkl.append(np.mean(all_trajectories_rkl[pos]))
            result_js.append(np.mean(all_trajectories_js[pos]))
            result_mean.append(np.mean(all_trajectories_mean[pos]))
        else:
            result_fkl.append(np.nan)
            result_rkl.append(np.nan)
            result_js.append(np.nan)
            result_mean.append(np.nan)
    return result_fkl, result_rkl, result_js, result_mean

c_roll_fkl, c_roll_rkl, c_roll_js, c_roll_mean = compute_rolling_kl_to_boxed(correct_token_ent_series)
ic_roll_fkl, ic_roll_rkl, ic_roll_js, ic_roll_mean = compute_rolling_kl_to_boxed(incorrect_token_ent_series)

# ============================================================
# 9. PLOTTING
# ============================================================
out_dir = base_dir

# ----- Figure 1: Position-normalized Token vs Boxed KL/JS -----
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('Each Token Position vs Final \\\\boxed{} Region: KL/JS Divergence', fontsize=16, fontweight='bold')

x_pos = np.arange(N_POS_BINS) / N_POS_BINS * 100

# Row 1: Correct responses
ax = axes[0][0]
ax.plot(x_pos, c_fwd_kl, 'b-', linewidth=1.5, label='Forward KL(token||boxed)')
ax.plot(x_pos, c_rev_kl, 'r-', linewidth=1.5, label='Reverse KL(boxed||token)')
ax.fill_between(x_pos, c_fwd_kl, alpha=0.15, color='blue')
ax.fill_between(x_pos, c_rev_kl, alpha=0.15, color='red')
ax.set_title('CORRECT: Token vs Boxed KL', fontsize=12)
ax.set_xlabel('Position in Response (%)')
ax.set_ylabel('KL Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvspan(90, 100, alpha=0.1, color='gold')

ax = axes[0][1]
ax.plot(x_pos, c_js, 'g-', linewidth=2, label='JS(token, boxed)')
ax.fill_between(x_pos, c_js, alpha=0.2, color='green')
ax.set_title('CORRECT: Token vs Boxed JS Divergence', fontsize=12)
ax.set_xlabel('Position in Response (%)')
ax.set_ylabel('JS Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvspan(90, 100, alpha=0.1, color='gold')

ax = axes[0][2]
c_asym = np.array(c_fwd_kl) - np.array(c_rev_kl)
colors = ['steelblue' if a > 0 else 'coral' for a in c_asym]
ax.bar(x_pos, c_asym, width=2, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('CORRECT: KL Asymmetry (Fwd-Rev)', fontsize=12)
ax.set_xlabel('Position in Response (%)')
ax.set_ylabel('Fwd KL - Rev KL')
ax.grid(True, alpha=0.3)

# Row 2: Incorrect responses
ax = axes[1][0]
ax.plot(x_pos, ic_fwd_kl, 'b-', linewidth=1.5, label='Forward KL(token||boxed)')
ax.plot(x_pos, ic_rev_kl, 'r-', linewidth=1.5, label='Reverse KL(boxed||token)')
ax.fill_between(x_pos, ic_fwd_kl, alpha=0.15, color='blue')
ax.fill_between(x_pos, ic_rev_kl, alpha=0.15, color='red')
ax.set_title('INCORRECT: Token vs Boxed KL', fontsize=12)
ax.set_xlabel('Position in Response (%)')
ax.set_ylabel('KL Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvspan(90, 100, alpha=0.1, color='gold')

ax = axes[1][1]
ax.plot(x_pos, ic_js, 'g-', linewidth=2, label='JS(token, boxed)')
ax.fill_between(x_pos, ic_js, alpha=0.2, color='green')
ax.set_title('INCORRECT: Token vs Boxed JS Divergence', fontsize=12)
ax.set_xlabel('Position in Response (%)')
ax.set_ylabel('JS Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvspan(90, 100, alpha=0.1, color='gold')

ax = axes[1][2]
ic_asym = np.array(ic_fwd_kl) - np.array(ic_rev_kl)
colors = ['steelblue' if a > 0 else 'coral' for a in ic_asym]
ax.bar(x_pos, ic_asym, width=2, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('INCORRECT: KL Asymmetry (Fwd-Rev)', fontsize=12)
ax.set_xlabel('Position in Response (%)')
ax.set_ylabel('Fwd KL - Rev KL')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/token_vs_boxed_position.png", dpi=150, bbox_inches='tight')
print("Saved: token_vs_boxed_position.png")

# ----- Figure 2: Offset from Boxed -----
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('Token Offset from \\\\boxed{} Start: KL/JS to Boxed Region', fontsize=16, fontweight='bold')

# Correct
ax = axes[0][0]
ax.plot(offsets_valid, c_off_fkl, 'b-o', markersize=2, linewidth=1, label='Fwd KL(token||boxed)')
ax.plot(offsets_valid, c_off_rkl, 'r-s', markersize=2, linewidth=1, label='Rev KL(boxed||token)')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('CORRECT: Offset KL', fontsize=12)
ax.set_xlabel('Token offset from \\\\boxed{ start')
ax.set_ylabel('KL Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0][1]
ax.plot(offsets_valid, c_off_js, 'g-o', markersize=2, linewidth=1.5, label='JS(token, boxed)')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('CORRECT: Offset JS', fontsize=12)
ax.set_xlabel('Token offset from \\\\boxed{ start')
ax.set_ylabel('JS Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0][2]
ax.plot(offsets_valid, c_off_mean, 'g-', linewidth=1.5, label='Mean Entropy')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('CORRECT: Mean Entropy by Offset', fontsize=12)
ax.set_xlabel('Token offset from \\\\boxed{ start')
ax.set_ylabel('Mean Token Entropy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Incorrect
ax = axes[1][0]
ax.plot(offsets_valid, ic_off_fkl, 'b-o', markersize=2, linewidth=1, label='Fwd KL(token||boxed)')
ax.plot(offsets_valid, ic_off_rkl, 'r-s', markersize=2, linewidth=1, label='Rev KL(boxed||token)')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('INCORRECT: Offset KL', fontsize=12)
ax.set_xlabel('Token offset from \\\\boxed{ start')
ax.set_ylabel('KL Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1][1]
ax.plot(offsets_valid, ic_off_js, 'g-o', markersize=2, linewidth=1.5, label='JS(token, boxed)')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('INCORRECT: Offset JS', fontsize=12)
ax.set_xlabel('Token offset from \\\\boxed{ start')
ax.set_ylabel('JS Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1][2]
ax.plot(offsets_valid, ic_off_mean, 'r-', linewidth=1.5, label='Mean Entropy')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('INCORRECT: Mean Entropy by Offset', fontsize=12)
ax.set_xlabel('Token offset from \\\\boxed{ start')
ax.set_ylabel('Mean Token Entropy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/token_vs_boxed_offset.png", dpi=150, bbox_inches='tight')
print("Saved: token_vs_boxed_offset.png")

# ----- Figure 3: Correct vs Incorrect Divergence at Each Position -----
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Correct vs Incorrect Divergence at Each Token Position', fontsize=16, fontweight='bold')

# Position-normalized
ax = axes[0][0]
ax.plot(x_pos, ci_fwd_kl, 'b-', linewidth=1.5, label='Fwd KL(C||IC)')
ax.plot(x_pos, ci_rev_kl, 'r-', linewidth=1.5, label='Rev KL(IC||C)')
ax.plot(x_pos, ci_js, 'g--', linewidth=1.5, label='JS(C, IC)')
ax.fill_between(x_pos, ci_rev_kl, alpha=0.1, color='red')
ax.set_title('By Position: C vs IC Divergence', fontsize=12)
ax.set_xlabel('Position in Response (%)')
ax.set_ylabel('Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvspan(90, 100, alpha=0.1, color='gold')

ax = axes[0][1]
ax.plot(x_pos, ci_c_mean, 'g-', linewidth=1.5, label='Correct Mean H')
ax.plot(x_pos, ci_ic_mean, 'r-', linewidth=1.5, label='Incorrect Mean H')
ax.fill_between(x_pos, ci_c_mean, ci_ic_mean, alpha=0.15, color='gray')
ax.set_title('By Position: Mean Entropy', fontsize=12)
ax.set_xlabel('Position in Response (%)')
ax.set_ylabel('Mean Token Entropy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Offset from boxed
ax = axes[1][0]
ax.plot(ci_offsets_valid, ci_off_fkl, 'b-o', markersize=2, linewidth=1, label='Fwd KL(C||IC)')
ax.plot(ci_offsets_valid, ci_off_rkl, 'r-s', markersize=2, linewidth=1, label='Rev KL(IC||C)')
ax.plot(ci_offsets_valid, ci_off_js, 'g--', markersize=2, linewidth=1, label='JS(C, IC)')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('By Offset from Boxed: C vs IC Divergence', fontsize=12)
ax.set_xlabel('Token offset from \\\\boxed{ start')
ax.set_ylabel('Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1][1]
ax.plot(ci_offsets_valid, ci_off_c_mean, 'g-o', markersize=2, linewidth=1, label='Correct Mean H')
ax.plot(ci_offsets_valid, ci_off_ic_mean, 'r-s', markersize=2, linewidth=1, label='Incorrect Mean H')
ax.fill_between(ci_offsets_valid, ci_off_c_mean, ci_off_ic_mean, alpha=0.15, color='gray')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('By Offset from Boxed: Mean Entropy', fontsize=12)
ax.set_xlabel('Token offset from \\\\boxed{ start')
ax.set_ylabel('Mean Token Entropy')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/correct_vs_incorrect_by_position.png", dpi=150, bbox_inches='tight')
print("Saved: correct_vs_incorrect_by_position.png")

# ----- Figure 4: Rolling window KL trajectory toward boxed -----
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Rolling Window (w={WINDOW}) KL/JS Trajectory Toward \\\\boxed{{}} Region', fontsize=16, fontweight='bold')

roll_x = np.arange(50) / 50 * 100

ax = axes[0][0]
ax.plot(roll_x, c_roll_fkl, 'b-', linewidth=1.5, label='Fwd KL(window||boxed)')
ax.plot(roll_x, c_roll_rkl, 'r-', linewidth=1.5, label='Rev KL(boxed||window)')
ax.set_title('CORRECT: Rolling KL to Boxed', fontsize=12)
ax.set_xlabel('Position (%)')
ax.set_ylabel('KL Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0][1]
ax.plot(roll_x, c_roll_js, 'g-', linewidth=2, label='JS(window, boxed)')
ax2 = ax.twinx()
ax2.plot(roll_x, c_roll_mean, 'k--', linewidth=1, alpha=0.5, label='Mean H')
ax.set_title('CORRECT: Rolling JS + Mean H', fontsize=12)
ax.set_xlabel('Position (%)')
ax.set_ylabel('JS Divergence')
ax2.set_ylabel('Mean Entropy', color='gray')
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1][0]
ax.plot(roll_x, ic_roll_fkl, 'b-', linewidth=1.5, label='Fwd KL(window||boxed)')
ax.plot(roll_x, ic_roll_rkl, 'r-', linewidth=1.5, label='Rev KL(boxed||window)')
ax.set_title('INCORRECT: Rolling KL to Boxed', fontsize=12)
ax.set_xlabel('Position (%)')
ax.set_ylabel('KL Divergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1][1]
ax.plot(roll_x, ic_roll_js, 'g-', linewidth=2, label='JS(window, boxed)')
ax2 = ax.twinx()
ax2.plot(roll_x, ic_roll_mean, 'k--', linewidth=1, alpha=0.5, label='Mean H')
ax.set_title('INCORRECT: Rolling JS + Mean H', fontsize=12)
ax.set_xlabel('Position (%)')
ax.set_ylabel('JS Divergence')
ax2.set_ylabel('Mean Entropy', color='gray')
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/rolling_kl_to_boxed.png", dpi=150, bbox_inches='tight')
print("Saved: rolling_kl_to_boxed.png")

# ----- Figure 5: Side-by-side comparison -----
fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle('Correct vs Incorrect: Token-to-Boxed Divergence Comparison', fontsize=16, fontweight='bold')

# Forward KL
ax = axes[0][0]
ax.plot(x_pos, c_fwd_kl, 'g-', linewidth=2, label='Correct')
ax.plot(x_pos, ic_fwd_kl, 'r-', linewidth=2, label='Incorrect')
ax.fill_between(x_pos, c_fwd_kl, ic_fwd_kl, alpha=0.15, color='gray')
ax.set_title('Forward KL(token||boxed) by Position', fontsize=12)
ax.set_xlabel('Position (%)')
ax.set_ylabel('Forward KL')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0][1]
idx_c = [offsets_valid.index(o) for o in offsets_valid]
ax.plot(offsets_valid, [c_off_fkl[i] for i in idx_c], 'g-o', markersize=2, linewidth=1, label='Correct')
ax.plot(offsets_valid, [ic_off_fkl[i] for i in idx_c], 'r-s', markersize=2, linewidth=1, label='Incorrect')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('Forward KL(token||boxed) by Offset', fontsize=12)
ax.set_xlabel('Offset from \\\\boxed{')
ax.set_ylabel('Forward KL')
ax.legend()
ax.grid(True, alpha=0.3)

# Reverse KL
ax = axes[1][0]
ax.plot(x_pos, c_rev_kl, 'g-', linewidth=2, label='Correct')
ax.plot(x_pos, ic_rev_kl, 'r-', linewidth=2, label='Incorrect')
ax.fill_between(x_pos, c_rev_kl, ic_rev_kl, alpha=0.15, color='gray')
ax.set_title('Reverse KL(boxed||token) by Position', fontsize=12)
ax.set_xlabel('Position (%)')
ax.set_ylabel('Reverse KL')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1][1]
ax.plot(offsets_valid, [c_off_rkl[i] for i in idx_c], 'g-o', markersize=2, linewidth=1, label='Correct')
ax.plot(offsets_valid, [ic_off_rkl[i] for i in idx_c], 'r-s', markersize=2, linewidth=1, label='Incorrect')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('Reverse KL(boxed||token) by Offset', fontsize=12)
ax.set_xlabel('Offset from \\\\boxed{')
ax.set_ylabel('Reverse KL')
ax.legend()
ax.grid(True, alpha=0.3)

# JS Divergence
ax = axes[2][0]
ax.plot(x_pos, c_js, 'g-', linewidth=2, label='Correct')
ax.plot(x_pos, ic_js, 'r-', linewidth=2, label='Incorrect')
ax.fill_between(x_pos, c_js, ic_js, alpha=0.15, color='gray')
ax.set_title('JS(token, boxed) by Position', fontsize=12)
ax.set_xlabel('Position (%)')
ax.set_ylabel('JS Divergence')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2][1]
ax.plot(offsets_valid, [c_off_js[i] for i in idx_c], 'g-o', markersize=2, linewidth=1, label='Correct')
ax.plot(offsets_valid, [ic_off_js[i] for i in idx_c], 'r-s', markersize=2, linewidth=1, label='Incorrect')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_title('JS(token, boxed) by Offset', fontsize=12)
ax.set_xlabel('Offset from \\\\boxed{')
ax.set_ylabel('JS Divergence')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/token_boxed_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: token_boxed_comparison.png")

# ============================================================
# 10. Numerical Summary
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY OF KEY PATTERNS")
print("=" * 80)

# Divide response into phases
phases = [
    ("Early (0-20%)", 0, int(N_POS_BINS * 0.2)),
    ("Middle (20-60%)", int(N_POS_BINS * 0.2), int(N_POS_BINS * 0.6)),
    ("Late (60-90%)", int(N_POS_BINS * 0.6), int(N_POS_BINS * 0.9)),
    ("Final (90-100%)", int(N_POS_BINS * 0.9), N_POS_BINS),
]

print("\n--- Phase-averaged Token-to-Boxed Divergence ---")
print(f"{'Phase':<20s} | {'C_FwdKL':>8s} {'C_RevKL':>8s} {'C_JS':>8s} | {'IC_FwdKL':>8s} {'IC_RevKL':>8s} {'IC_JS':>8s} | {'C_FwdKL/IC':>10s}")
print("-" * 100)
for name, s, e in phases:
    c_f = np.nanmean(c_fwd_kl[s:e])
    c_r = np.nanmean(c_rev_kl[s:e])
    c_j = np.nanmean(c_js[s:e])
    ic_f = np.nanmean(ic_fwd_kl[s:e])
    ic_r = np.nanmean(ic_rev_kl[s:e])
    ic_j = np.nanmean(ic_js[s:e])
    ratio = c_f / ic_f if ic_f > 0 else float('inf')
    print(f"{name:<20s} | {c_f:>8.4f} {c_r:>8.4f} {c_j:>8.4f} | {ic_f:>8.4f} {ic_r:>8.4f} {ic_j:>8.4f} | {ratio:>10.3f}")

print("\n--- Phase-averaged Correct vs Incorrect Divergence ---")
print(f"{'Phase':<20s} | {'FwdKL(C||IC)':>12s} {'RevKL(IC||C)':>12s} {'JS(C,IC)':>10s} | {'Rev/Fwd':>8s}")
print("-" * 75)
for name, s, e in phases:
    f = np.nanmean(ci_fwd_kl[s:e])
    r = np.nanmean(ci_rev_kl[s:e])
    j = np.nanmean(ci_js[s:e])
    rat = r / f if f > 0 else float('inf')
    print(f"{name:<20s} | {f:>12.4f} {r:>12.4f} {j:>10.4f} | {rat:>8.2f}")

# Convergence analysis
print("\n--- Convergence Pattern ---")
early_c = np.nanmean(c_js[:int(N_POS_BINS*0.2)])
late_c = np.nanmean(c_js[int(N_POS_BINS*0.8):])
early_ic = np.nanmean(ic_js[:int(N_POS_BINS*0.2)])
late_ic = np.nanmean(ic_js[int(N_POS_BINS*0.8):])
print(f"Correct:   JS(token,boxed) drops from {early_c:.4f} (early) to {late_c:.4f} (late) = {(early_c-late_c)/early_c*100:.1f}% reduction")
print(f"Incorrect: JS(token,boxed) drops from {early_ic:.4f} (early) to {late_ic:.4f} (late) = {(early_ic-late_ic)/early_ic*100:.1f}% reduction")
print(f"Convergence ratio (correct/incorrect): {((early_c-late_c)/early_c) / ((early_ic-late_ic)/early_ic + 1e-10):.2f}x faster")

print(f"\nAll plots saved to: {out_dir}")
