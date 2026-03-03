"""
Token-level Forward & Reverse KL Divergence Analysis
Comparing correct vs incorrect samples, with focus on \\boxed{} region tokens.
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
plt.rcParams['axes.unicode_minus'] = False

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
            all_responses.append({
                'file_tag': tag,
                'prompt_id': result['id'],
                'ground_truth': result['ground_truth'],
                'response_id': resp['response_id'],
                'is_correct': resp['is_correct'],
                'extracted_answer': resp['extracted_answer'],
                'response_text': resp['response'],
                'entropy_analysis': resp['entropy_analysis'],
            })

correct_responses = [r for r in all_responses if r['is_correct']]
incorrect_responses = [r for r in all_responses if not r['is_correct']]
print(f"Total responses: {len(all_responses)}")
print(f"Correct: {len(correct_responses)}, Incorrect: {len(incorrect_responses)}")

# ============================================================
# 2. Extract token-level entropies and identify \\boxed{} tokens
# ============================================================
def extract_token_entropies(resp):
    """Flatten all steps' token entropies into a single sequence."""
    tokens = []
    entropies = []
    for step in resp['entropy_analysis']['steps']:
        tokens.extend(step['tokens'])
        entropies.extend(step['token_entropies'])
    return tokens, entropies

def find_boxed_token_indices(tokens):
    """
    Find indices of tokens inside \\boxed{...}, including the boxed keyword,
    braces, and the answer content.
    Returns: (all_boxed_indices, answer_digit_indices, boxed_keyword_indices)
    """
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
    boxed_keyword_indices = set()

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
            if ts < boxed_region[0] + len('\\boxed') and te > boxed_region[0]:
                boxed_keyword_indices.add(i)
            if ts < content_region[1] and te > content_region[0]:
                answer_content_indices.add(i)

    return sorted(all_boxed_indices), sorted(answer_content_indices), sorted(boxed_keyword_indices)

# ============================================================
# 3. Gather data
# ============================================================
correct_all_entropies = []
incorrect_all_entropies = []

correct_boxed_entropies = []
incorrect_boxed_entropies = []

correct_answer_entropies = []
incorrect_answer_entropies = []

correct_nonboxed_entropies = []
incorrect_nonboxed_entropies = []

correct_boxed_keyword_entropies = []
incorrect_boxed_keyword_entropies = []

# Position-normalized entropies (normalize to [0, 1] range)
N_BINS = 100
correct_position_entropies = defaultdict(list)
incorrect_position_entropies = defaultdict(list)

# Boxed region relative position entropies
correct_boxed_position = defaultdict(list)
incorrect_boxed_position = defaultdict(list)

# Last N tokens before and after boxed
CONTEXT_WINDOW = 30
correct_boxed_context = defaultdict(list)  # key: offset from boxed start
incorrect_boxed_context = defaultdict(list)

for resp in all_responses:
    tokens, entropies = extract_token_entropies(resp)
    if len(tokens) == 0:
        continue

    all_boxed_idx, answer_idx, keyword_idx = find_boxed_token_indices(tokens)
    boxed_set = set(all_boxed_idx)
    answer_set = set(answer_idx)
    keyword_set = set(keyword_idx)

    is_correct = resp['is_correct']

    target_all = correct_all_entropies if is_correct else incorrect_all_entropies
    target_boxed = correct_boxed_entropies if is_correct else incorrect_boxed_entropies
    target_answer = correct_answer_entropies if is_correct else incorrect_answer_entropies
    target_nonboxed = correct_nonboxed_entropies if is_correct else incorrect_nonboxed_entropies
    target_keyword = correct_boxed_keyword_entropies if is_correct else incorrect_boxed_keyword_entropies
    target_pos = correct_position_entropies if is_correct else incorrect_position_entropies

    for i, (tok, ent) in enumerate(zip(tokens, entropies)):
        target_all.append(ent)

        normalized_pos = int(i / len(tokens) * N_BINS)
        normalized_pos = min(normalized_pos, N_BINS - 1)
        target_pos[normalized_pos].append(ent)

        if i in boxed_set:
            target_boxed.append(ent)
        else:
            target_nonboxed.append(ent)

        if i in answer_set:
            target_answer.append(ent)

        if i in keyword_set:
            target_keyword.append(ent)

    # Context window around boxed region
    if all_boxed_idx:
        boxed_start = all_boxed_idx[0]
        target_ctx = correct_boxed_context if is_correct else incorrect_boxed_context
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + len(all_boxed_idx)):
            idx = boxed_start + offset
            if 0 <= idx < len(entropies):
                target_ctx[offset].append(entropies[idx])

print(f"\nToken counts:")
print(f"  Correct all tokens: {len(correct_all_entropies)}")
print(f"  Incorrect all tokens: {len(incorrect_all_entropies)}")
print(f"  Correct boxed tokens: {len(correct_boxed_entropies)}")
print(f"  Incorrect boxed tokens: {len(incorrect_boxed_entropies)}")
print(f"  Correct answer-digit tokens: {len(correct_answer_entropies)}")
print(f"  Incorrect answer-digit tokens: {len(incorrect_answer_entropies)}")

# ============================================================
# 4. Compute Forward & Reverse KL divergence (token-level)
# ============================================================
def compute_kl_divergence(p_samples, q_samples, n_bins=200, eps=1e-10):
    """
    Compute KL(P || Q) from samples.
    Uses histogram-based density estimation.
    Returns KL(P||Q), the bin edges, P distribution, Q distribution.
    """
    all_samples = np.concatenate([p_samples, q_samples])
    min_val, max_val = np.min(all_samples), np.max(all_samples)
    bins = np.linspace(min_val - 0.01, max_val + 0.01, n_bins + 1)

    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)

    p_hist = p_hist + eps
    q_hist = q_hist + eps

    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return kl, bins, p_hist, q_hist

def compute_token_level_kl_stats(correct_ent, incorrect_ent, label=""):
    """Compute comprehensive KL statistics."""
    c = np.array(correct_ent)
    ic = np.array(incorrect_ent)

    fwd_kl, bins, p_c, p_ic = compute_kl_divergence(c, ic)
    rev_kl, _, _, _ = compute_kl_divergence(ic, c)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    js_div = 0.5 * np.sum(p_c * np.log(2 * p_c / (p_c + p_ic))) + \
             0.5 * np.sum(p_ic * np.log(2 * p_ic / (p_c + p_ic)))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Correct: n={len(c)}, mean={c.mean():.4f}, std={c.std():.4f}, median={np.median(c):.4f}")
    print(f"  Incorrect: n={len(ic)}, mean={ic.mean():.4f}, std={ic.std():.4f}, median={np.median(ic):.4f}")
    print(f"  Forward  KL(Correct || Incorrect) = {fwd_kl:.6f}")
    print(f"  Reverse  KL(Incorrect || Correct) = {rev_kl:.6f}")
    print(f"  JS Divergence                     = {js_div:.6f}")
    print(f"  KL Asymmetry (Fwd - Rev)          = {fwd_kl - rev_kl:.6f}")
    print(f"  KL Ratio (Fwd / Rev)              = {fwd_kl / rev_kl:.4f}" if rev_kl > 0 else "")

    ks_stat, ks_pval = stats.ks_2samp(c, ic)
    mw_stat, mw_pval = stats.mannwhitneyu(c, ic, alternative='two-sided')
    print(f"  KS test: stat={ks_stat:.4f}, p={ks_pval:.2e}")
    print(f"  Mann-Whitney U: stat={mw_stat:.0f}, p={mw_pval:.2e}")

    return {
        'fwd_kl': fwd_kl, 'rev_kl': rev_kl, 'js_div': js_div,
        'bins': bins, 'p_c': p_c, 'p_ic': p_ic,
        'c_mean': c.mean(), 'ic_mean': ic.mean(),
        'c_std': c.std(), 'ic_std': ic.std(),
        'c_median': np.median(c), 'ic_median': np.median(ic),
    }

print("\n" + "=" * 70)
print("TOKEN-LEVEL FORWARD & REVERSE KL DIVERGENCE ANALYSIS")
print("=" * 70)

kl_all = compute_token_level_kl_stats(
    correct_all_entropies, incorrect_all_entropies,
    "ALL TOKENS"
)

kl_boxed = compute_token_level_kl_stats(
    correct_boxed_entropies, incorrect_boxed_entropies,
    "\\boxed{...} REGION TOKENS"
)

kl_answer = compute_token_level_kl_stats(
    correct_answer_entropies, incorrect_answer_entropies,
    "ANSWER CONTENT TOKENS (digits inside \\boxed{})"
)

kl_nonboxed = compute_token_level_kl_stats(
    correct_nonboxed_entropies, incorrect_nonboxed_entropies,
    "NON-BOXED TOKENS (reasoning body)"
)

kl_keyword = compute_token_level_kl_stats(
    correct_boxed_keyword_entropies, incorrect_boxed_keyword_entropies,
    "\\boxed KEYWORD TOKENS"
)

# ============================================================
# 5. Position-normalized KL divergence curve
# ============================================================
print("\n" + "=" * 70)
print("POSITION-NORMALIZED TOKEN KL DIVERGENCE")
print("=" * 70)

position_fwd_kl = []
position_rev_kl = []
position_js = []
position_c_mean = []
position_ic_mean = []

for pos in range(N_BINS):
    c_vals = correct_position_entropies.get(pos, [])
    ic_vals = incorrect_position_entropies.get(pos, [])
    if len(c_vals) >= 10 and len(ic_vals) >= 10:
        fkl, _, _, _ = compute_kl_divergence(np.array(c_vals), np.array(ic_vals), n_bins=50)
        rkl, _, _, _ = compute_kl_divergence(np.array(ic_vals), np.array(c_vals), n_bins=50)
        m = 0.5 * (np.array(c_vals).mean() + np.array(ic_vals).mean())
        position_fwd_kl.append(fkl)
        position_rev_kl.append(rkl)
        position_c_mean.append(np.mean(c_vals))
        position_ic_mean.append(np.mean(ic_vals))
    else:
        position_fwd_kl.append(np.nan)
        position_rev_kl.append(np.nan)
        position_c_mean.append(np.nan)
        position_ic_mean.append(np.nan)

# ============================================================
# 6. Context window around \\boxed{} KL analysis
# ============================================================
print("\n" + "=" * 70)
print("CONTEXT WINDOW AROUND \\boxed{} - TOKEN KL DIVERGENCE")
print("=" * 70)

context_offsets = sorted(set(list(correct_boxed_context.keys()) + list(incorrect_boxed_context.keys())))
context_fwd_kl = []
context_rev_kl = []
context_offsets_valid = []
context_c_mean = []
context_ic_mean = []

for offset in context_offsets:
    c_vals = correct_boxed_context.get(offset, [])
    ic_vals = incorrect_boxed_context.get(offset, [])
    if len(c_vals) >= 5 and len(ic_vals) >= 5:
        fkl, _, _, _ = compute_kl_divergence(np.array(c_vals), np.array(ic_vals), n_bins=30)
        rkl, _, _, _ = compute_kl_divergence(np.array(ic_vals), np.array(c_vals), n_bins=30)
        context_offsets_valid.append(offset)
        context_fwd_kl.append(fkl)
        context_rev_kl.append(rkl)
        context_c_mean.append(np.mean(c_vals))
        context_ic_mean.append(np.mean(ic_vals))

print(f"Valid offsets: {len(context_offsets_valid)} (from {min(context_offsets_valid)} to {max(context_offsets_valid)})")

# ============================================================
# 7. Per-file analysis (training progression)
# ============================================================
print("\n" + "=" * 70)
print("PER-FILE (CHECKPOINT) ANALYSIS")
print("=" * 70)

for tag_label, fpath in zip(["092147 (better)", "090427 (weaker)"], files):
    with open(fpath) as f:
        data = json.load(f)

    c_ent, ic_ent = [], []
    c_boxed, ic_boxed = [], []
    c_answer, ic_answer = [], []

    for result in data['results']:
        for resp in result['responses']:
            tokens, entropies = extract_token_entropies(resp)
            all_bi, ans_i, _ = find_boxed_token_indices(tokens)
            boxed_set = set(all_bi)
            ans_set = set(ans_i)

            target_all = c_ent if resp['is_correct'] else ic_ent
            target_b = c_boxed if resp['is_correct'] else ic_boxed
            target_a = c_answer if resp['is_correct'] else ic_answer

            for i, ent in enumerate(entropies):
                target_all.append(ent)
                if i in boxed_set:
                    target_b.append(ent)
                if i in ans_set:
                    target_a.append(ent)

    print(f"\n--- Checkpoint {tag_label} ---")
    if len(c_ent) > 0 and len(ic_ent) > 0:
        compute_token_level_kl_stats(c_ent, ic_ent, f"ALL TOKENS [{tag_label}]")
    if len(c_boxed) > 0 and len(ic_boxed) > 0:
        compute_token_level_kl_stats(c_boxed, ic_boxed, f"BOXED TOKENS [{tag_label}]")
    if len(c_answer) > 0 and len(ic_answer) > 0:
        compute_token_level_kl_stats(c_answer, ic_answer, f"ANSWER TOKENS [{tag_label}]")

# ============================================================
# 8. Entropy quantile analysis for KL contribution
# ============================================================
print("\n" + "=" * 70)
print("ENTROPY QUANTILE ANALYSIS - WHERE KL DIVERGENCE CONCENTRATES")
print("=" * 70)

c_all = np.array(correct_all_entropies)
ic_all = np.array(incorrect_all_entropies)

quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
print(f"\n{'Quantile':>10s} | {'Correct':>10s} | {'Incorrect':>10s} | {'Delta':>10s}")
print("-" * 50)
for q in quantiles:
    cq = np.quantile(c_all, q)
    iq = np.quantile(ic_all, q)
    print(f"{q:>10.2f} | {cq:>10.4f} | {iq:>10.4f} | {cq - iq:>+10.4f}")

# Boxed tokens
c_boxed_arr = np.array(correct_boxed_entropies)
ic_boxed_arr = np.array(incorrect_boxed_entropies)
print(f"\nBOXED region quantiles:")
print(f"{'Quantile':>10s} | {'Correct':>10s} | {'Incorrect':>10s} | {'Delta':>10s}")
print("-" * 50)
for q in quantiles:
    cq = np.quantile(c_boxed_arr, q)
    iq = np.quantile(ic_boxed_arr, q)
    print(f"{q:>10.2f} | {cq:>10.4f} | {iq:>10.4f} | {cq - iq:>+10.4f}")

# Answer content tokens
c_ans_arr = np.array(correct_answer_entropies)
ic_ans_arr = np.array(incorrect_answer_entropies)
print(f"\nANSWER CONTENT (digits) quantiles:")
print(f"{'Quantile':>10s} | {'Correct':>10s} | {'Incorrect':>10s} | {'Delta':>10s}")
print("-" * 50)
for q in quantiles:
    cq = np.quantile(c_ans_arr, q)
    iq = np.quantile(ic_ans_arr, q)
    print(f"{q:>10.2f} | {cq:>10.4f} | {iq:>10.4f} | {cq - iq:>+10.4f}")

# ============================================================
# 9. Zero-entropy token ratio analysis
# ============================================================
print("\n" + "=" * 70)
print("ZERO-ENTROPY (DETERMINISTIC) TOKEN ANALYSIS")
print("=" * 70)

c_zero_ratio = np.mean(c_all == 0)
ic_zero_ratio = np.mean(ic_all == 0)
print(f"All tokens - Correct zero-entropy ratio: {c_zero_ratio:.4f}")
print(f"All tokens - Incorrect zero-entropy ratio: {ic_zero_ratio:.4f}")

c_boxed_zero = np.mean(c_boxed_arr == 0)
ic_boxed_zero = np.mean(ic_boxed_arr == 0)
print(f"Boxed tokens - Correct zero-entropy ratio: {c_boxed_zero:.4f}")
print(f"Boxed tokens - Incorrect zero-entropy ratio: {ic_boxed_zero:.4f}")

c_ans_zero = np.mean(c_ans_arr == 0)
ic_ans_zero = np.mean(ic_ans_arr == 0)
print(f"Answer tokens - Correct zero-entropy ratio: {c_ans_zero:.4f}")
print(f"Answer tokens - Incorrect zero-entropy ratio: {ic_ans_zero:.4f}")

# ============================================================
# 10. PLOTTING
# ============================================================
out_dir = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy"

# --- Figure 1: Main KL comparison ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Token-Level Forward & Reverse KL Divergence: Correct vs Incorrect', fontsize=16, fontweight='bold')

regions = [
    ("All Tokens", kl_all),
    ("\\\\boxed{...} Region", kl_boxed),
    ("Answer Digits", kl_answer),
    ("Non-boxed (Reasoning)", kl_nonboxed),
    ("\\\\boxed Keyword", kl_keyword),
]

for idx, (name, kl_data) in enumerate(regions):
    ax = axes[idx // 3][idx % 3]
    bin_centers = (kl_data['bins'][:-1] + kl_data['bins'][1:]) / 2
    ax.fill_between(bin_centers, kl_data['p_c'], alpha=0.4, color='green', label='Correct')
    ax.fill_between(bin_centers, kl_data['p_ic'], alpha=0.4, color='red', label='Incorrect')
    ax.plot(bin_centers, kl_data['p_c'], color='green', linewidth=1.5)
    ax.plot(bin_centers, kl_data['p_ic'], color='red', linewidth=1.5)
    ax.set_title(f"{name}\nFwd KL={kl_data['fwd_kl']:.4f}, Rev KL={kl_data['rev_kl']:.4f}")
    ax.set_xlabel('Token Entropy')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.axvline(kl_data['c_mean'], color='green', linestyle='--', alpha=0.7, label=f"Correct mean={kl_data['c_mean']:.3f}")
    ax.axvline(kl_data['ic_mean'], color='red', linestyle='--', alpha=0.7, label=f"Incorrect mean={kl_data['ic_mean']:.3f}")

# 6th subplot: bar chart of KL values
ax = axes[1][2]
region_names = [r[0] for r in regions]
fwd_kls = [r[1]['fwd_kl'] for r in regions]
rev_kls = [r[1]['rev_kl'] for r in regions]
x = np.arange(len(region_names))
w = 0.35
bars1 = ax.bar(x - w/2, fwd_kls, w, color='steelblue', label='Forward KL(C||IC)')
bars2 = ax.bar(x + w/2, rev_kls, w, color='coral', label='Reverse KL(IC||C)')
ax.set_xticks(x)
ax.set_xticklabels(region_names, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('KL Divergence')
ax.set_title('KL Divergence by Token Region')
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}',
            ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}',
            ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(f"{out_dir}/token_kl_distributions.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: token_kl_distributions.png")

# --- Figure 2: Position-normalized KL curve ---
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Position-Normalized Token KL Divergence Along Response', fontsize=14, fontweight='bold')

x_pos = np.arange(N_BINS) / N_BINS * 100

ax = axes[0]
ax.plot(x_pos, position_fwd_kl, color='steelblue', linewidth=1.5, label='Forward KL(C||IC)')
ax.plot(x_pos, position_rev_kl, color='coral', linewidth=1.5, label='Reverse KL(IC||C)')
ax.fill_between(x_pos, position_fwd_kl, alpha=0.2, color='steelblue')
ax.fill_between(x_pos, position_rev_kl, alpha=0.2, color='coral')
ax.set_xlabel('Relative Position in Response (%)')
ax.set_ylabel('KL Divergence')
ax.set_title('Forward & Reverse KL by Position')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvspan(85, 100, alpha=0.1, color='yellow', label='Typical \\boxed{} region')

ax = axes[1]
ax.plot(x_pos, position_c_mean, color='green', linewidth=1.5, label='Correct Mean Entropy')
ax.plot(x_pos, position_ic_mean, color='red', linewidth=1.5, label='Incorrect Mean Entropy')
ax.fill_between(x_pos, position_c_mean, position_ic_mean, alpha=0.2, color='gray')
ax.set_xlabel('Relative Position in Response (%)')
ax.set_ylabel('Mean Token Entropy')
ax.set_title('Mean Entropy by Position: Correct vs Incorrect')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/token_kl_by_position.png", dpi=150, bbox_inches='tight')
print(f"Saved: token_kl_by_position.png")

# --- Figure 3: Context around \\boxed{} ---
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Token KL Divergence Around \\\\boxed{} Region', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(context_offsets_valid, context_fwd_kl, 'o-', color='steelblue', markersize=3, linewidth=1.2, label='Forward KL(C||IC)')
ax.plot(context_offsets_valid, context_rev_kl, 's-', color='coral', markersize=3, linewidth=1.2, label='Reverse KL(IC||C)')
ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='\\\\boxed{ start')
ax.axvspan(0, max(context_offsets_valid) * 0.3, alpha=0.08, color='orange')
ax.set_xlabel('Token Offset from \\\\boxed{ Start')
ax.set_ylabel('KL Divergence')
ax.set_title('KL Divergence by Token Offset from \\\\boxed{}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(context_offsets_valid, context_c_mean, 'o-', color='green', markersize=3, linewidth=1.2, label='Correct Mean Entropy')
ax.plot(context_offsets_valid, context_ic_mean, 's-', color='red', markersize=3, linewidth=1.2, label='Incorrect Mean Entropy')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.axvspan(0, max(context_offsets_valid) * 0.3, alpha=0.08, color='orange')
ax.set_xlabel('Token Offset from \\\\boxed{ Start')
ax.set_ylabel('Mean Token Entropy')
ax.set_title('Mean Entropy Around \\\\boxed{} Region')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/token_kl_boxed_context.png", dpi=150, bbox_inches='tight')
print(f"Saved: token_kl_boxed_context.png")

# --- Figure 4: KL asymmetry heatmap ---
fig, ax = plt.subplots(figsize=(12, 5))
asymmetry = np.array(position_fwd_kl) - np.array(position_rev_kl)
colors = ['steelblue' if a > 0 else 'coral' for a in asymmetry]
ax.bar(x_pos, asymmetry, color=colors, width=1.0, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Relative Position in Response (%)')
ax.set_ylabel('KL Asymmetry (Fwd - Rev)')
ax.set_title('KL Asymmetry Along Response\n(Positive = Forward KL > Reverse KL, Negative = Reverse > Forward)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/token_kl_asymmetry.png", dpi=150, bbox_inches='tight')
print(f"Saved: token_kl_asymmetry.png")

# --- Figure 5: CDF comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Cumulative Distribution of Token Entropies: Correct vs Incorrect', fontsize=14, fontweight='bold')

for ax, (name, c_data, ic_data) in zip(axes, [
    ("All Tokens", c_all, ic_all),
    ("\\\\boxed{} Region", c_boxed_arr, ic_boxed_arr),
    ("Answer Digits", c_ans_arr, ic_ans_arr),
]):
    c_sorted = np.sort(c_data)
    ic_sorted = np.sort(ic_data)
    ax.plot(c_sorted, np.linspace(0, 1, len(c_sorted)), color='green', linewidth=1.5, label='Correct')
    ax.plot(ic_sorted, np.linspace(0, 1, len(ic_sorted)), color='red', linewidth=1.5, label='Incorrect')
    ax.set_xlabel('Token Entropy')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/token_entropy_cdf.png", dpi=150, bbox_inches='tight')
print(f"Saved: token_entropy_cdf.png")

# --- Figure 6: Pointwise KL contribution ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Pointwise KL Contribution by Entropy Value', fontsize=14, fontweight='bold')

for ax, (name, kl_data) in zip(axes, [("All Tokens", kl_all), ("\\\\boxed{} Answer Digits", kl_answer)]):
    bin_centers = (kl_data['bins'][:-1] + kl_data['bins'][1:]) / 2
    p_c = kl_data['p_c']
    p_ic = kl_data['p_ic']

    fwd_pointwise = p_c * np.log(p_c / p_ic)
    rev_pointwise = p_ic * np.log(p_ic / p_c)

    ax.bar(bin_centers, fwd_pointwise, width=(bin_centers[1]-bin_centers[0]),
           alpha=0.5, color='steelblue', label='Fwd KL contrib')
    ax.bar(bin_centers, -rev_pointwise, width=(bin_centers[1]-bin_centers[0]),
           alpha=0.5, color='coral', label='-Rev KL contrib')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Token Entropy')
    ax.set_ylabel('Pointwise KL Contribution')
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/token_kl_pointwise.png", dpi=150, bbox_inches='tight')
print(f"Saved: token_kl_pointwise.png")

# ============================================================
# 11. Final Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF KEY FINDINGS")
print("=" * 70)
print(f"""
1. OVERALL TOKEN-LEVEL KL DIVERGENCE:
   Forward KL(Correct || Incorrect) = {kl_all['fwd_kl']:.6f}
   Reverse KL(Incorrect || Correct) = {kl_all['rev_kl']:.6f}
   {'Forward > Reverse' if kl_all['fwd_kl'] > kl_all['rev_kl'] else 'Reverse > Forward'}: KL asymmetry = {abs(kl_all['fwd_kl'] - kl_all['rev_kl']):.6f}

2. \\boxed{{}} REGION KL DIVERGENCE:
   Forward KL = {kl_boxed['fwd_kl']:.6f}
   Reverse KL = {kl_boxed['rev_kl']:.6f}
   {'Forward > Reverse' if kl_boxed['fwd_kl'] > kl_boxed['rev_kl'] else 'Reverse > Forward'}: KL asymmetry = {abs(kl_boxed['fwd_kl'] - kl_boxed['rev_kl']):.6f}

3. ANSWER DIGITS KL DIVERGENCE:
   Forward KL = {kl_answer['fwd_kl']:.6f}
   Reverse KL = {kl_answer['rev_kl']:.6f}
   {'Forward > Reverse' if kl_answer['fwd_kl'] > kl_answer['rev_kl'] else 'Reverse > Forward'}: KL asymmetry = {abs(kl_answer['fwd_kl'] - kl_answer['rev_kl']):.6f}

4. MEAN ENTROPY COMPARISON:
   All tokens:     Correct={kl_all['c_mean']:.4f} vs Incorrect={kl_all['ic_mean']:.4f} (delta={kl_all['c_mean']-kl_all['ic_mean']:+.4f})
   Boxed region:   Correct={kl_boxed['c_mean']:.4f} vs Incorrect={kl_boxed['ic_mean']:.4f} (delta={kl_boxed['c_mean']-kl_boxed['ic_mean']:+.4f})
   Answer digits:  Correct={kl_answer['c_mean']:.4f} vs Incorrect={kl_answer['ic_mean']:.4f} (delta={kl_answer['c_mean']-kl_answer['ic_mean']:+.4f})
   Non-boxed:      Correct={kl_nonboxed['c_mean']:.4f} vs Incorrect={kl_nonboxed['ic_mean']:.4f} (delta={kl_nonboxed['c_mean']-kl_nonboxed['ic_mean']:+.4f})

5. ZERO-ENTROPY TOKEN RATIO:
   All:    Correct={c_zero_ratio:.4f}, Incorrect={ic_zero_ratio:.4f}
   Boxed:  Correct={c_boxed_zero:.4f}, Incorrect={ic_boxed_zero:.4f}
   Answer: Correct={c_ans_zero:.4f}, Incorrect={ic_ans_zero:.4f}

6. KL DIVERGENCE AMPLIFICATION IN ANSWER REGION:
   All->Boxed amplification (Fwd): {kl_boxed['fwd_kl']/kl_all['fwd_kl']:.2f}x
   All->Answer amplification (Fwd): {kl_answer['fwd_kl']/kl_all['fwd_kl']:.2f}x
   All->Boxed amplification (Rev): {kl_boxed['rev_kl']/kl_all['rev_kl']:.2f}x
   All->Answer amplification (Rev): {kl_answer['rev_kl']/kl_all['rev_kl']:.2f}x
""")

print("All plots saved to:", out_dir)
