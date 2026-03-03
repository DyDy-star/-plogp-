"""
Analyze why KL and entropy explode under symmetric -log(p) redistribution
while accuracy still improves.

Key questions:
1. What determines how much each candidate token's probability rises/falls?
2. Are the changes consistent across token positions?
3. Joint distribution analysis of p and -log(p)
4. Why is there asymmetry despite "symmetric" redistribution?
"""

import json
import numpy as np
import torch
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = Path("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy")
V = 151936

# ============================================================
# Part 1: Load real entropy distributions from data files
# ============================================================
print("=" * 70)
print("Part 1: Load real token entropy distributions")
print("=" * 70)

trained_file = DATA_DIR / "aime_eval_full_entropy_20260207_092147.json"
base_file = DATA_DIR / "aime_eval_full_entropy_20260207_090427.json"

with open(trained_file) as f:
    trained_data = json.load(f)
with open(base_file) as f:
    base_data = json.load(f)

all_entropies_trained = []
all_top1_trained = []
all_top5_trained = []
all_top10_trained = []
correct_entropies = []
incorrect_entropies = []

for result in trained_data['results']:
    for resp in result['responses']:
        ea = resp['entropy_analysis']
        is_correct = resp['is_correct']
        for step in ea['steps']:
            ents = step['token_entropies']
            top1 = step.get('top1_prob', None)
            top5 = step.get('top5_prob_mass', None)
            top10 = step.get('top10_prob_mass', None)
            all_entropies_trained.extend(ents)
            if is_correct:
                correct_entropies.extend(ents)
            else:
                incorrect_entropies.extend(ents)

all_entropies_trained = np.array(all_entropies_trained)
correct_entropies = np.array(correct_entropies)
incorrect_entropies = np.array(incorrect_entropies)

print(f"Total tokens: {len(all_entropies_trained)}")
print(f"Correct response tokens: {len(correct_entropies)}")
print(f"Incorrect response tokens: {len(incorrect_entropies)}")
print(f"Mean entropy (correct):   {correct_entropies.mean():.4f}")
print(f"Mean entropy (incorrect): {incorrect_entropies.mean():.4f}")
print(f"Mean entropy (all):       {all_entropies_trained.mean():.4f}")
print()

# ============================================================
# Part 2: Approximate distributions and analyze redistribution
# ============================================================
print("=" * 70)
print("Part 2: Simulate redistribution on synthetic distributions")
print("=" * 70)

def make_distribution(entropy_target, vocab_size=V):
    """Create a distribution with approximately the given entropy using Zipf."""
    if entropy_target < 0.01:
        p = torch.zeros(vocab_size)
        p[0] = 1.0
        return p

    s_low, s_high = 0.1, 10.0
    for _ in range(50):
        s_mid = (s_low + s_high) / 2
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float64)
        unnorm = 1.0 / ranks.pow(s_mid)
        p = unnorm / unnorm.sum()
        H = -(p * torch.log(p + 1e-30)).sum().item()
        if H > entropy_target:
            s_low = s_mid
        else:
            s_high = s_mid

    return p.float()


def analyze_redistribution(p, alpha=0.3):
    """
    Given distribution p, analyze how -log(p) redistribution changes gradients.
    Returns detailed per-token analysis.
    """
    p = p.float()
    logp = torch.log(p + 1e-30)
    surp = -logp  # = -log(p)

    target_idx = 0
    p_a = p[target_idx].item()

    # Standard gradient for non-target tokens: g_v = A * p_v
    # With A > 0 (positive sample):
    A_pos = 1.0
    g_std_pos = A_pos * p.clone()
    g_std_pos[target_idx] = A_pos * (p_a - 1.0)

    # -log(p) redistribution
    surp_total = surp.sum() - surp[target_idx] + 1e-10
    G_total_pos = g_std_pos.sum() - g_std_pos[target_idx]

    g_surp_pos = G_total_pos * surp / surp_total
    g_surp_pos[target_idx] = 0.0
    g_blend_pos = (1 - alpha) * g_std_pos.clone() + alpha * g_surp_pos
    g_blend_pos[target_idx] = g_std_pos[target_idx]

    # Same for negative sample (A < 0)
    A_neg = -1.0
    g_std_neg = A_neg * p.clone()
    g_std_neg[target_idx] = A_neg * (p_a - 1.0)

    G_total_neg = g_std_neg.sum() - g_std_neg[target_idx]
    g_surp_neg = G_total_neg * surp / surp_total
    g_surp_neg[target_idx] = 0.0
    g_blend_neg = (1 - alpha) * g_std_neg.clone() + alpha * g_surp_neg
    g_blend_neg[target_idx] = g_std_neg[target_idx]

    return {
        'p': p, 'surp': surp, 'p_a': p_a,
        'g_std_pos': g_std_pos, 'g_blend_pos': g_blend_pos,
        'g_std_neg': g_std_neg, 'g_blend_neg': g_blend_neg,
        'G_total_pos': G_total_pos.item(),
        'G_total_neg': G_total_neg.item(),
    }


# Test with different entropy levels
entropy_levels = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

print(f"\n{'Entropy':>8} {'p_target':>10} {'|G_pos|':>10} {'|G_neg|':>10} "
      f"{'Ratio':>8} {'ΔH_pos':>10} {'ΔH_neg':>10} {'ΔH_net':>10}")
print("-" * 90)

lr = 0.01
for H_target in entropy_levels:
    p = make_distribution(H_target)
    res = analyze_redistribution(p, alpha=0.3)

    # Compute entropy change after one gradient step
    logits = torch.log(p + 1e-30)

    # Positive sample
    z_after_pos = logits - lr * res['g_blend_pos']
    p_after_pos = torch.softmax(z_after_pos, dim=-1)
    H_after_pos = -(p_after_pos * torch.log(p_after_pos + 1e-30)).sum().item()

    # Negative sample
    z_after_neg = logits - lr * res['g_blend_neg']
    p_after_neg = torch.softmax(z_after_neg, dim=-1)
    H_after_neg = -(p_after_neg * torch.log(p_after_neg + 1e-30)).sum().item()

    H_before = -(p * torch.log(p + 1e-30)).sum().item()
    dH_pos = H_after_pos - H_before
    dH_neg = H_after_neg - H_before

    ratio = abs(res['G_total_neg']) / (abs(res['G_total_pos']) + 1e-20)

    print(f"{H_target:8.2f} {res['p_a']:10.6f} {abs(res['G_total_pos']):10.6f} "
          f"{abs(res['G_total_neg']):10.6f} {ratio:8.4f} "
          f"{dH_pos:+10.6f} {dH_neg:+10.6f} {dH_pos + dH_neg:+10.6f}")

print()

# ============================================================
# Part 3: Analyze with REALISTIC p_a values for correct vs incorrect
# ============================================================
print("=" * 70)
print("Part 3: Asymmetry from p_a difference (correct vs incorrect)")
print("=" * 70)

# Entropy → effective p_a for target token
# For correct responses: higher p_a (model somewhat knows the answer)
# For incorrect responses: lower p_a (model chose a less-confident option)

print("\nSimulating GRPO batch with realistic p_a distribution:")
print("Correct responses: p_a ~ higher (model knows correct token)")
print("Incorrect responses: p_a ~ lower (model chose wrong token)")
print()

# Sample from real entropy distributions
n_samples = 1000
correct_ents = np.random.choice(correct_entropies, n_samples)
incorrect_ents = np.random.choice(incorrect_entropies, n_samples)

# For each entropy, get the distribution and compute redistribution
dH_pos_list = []
dH_neg_list = []
pa_pos_list = []
pa_neg_list = []

# Use a smaller vocab for speed in simulation
V_sim = 10000
lr_sim = 0.01
alpha = 0.3

# Pre-compute distributions for different entropy levels
entropy_bins = np.linspace(0, 4.0, 50)
cached_dists = {}

for H_target in entropy_bins:
    cached_dists[round(H_target, 2)] = make_distribution(H_target, V_sim)

def get_cached_dist(entropy):
    key = round(min(max(entropy, 0.0), 3.98), 2)
    closest = min(cached_dists.keys(), key=lambda x: abs(x - key))
    return cached_dists[closest]

print("Computing redistribution effects for correct (positive A) samples...")
for ent in correct_ents[:500]:
    p = get_cached_dist(ent)
    p_a = p[0].item()
    pa_pos_list.append(p_a)

    surp = -torch.log(p + 1e-30)
    surp_total = surp.sum() - surp[0] + 1e-10
    A = 1.0

    g_std = A * p.clone()
    g_std[0] = A * (p_a - 1.0)
    G_total = g_std[1:].sum()
    g_surp = G_total * surp / surp_total
    g_surp[0] = 0.0
    g_blend = (1 - alpha) * g_std + alpha * g_surp
    g_blend[0] = g_std[0]

    logits = torch.log(p + 1e-30)
    z_after = logits - lr_sim * g_blend
    p_after = torch.softmax(z_after, dim=-1)
    H_before = -(p * torch.log(p + 1e-30)).sum().item()
    H_after = -(p_after * torch.log(p_after + 1e-30)).sum().item()
    dH_pos_list.append(H_after - H_before)

print("Computing redistribution effects for incorrect (negative A) samples...")
for ent in incorrect_ents[:500]:
    p = get_cached_dist(ent)
    p_a = p[0].item()
    pa_neg_list.append(p_a)

    surp = -torch.log(p + 1e-30)
    surp_total = surp.sum() - surp[0] + 1e-10
    A = -1.0

    g_std = A * p.clone()
    g_std[0] = A * (p_a - 1.0)
    G_total = g_std[1:].sum()
    g_surp = G_total * surp / surp_total
    g_surp[0] = 0.0
    g_blend = (1 - alpha) * g_std + alpha * g_surp
    g_blend[0] = g_std[0]

    logits = torch.log(p + 1e-30)
    z_after = logits - lr_sim * g_blend
    p_after = torch.softmax(z_after, dim=-1)
    H_before = -(p * torch.log(p + 1e-30)).sum().item()
    H_after = -(p_after * torch.log(p_after + 1e-30)).sum().item()
    dH_neg_list.append(H_after - H_before)

dH_pos = np.array(dH_pos_list)
dH_neg = np.array(dH_neg_list)
pa_pos = np.array(pa_pos_list)
pa_neg = np.array(pa_neg_list)

print(f"\n  Correct samples (A > 0):")
print(f"    mean p_a:     {pa_pos.mean():.6f}")
print(f"    mean (1-p_a): {(1-pa_pos).mean():.6f}")
print(f"    mean ΔH:      {dH_pos.mean():+.8f}")

print(f"\n  Incorrect samples (A < 0):")
print(f"    mean p_a:     {pa_neg.mean():.6f}")
print(f"    mean (1-p_a): {(1-pa_neg).mean():.6f}")
print(f"    mean ΔH:      {dH_neg.mean():+.8f}")

print(f"\n  Net ΔH per step:  {(dH_pos.mean() + dH_neg.mean()):+.8f}")
print(f"  Ratio |ΔH_neg/ΔH_pos|: {abs(dH_neg.mean()) / (abs(dH_pos.mean()) + 1e-20):.4f}")

print()

# ============================================================
# Part 4: Joint distribution analysis — how does (p_v, -log p_v) jointly
#          determine the redistribution magnitude?
# ============================================================
print("=" * 70)
print("Part 4: Joint distribution of p_v and -log(p_v) in redistribution")
print("=" * 70)

# For a typical distribution, analyze per-token gradient change
H_target = 0.5  # typical entropy
p = make_distribution(H_target, V_sim)
surp = -torch.log(p + 1e-30)

# Standard redistribution weight: p_v / (1 - p_a)
# Surprisal redistribution weight: surp_v / surp_total
# Ratio (surprisal vs standard): (surp_v / surp_total) / (p_v / (1 - p_a))

p_a = p[0].item()
std_weight = p[1:] / (1 - p_a)
surp_total = surp[1:].sum() + 1e-10
surp_weight = surp[1:] / surp_total

# Correction factor: how much does each token's gradient change?
correction = surp_weight / (std_weight + 1e-20)

# Group by probability rank
ranks = torch.argsort(p[1:], descending=True)
n_ranks = len(ranks)

print(f"\nDistribution: H={H_target:.1f}, p_target={p_a:.6f}, V_sim={V_sim}")
print(f"\n{'Rank Range':>15} {'Avg p_v':>12} {'Avg -log(p)':>12} {'Avg p*log(p)':>12} "
      f"{'Std Weight':>12} {'Surp Weight':>12} {'Correction':>12}")
print("-" * 100)

rank_groups = [
    ("Top 1-10", 0, 10),
    ("Top 11-50", 10, 50),
    ("Top 51-200", 50, 200),
    ("Top 201-1000", 200, 1000),
    ("Tail 1001+", 1000, n_ranks),
]

for name, start, end in rank_groups:
    idx = ranks[start:min(end, n_ranks)]
    if len(idx) == 0:
        continue
    avg_p = p[1:][idx].mean().item()
    avg_surp = surp[1:][idx].mean().item()
    avg_plogp = -(p[1:][idx] * torch.log(p[1:][idx] + 1e-30)).mean().item()
    avg_std_w = std_weight[idx].mean().item()
    avg_surp_w = surp_weight[idx].mean().item()
    avg_corr = correction[idx].mean().item()
    print(f"{name:>15} {avg_p:12.8f} {avg_surp:12.4f} {avg_plogp:12.8f} "
          f"{avg_std_w:12.8f} {avg_surp_w:12.8f} {avg_corr:12.4f}")

print()
print("Correction > 1: token gets MORE gradient under -log(p) (tail tokens)")
print("Correction < 1: token gets LESS gradient under -log(p) (runners-up)")

# ============================================================
# Part 5: Verify — does the STANDARD gradient (no redistribution)
#          also show asymmetry between positive and negative?
# ============================================================
print()
print("=" * 70)
print("Part 5: Standard gradient also has ΔH asymmetry?")
print("=" * 70)

dH_std_pos_list = []
dH_std_neg_list = []

for ent in correct_ents[:500]:
    p = get_cached_dist(ent)
    p_a = p[0].item()
    A = 1.0
    g_std = A * p.clone()
    g_std[0] = A * (p_a - 1.0)

    logits = torch.log(p + 1e-30)
    z_after = logits - lr_sim * g_std
    p_after = torch.softmax(z_after, dim=-1)
    H_before = -(p * torch.log(p + 1e-30)).sum().item()
    H_after = -(p_after * torch.log(p_after + 1e-30)).sum().item()
    dH_std_pos_list.append(H_after - H_before)

for ent in incorrect_ents[:500]:
    p = get_cached_dist(ent)
    p_a = p[0].item()
    A = -1.0
    g_std = A * p.clone()
    g_std[0] = A * (p_a - 1.0)

    logits = torch.log(p + 1e-30)
    z_after = logits - lr_sim * g_std
    p_after = torch.softmax(z_after, dim=-1)
    H_before = -(p * torch.log(p + 1e-30)).sum().item()
    H_after = -(p_after * torch.log(p_after + 1e-30)).sum().item()
    dH_std_neg_list.append(H_after - H_before)

dH_std_pos = np.array(dH_std_pos_list)
dH_std_neg = np.array(dH_std_neg_list)

print(f"\n  STANDARD gradient (∝ p, no redistribution):")
print(f"    Positive (correct)  mean ΔH: {dH_std_pos.mean():+.8f}")
print(f"    Negative (incorrect) mean ΔH: {dH_std_neg.mean():+.8f}")
print(f"    Net ΔH:                       {(dH_std_pos.mean() + dH_std_neg.mean()):+.8f}")
print(f"    Ratio |ΔH_neg/ΔH_pos|:        {abs(dH_std_neg.mean()) / (abs(dH_std_pos.mean()) + 1e-20):.4f}")

print(f"\n  MODIFIED gradient (∝ -log p, symmetric redistribution):")
print(f"    Positive (correct)  mean ΔH: {dH_pos.mean():+.8f}")
print(f"    Negative (incorrect) mean ΔH: {dH_neg.mean():+.8f}")
print(f"    Net ΔH:                       {(dH_pos.mean() + dH_neg.mean()):+.8f}")
print(f"    Ratio |ΔH_neg/ΔH_pos|:        {abs(dH_neg.mean()) / (abs(dH_pos.mean()) + 1e-20):.4f}")

amplification = abs(dH_pos.mean() + dH_neg.mean()) / (abs(dH_std_pos.mean() + dH_std_neg.mean()) + 1e-20)
print(f"\n  Entropy drift amplification factor: {amplification:.2f}x")
print()

# ============================================================
# Part 6: Why does accuracy still improve?
# ============================================================
print("=" * 70)
print("Part 6: Target gradient is UNCHANGED — accuracy unaffected")
print("=" * 70)

H_target = 0.5
p = make_distribution(H_target, V_sim)
res = analyze_redistribution(p, alpha=0.3)

tgt_grad_std = res['g_std_pos'][0].item()
tgt_grad_mod = res['g_blend_pos'][0].item()
print(f"\n  Target token gradient:")
print(f"    Standard:  {tgt_grad_std:.8f}")
print(f"    Modified:  {tgt_grad_mod:.8f}")
print(f"    Difference: {abs(tgt_grad_std - tgt_grad_mod):.2e}")
print()
print("  --> Target gradient is EXACTLY preserved.")
print("  --> The model's learning signal for 'which token to choose' is unchanged.")
print("  --> Only the redistribution among NON-TARGET tokens changes.")
print("  --> This is why accuracy improves despite KL/entropy explosion.")

# Verify: non-target L1 is also preserved
nontgt_l1_std = res['g_std_pos'][1:].sum().item()
nontgt_l1_mod = res['g_blend_pos'][1:].sum().item()
print(f"\n  Non-target gradient L1:")
print(f"    Standard:  {nontgt_l1_std:.8f}")
print(f"    Modified:  {nontgt_l1_mod:.8f}")
print(f"    Difference: {abs(nontgt_l1_std - nontgt_l1_mod):.2e}")
print()

# ============================================================
# Part 7: Quantify — how much does redistribution amplify entropy changes?
# ============================================================
print("=" * 70)
print("Part 7: Per-rank entropy contribution analysis")
print("=" * 70)

# For positive sample, compute how much each rank group contributes to entropy change
H_target = 0.5
p = make_distribution(H_target, V_sim)
logits = torch.log(p + 1e-30)
p_a = p[0].item()
surp = -torch.log(p + 1e-30)

A = 1.0
g_std = A * p.clone()
g_std[0] = A * (p_a - 1.0)

surp_total_v = surp[1:].sum() + 1e-10
G_total = g_std[1:].sum()
g_surp = G_total * surp / surp_total_v
g_surp[0] = 0.0
g_blend = (1 - alpha) * g_std + alpha * g_surp
g_blend[0] = g_std[0]

z_std = logits - lr_sim * g_std
z_mod = logits - lr_sim * g_blend
p_std = torch.softmax(z_std, dim=-1)
p_mod = torch.softmax(z_mod, dim=-1)

print(f"\nPositive sample: how probability mass shifts per rank group")
print(f"{'Rank Range':>15} {'Δp_std':>12} {'Δp_mod':>12} {'Diff':>12} {'Direction':>12}")
print("-" * 70)

ranks = torch.argsort(p[1:], descending=True)
for name, start, end in rank_groups:
    idx = ranks[start:min(end, n_ranks)] + 1  # +1 because rank 0 is target
    if len(idx) == 0:
        continue
    dp_std = (p_std[idx] - p[idx]).sum().item()
    dp_mod = (p_mod[idx] - p[idx]).sum().item()
    diff = dp_mod - dp_std
    direction = "MORE loss" if diff < 0 else "LESS loss"
    print(f"{name:>15} {dp_std:+12.8f} {dp_mod:+12.8f} {diff:+12.8f} {direction:>12}")

print()
print("For POSITIVE samples, -log(p) redistribution:")
print("  Runners-up lose LESS probability (protected)")
print("  Tail tokens lose MORE probability (suppressed)")
print("  -> Conditional distribution becomes MORE peaked -> entropy ↓")
print()

# Same for negative sample
A = -1.0
g_std = A * p.clone()
g_std[0] = A * (p_a - 1.0)

G_total = g_std[1:].sum()
g_surp = G_total * surp / surp_total_v
g_surp[0] = 0.0
g_blend = (1 - alpha) * g_std + alpha * g_surp
g_blend[0] = g_std[0]

z_std = logits - lr_sim * g_std
z_mod = logits - lr_sim * g_blend
p_std = torch.softmax(z_std, dim=-1)
p_mod = torch.softmax(z_mod, dim=-1)

print(f"Negative sample: how probability mass shifts per rank group")
print(f"{'Rank Range':>15} {'Δp_std':>12} {'Δp_mod':>12} {'Diff':>12} {'Direction':>12}")
print("-" * 70)

for name, start, end in rank_groups:
    idx = ranks[start:min(end, n_ranks)] + 1
    if len(idx) == 0:
        continue
    dp_std = (p_std[idx] - p[idx]).sum().item()
    dp_mod = (p_mod[idx] - p[idx]).sum().item()
    diff = dp_mod - dp_std
    direction = "MORE gain" if diff > 0 else "LESS gain"
    print(f"{name:>15} {dp_std:+12.8f} {dp_mod:+12.8f} {diff:+12.8f} {direction:>12}")

print()
print("For NEGATIVE samples, -log(p) redistribution:")
print("  Runners-up gain LESS probability")
print("  Tail tokens gain MORE probability (restored)")
print("  -> Conditional distribution becomes MORE uniform -> entropy ↑")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key findings:

1. WHAT DETERMINES each token's change:
   - Standard gradient: change ∝ p_v (high-p tokens change most)
   - Modified gradient: change ∝ (1-α)*p_v + α*surp_v/C
   - The CORRECTION FACTOR = surp_v/(C*p_v) = -log(p_v)/C'
   - Tail tokens (low p, high -log p): correction >> 1 (amplified)
   - Runners-up (high p, low -log p): correction << 1 (dampened)

2. WHY ENTROPY EXPLODES despite symmetric redistribution:
   The formula is symmetric, but the INPUT is not:
   - Correct responses: higher p_a → smaller |G_total| → weaker compression
   - Incorrect responses: lower p_a → larger |G_total| → stronger restoration
   - Net: tail restoration > tail compression → entropy ↑↑

   The STANDARD gradient doesn't have this problem because ∝ p
   redistribution preserves conditional distribution regardless of |G_total|.

3. WHY ACCURACY STILL IMPROVES:
   SurprisalRedistribution only changes gradients for NON-TARGET tokens.
   The target token's gradient is exactly preserved.
   → The learning signal for "which token to choose" is unchanged.
   → Accuracy improves normally.
   → But the distribution SHAPE changes dramatically → KL explodes.

4. JOINT DISTRIBUTION perspective:
   p and -log(p) are monotonically related, but their RATIO (-log(p)/p)
   varies enormously: ~1 for p≈0.37, → ∞ for p→0, → 0 for p→1.
   This ratio determines the correction factor.
   The redistribution effect is strongest for tokens where -log(p)/p
   is most different from the average, i.e., the extreme tail.
""")
