"""
Multi-step simulation v2: Different targets per sample (realistic GRPO).

In real training, each response generates DIFFERENT tokens at each position.
The gradient at each position combines signals from multiple responses
with different targets. This is the KEY driver of entropy dynamics.
"""

import torch
import numpy as np

torch.manual_seed(42)

V = 5000
n_steps = 200
lr = 0.005
alpha = 0.3


def entropy(p):
    return -(p * torch.log(p + 1e-30)).sum().item()


def kl_div(p, q):
    return (p * torch.log((p + 1e-30) / (q + 1e-30))).sum().item()


# ============================================================
# Realistic simulation: different targets, different advantages
# ============================================================
print("=" * 70)
print("Realistic GRPO simulation: different targets per sample")
print("=" * 70)

# Initialize distribution (moderately peaked)
logits_init = torch.randn(V) * 0.5
logits_init[0] += 3.0  # token 0 is the "correct" answer
p_init = torch.softmax(logits_init, dim=-1)
print(f"Initial: H={entropy(p_init):.4f}, p[0]={p_init[0]:.6f}")

# Simulate GRPO group: 16 responses, 4 correct (target=0), 12 incorrect (varied targets)
# GRPO normalizes advantages: mean=0, std=1 within group
n_correct = 4
n_incorrect = 12
n_total = n_correct + n_incorrect

# Correct responses get target=0 (the right answer)
# Incorrect responses get various targets (wrong tokens the model generated)
# Higher-probability tokens more likely to be generated
incorrect_targets = [1, 2, 3, 5, 7, 10, 15, 20, 1, 3, 5, 10]

# GRPO advantages: correct get A>0, incorrect get A<0, mean=0
reward_correct = 1.0
reward_incorrect = 0.0
mean_reward = (n_correct * reward_correct + n_incorrect * reward_incorrect) / n_total
std_reward = np.sqrt((n_correct * (reward_correct - mean_reward)**2 +
                       n_incorrect * (reward_incorrect - mean_reward)**2) / n_total)
A_correct = (reward_correct - mean_reward) / (std_reward + 1e-8)
A_incorrect = (reward_incorrect - mean_reward) / (std_reward + 1e-8)
print(f"GRPO advantages: correct A={A_correct:.4f}, incorrect A={A_incorrect:.4f}")
print(f"Sum check: {n_correct*A_correct + n_incorrect*A_incorrect:.6f} (should be ~0)")
print()

for sim_label, use_surp_redist in [("Standard (∝ p)", False), ("Modified (∝ -log p, α=0.3)", True)]:
    logits = logits_init.clone()
    H_history = []
    kl_history = []

    for step in range(n_steps):
        p = torch.softmax(logits, dim=-1)
        H_history.append(entropy(p))
        kl_history.append(kl_div(p, p_init))

        total_grad = torch.zeros_like(logits)

        # Correct responses (A > 0, target = 0)
        for _ in range(n_correct):
            tgt = 0
            A = A_correct
            g = A * p.clone()
            g[tgt] = A * (p[tgt].item() - 1.0)

            if use_surp_redist:
                surp = -torch.log(p + 1e-30)
                surp_total = surp.sum() - surp[tgt] + 1e-10
                G_total = g.sum() - g[tgt]
                g_desired = G_total * surp / surp_total
                g_desired[tgt] = 0.0
                g = (1 - alpha) * g + alpha * g_desired
                g_desired_tgt = A * (p[tgt].item() - 1.0)
                g[tgt] = g_desired_tgt

            total_grad += g

        # Incorrect responses (A < 0, various targets)
        for tgt in incorrect_targets:
            A = A_incorrect
            g = A * p.clone()
            g[tgt] = A * (p[tgt].item() - 1.0)

            if use_surp_redist:
                surp = -torch.log(p + 1e-30)
                surp_total = surp.sum() - surp[tgt] + 1e-10
                G_total = g.sum() - g[tgt]
                g_desired = G_total * surp / surp_total
                g_desired[tgt] = 0.0
                g = (1 - alpha) * g + alpha * g_desired
                g_desired_tgt = A * (p[tgt].item() - 1.0)
                g[tgt] = g_desired_tgt

            total_grad += g

        logits = logits - lr * total_grad / n_total

    print(f"\n{sim_label}:")
    for s in [0, 10, 20, 50, 100, 150, 199]:
        if s < len(H_history):
            print(f"  Step {s:>3}: H={H_history[s]:.4f}, KL={kl_history[s]:.6f}")

print()

# ============================================================
# Alpha sweep with realistic simulation
# ============================================================
print("=" * 70)
print("Alpha sweep: entropy evolution with different targets")
print("=" * 70)

alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

print(f"\n{'Alpha':>8} {'H_step0':>10} {'H_step50':>10} {'H_step100':>10} "
      f"{'H_step199':>10} {'KL_final':>10}")
print("-" * 65)

for alpha_test in alphas:
    logits = logits_init.clone()
    H_at = {}

    for step in range(n_steps):
        p = torch.softmax(logits, dim=-1)
        if step in [0, 50, 100, 199]:
            H_at[step] = entropy(p)
            if step == 199:
                kl_final = kl_div(p, p_init)

        total_grad = torch.zeros_like(logits)

        for _ in range(n_correct):
            tgt = 0
            A = A_correct
            g = A * p.clone()
            g[tgt] = A * (p[tgt].item() - 1.0)

            if alpha_test > 0:
                surp = -torch.log(p + 1e-30)
                surp_total = surp.sum() - surp[tgt] + 1e-10
                G_total = g.sum() - g[tgt]
                g_desired = G_total * surp / surp_total
                g_desired[tgt] = 0.0
                g_nontgt = (1 - alpha_test) * g + alpha_test * g_desired
                g_nontgt[tgt] = g[tgt]
                total_grad += g_nontgt
            else:
                total_grad += g

        for tgt in incorrect_targets:
            A = A_incorrect
            g = A * p.clone()
            g[tgt] = A * (p[tgt].item() - 1.0)

            if alpha_test > 0:
                surp = -torch.log(p + 1e-30)
                surp_total = surp.sum() - surp[tgt] + 1e-10
                G_total = g.sum() - g[tgt]
                g_desired = G_total * surp / surp_total
                g_desired[tgt] = 0.0
                g_nontgt = (1 - alpha_test) * g + alpha_test * g_desired
                g_nontgt[tgt] = g[tgt]
                total_grad += g_nontgt
            else:
                total_grad += g

        logits = logits - lr * total_grad / n_total

    print(f"{alpha_test:8.2f} {H_at[0]:10.4f} {H_at[50]:10.4f} {H_at[100]:10.4f} "
          f"{H_at[199]:10.4f} {kl_final:10.4f}")

print()

# ============================================================
# Decompose: where does the asymmetry come from?
# ============================================================
print("=" * 70)
print("Decomposition: source of entropy asymmetry")
print("=" * 70)

logits = logits_init.clone()
p = torch.softmax(logits, dim=-1)

print(f"\nTarget probabilities:")
print(f"  Correct target (token 0): p={p[0].item():.6f}")
for tgt in sorted(set(incorrect_targets)):
    print(f"  Incorrect target (token {tgt}): p={p[tgt].item():.6f}")

print(f"\n(1 - p_a) values (determines |G_total|):")
print(f"  Correct target (token 0): (1-p_a)={1-p[0].item():.6f}")
for tgt in sorted(set(incorrect_targets)):
    print(f"  Incorrect target (token {tgt}): (1-p_a)={1-p[tgt].item():.6f}")

# Standard gradient net effect
print(f"\nNet gradient analysis (one batch):")
total_g_std = torch.zeros_like(logits)
total_g_mod = torch.zeros_like(logits)

for _ in range(n_correct):
    tgt = 0
    A = A_correct
    g = A * p.clone()
    g[tgt] = A * (p[tgt].item() - 1.0)
    total_g_std += g

    surp = -torch.log(p + 1e-30)
    surp_total = surp.sum() - surp[tgt] + 1e-10
    G_total = g.sum() - g[tgt]
    g_desired = G_total * surp / surp_total
    g_desired[tgt] = 0.0
    g_mod = (1 - alpha) * g + alpha * g_desired
    g_mod[tgt] = g[tgt]
    total_g_mod += g_mod

for tgt in incorrect_targets:
    A = A_incorrect
    g = A * p.clone()
    g[tgt] = A * (p[tgt].item() - 1.0)
    total_g_std += g

    surp = -torch.log(p + 1e-30)
    surp_total = surp.sum() - surp[tgt] + 1e-10
    G_total = g.sum() - g[tgt]
    g_desired = G_total * surp / surp_total
    g_desired[tgt] = 0.0
    g_mod = (1 - alpha) * g + alpha * g_desired
    g_mod[tgt] = g[tgt]
    total_g_mod += g_mod

total_g_std /= n_total
total_g_mod /= n_total

# Group by rank
ranks = torch.argsort(p, descending=True)
groups = [("Top 1-10", 0, 10), ("Top 11-50", 10, 50),
          ("Top 51-200", 50, 200), ("Top 201-1000", 200, 1000),
          ("Tail 1001+", 1000, V)]

print(f"\n{'Group':>15} {'Σg_std':>12} {'Σg_mod':>12} {'Diff':>12} {'Direction':>15}")
print("-" * 75)

for name, s, e in groups:
    idx = ranks[s:min(e, V)]
    sum_std = total_g_std[idx].sum().item()
    sum_mod = total_g_mod[idx].sum().item()
    diff = sum_mod - sum_std
    direction = "more ↓" if diff > 0 else "less ↓ / more ↑"
    print(f"{name:>15} {sum_std:+12.6f} {sum_mod:+12.6f} {diff:+12.6f} {direction:>15}")

print(f"\n  Total gradient (should be 0): std={total_g_std.sum():.2e}, mod={total_g_mod.sum():.2e}")

# The KEY: in standard gradient, positive and negative effects cancel for non-target tokens
# that are NOT any sample's target. But in -log(p) redistribution, they DON'T perfectly cancel
# because different targets produce different surp_total normalization constants.
print()
print("KEY INSIGHT:")
print("In standard gradient, for a non-target token v (not target of ANY sample):")
print("  Contribution from sample i: A_i * p_v")
print("  Sum over all samples: (Σ A_i) * p_v = 0 (since GRPO normalizes Σ A = 0)")
print("  → Perfect cancellation for non-target tokens!")
print()
print("In -log(p) redistribution, for a non-target token v:")
print("  Contribution from sample i: A_i * (1-p_{a_i}) * surp_v / surp_total_i")
print("  Note: surp_total_i DEPENDS ON which token is the target!")
print("  → Different surp_total_i values break the cancellation!")
print("  → Net gradient is NON-ZERO even though Σ A_i = 0")
print("  → This non-zero residual drives entropy drift!")

# Verify: compute residual gradient for a pure tail token
tail_v = ranks[-1].item()
print(f"\n  Example: token {tail_v} (deepest tail, p={p[tail_v]:.2e})")
g_residuals = []
for i, (A, tgt) in enumerate([(A_correct, 0)]*n_correct +
                               [(A_incorrect, t) for t in incorrect_targets]):
    surp = -torch.log(p + 1e-30)
    surp_total = surp.sum() - surp[tgt] + 1e-10
    G = A * (1 - p[tgt].item())
    g_v = G * surp[tail_v] / surp_total
    g_residuals.append(g_v.item())

print(f"  Per-sample grad contributions: {[f'{x:+.2e}' for x in g_residuals[:4]]}... (correct)")
print(f"                                 {[f'{x:+.2e}' for x in g_residuals[4:8]]}... (incorrect)")
print(f"  Sum: {sum(g_residuals):+.2e}")
print(f"  Standard gradient sum: {n_correct*A_correct*p[tail_v].item() + n_incorrect*A_incorrect*p[tail_v].item():+.2e}")
