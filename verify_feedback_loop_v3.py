"""
Multi-step simulation v3: Peaked distribution (realistic LLM output).
Real model has H~0.2, meaning very peaked distributions.
This is where the correction factor becomes enormous.
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


print("=" * 70)
print("Realistic LLM: PEAKED distributions (H~0.2-0.5)")
print("=" * 70)

# Create peaked logits: one dominant token, a few runners-up, long tail
logits_init = torch.randn(V) * 0.3 - 5.0  # low base logits
logits_init[0] = 3.0   # correct token: very high probability
logits_init[1] = 0.5   # runner-up 1
logits_init[2] = 0.0   # runner-up 2
logits_init[3] = -0.5  # runner-up 3
logits_init[5] = -1.0  # mid-range
logits_init[10] = -2.0 # low
logits_init[20] = -3.0 # very low

p_init = torch.softmax(logits_init, dim=-1)
print(f"Initial: H={entropy(p_init):.4f}")
print(f"  p[0] (correct target) = {p_init[0]:.6f}")
print(f"  p[1] (runner-up 1)    = {p_init[1]:.6f}")
print(f"  p[2] (runner-up 2)    = {p_init[2]:.6f}")
print(f"  p[100] (mid)          = {p_init[100]:.6e}")
print(f"  p[1000] (tail)        = {p_init[1000]:.6e}")
print()

# GRPO: 4 correct (target=0), 12 incorrect (various targets)
incorrect_targets = [1, 2, 3, 5, 1, 2, 10, 20, 3, 1, 5, 2]
n_correct = 4
n_incorrect = 12
n_total = n_correct + n_incorrect

A_correct = 1.7321
A_incorrect = -0.5774

print(f"GRPO advantages: A_correct={A_correct:.4f}, A_incorrect={A_incorrect:.4f}")
print(f"Advantage sum: {n_correct * A_correct + n_incorrect * A_incorrect:.6f}")
print()

for sim_label, use_surp_redist in [("Standard (∝ p)", False),
                                     ("Modified (∝ -log p, α=0.3)", True)]:
    logits = logits_init.clone()
    H_hist = []
    kl_hist = []
    p0_hist = []
    tail_hist = []

    for step in range(n_steps):
        p = torch.softmax(logits, dim=-1)
        H_hist.append(entropy(p))
        kl_hist.append(kl_div(p, p_init))
        p0_hist.append(p[0].item())
        tail_hist.append(p[100:].sum().item())

        total_grad = torch.zeros_like(logits)

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
                g_final = (1 - alpha) * g + alpha * g_desired
                g_final[tgt] = g[tgt]
                total_grad += g_final
            else:
                total_grad += g

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
                g_final = (1 - alpha) * g + alpha * g_desired
                g_final[tgt] = g[tgt]
                total_grad += g_final
            else:
                total_grad += g

        logits = logits - lr * total_grad / n_total

    print(f"{sim_label}:")
    for s in [0, 10, 20, 50, 100, 150, 199]:
        if s < len(H_hist):
            print(f"  Step {s:>3}: H={H_hist[s]:.4f}, KL={kl_hist[s]:.6f}, "
                  f"p_correct={p0_hist[s]:.6f}, tail_mass={tail_hist[s]:.6e}")
    print()

# ============================================================
# Alpha sweep
# ============================================================
print("=" * 70)
print("Alpha sweep with peaked distribution")
print("=" * 70)

alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

print(f"\n{'Alpha':>8} {'H_0':>8} {'H_50':>8} {'H_100':>8} {'H_199':>8} "
      f"{'KL_199':>10} {'p0_199':>10} {'Tail_199':>10}")
print("-" * 90)

for alpha_test in alphas:
    logits = logits_init.clone()
    H_at = {}
    for step in range(n_steps):
        p = torch.softmax(logits, dim=-1)
        if step in [0, 50, 100, 199]:
            H_at[step] = entropy(p)
            if step == 199:
                kl_f = kl_div(p, p_init)
                p0_f = p[0].item()
                tail_f = p[100:].sum().item()

        total_grad = torch.zeros_like(logits)
        for _ in range(n_correct):
            tgt = 0
            A = A_correct
            g = A * p.clone()
            g[tgt] = A * (p[tgt].item() - 1.0)
            if alpha_test > 0:
                surp = -torch.log(p + 1e-30)
                st = surp.sum() - surp[tgt] + 1e-10
                Gt = g.sum() - g[tgt]
                gd = Gt * surp / st; gd[tgt] = 0
                g = (1 - alpha_test) * g + alpha_test * gd
                g[tgt] = A * (p[tgt].item() - 1.0)
            total_grad += g

        for tgt in incorrect_targets:
            A = A_incorrect
            g = A * p.clone()
            g[tgt] = A * (p[tgt].item() - 1.0)
            if alpha_test > 0:
                surp = -torch.log(p + 1e-30)
                st = surp.sum() - surp[tgt] + 1e-10
                Gt = g.sum() - g[tgt]
                gd = Gt * surp / st; gd[tgt] = 0
                g = (1 - alpha_test) * g + alpha_test * gd
                g[tgt] = A * (p[tgt].item() - 1.0)
            total_grad += g

        logits = logits - lr * total_grad / n_total

    print(f"{alpha_test:8.2f} {H_at[0]:8.4f} {H_at[50]:8.4f} {H_at[100]:8.4f} "
          f"{H_at[199]:8.4f} {kl_f:10.4f} {p0_f:10.6f} {tail_f:10.6e}")

print()

# ============================================================
# Residual gradient analysis: WHY does cancellation break?
# ============================================================
print("=" * 70)
print("Residual gradient analysis (why cancellation breaks)")
print("=" * 70)

p = torch.softmax(logits_init, dim=-1)
surp_full = -torch.log(p + 1e-30)

# For each non-target token, compute residual gradient
# Residual = Σ_i g_v_modified_i - Σ_i g_v_standard_i (should be 0 for standard)
residual_mod = torch.zeros(V)

for _ in range(n_correct):
    tgt = 0
    A = A_correct
    G = A * (1 - p[tgt].item())
    surp_total_i = surp_full.sum() - surp_full[tgt] + 1e-10

    for v in range(V):
        if v == tgt:
            continue
        g_std = A * p[v].item()
        g_mod = (1 - alpha) * g_std + alpha * G * surp_full[v].item() / surp_total_i.item()
        residual_mod[v] += (g_mod - g_std)

for tgt in incorrect_targets:
    A = A_incorrect
    G = A * (1 - p[tgt].item())
    surp_total_i = surp_full.sum() - surp_full[tgt] + 1e-10

    for v in range(V):
        if v == tgt:
            continue
        g_std = A * p[v].item()
        g_mod = (1 - alpha) * g_std + alpha * G * surp_full[v].item() / surp_total_i.item()
        residual_mod[v] += (g_mod - g_std)

# Exclude target tokens from analysis
non_target_mask = torch.ones(V, dtype=torch.bool)
non_target_mask[0] = False
for t in incorrect_targets:
    non_target_mask[t] = False

residual_pure = residual_mod[non_target_mask]
p_pure = p[non_target_mask]
surp_pure = surp_full[non_target_mask]

print(f"\nResidual gradient (modified - standard) for non-target tokens:")
print(f"  Mean residual:   {residual_pure.mean():.6e}")
print(f"  Std residual:    {residual_pure.std():.6e}")
print(f"  Max |residual|:  {residual_pure.abs().max():.6e}")
print(f"  Sum residual:    {residual_pure.sum():.6e}")
print()

# Correlation: residual vs surp_v
corr = torch.corrcoef(torch.stack([residual_pure, surp_pure]))[0, 1].item()
print(f"  Correlation(residual, -log p):  {corr:.6f}")
corr_p = torch.corrcoef(torch.stack([residual_pure, p_pure]))[0, 1].item()
print(f"  Correlation(residual, p):       {corr_p:.6f}")
print()

# Show by rank
ranks = torch.argsort(p, descending=True)
groups = [("Top 1-10", 0, 10), ("Top 11-50", 10, 50),
          ("Top 51-200", 50, 200), ("Top 201-1000", 200, 1000),
          ("Tail 1001+", 1000, V)]

print(f"{'Group':>15} {'Mean Residual':>14} {'Mean -log(p)':>14} {'Mean p':>14}")
print("-" * 65)
for name, s, e in groups:
    idx = ranks[s:min(e, V)]
    mr = residual_mod[idx].mean().item()
    ms = surp_full[idx].mean().item()
    mp = p[idx].mean().item()
    print(f"{name:>15} {mr:+14.6e} {ms:14.4f} {mp:14.6e}")

print()
print("Negative residual → logit INCREASES → probability INCREASES")
print("The tail has the largest negative residual → tail probability grows!")
print()

# ============================================================
# Why does this happen? surp_total varies with target
# ============================================================
print("=" * 70)
print("Root cause: surp_total varies with target token")
print("=" * 70)

surp_totals = []
for tgt in [0] + list(set(incorrect_targets)):
    st = surp_full.sum() - surp_full[tgt]
    surp_totals.append((tgt, st.item(), surp_full[tgt].item()))

print(f"\n{'Target':>8} {'surp_total':>14} {'surp_target':>14} {'p_target':>14}")
print("-" * 55)
for tgt, st, s_tgt in sorted(surp_totals, key=lambda x: x[0]):
    print(f"{tgt:>8} {st:14.4f} {s_tgt:14.4f} {p[tgt].item():14.6e}")

print(f"\n  surp_total range: {min(s[1] for s in surp_totals):.4f} to {max(s[1] for s in surp_totals):.4f}")
print(f"  Variation: {(max(s[1] for s in surp_totals) - min(s[1] for s in surp_totals)):.4f}")
print()
print("When target=0 (high p, low surp): surp_total is LARGER (less excluded)")
print("When target=20 (low p, high surp): surp_total is SMALLER (more excluded)")
print()
print("For a tail token v:")
print("  From correct sample (tgt=0):   g_v = α * A_pos * (1-p_0) * surp_v / surp_total_LARGE")
print("  From incorrect sample (tgt=20): g_v = α * A_neg * (1-p_20) * surp_v / surp_total_SMALL")
print()
print("  The incorrect sample has BOTH:")
print("    - Larger (1-p_tgt) ≈ 1.0")
print("    - Smaller surp_total (larger per-token weight)")
print("  → Negative gradient is amplified relative to positive")
print("  → Net residual is NEGATIVE → tail probability INCREASES")
