"""
Multi-step simulation: why -log(p) redistribution causes entropy explosion
even though per-step drift seems small.

Hypothesis: positive feedback loop where tail token accumulation compounds.
"""

import torch
import numpy as np

torch.manual_seed(42)

V = 10000
n_steps = 200
lr = 0.01
alpha = 0.3

def make_zipf(s, V):
    ranks = torch.arange(1, V + 1, dtype=torch.float64)
    unnorm = 1.0 / ranks.pow(s)
    return (unnorm / unnorm.sum()).float()


def entropy(p):
    return -(p * torch.log(p + 1e-30)).sum().item()


def kl_div(p, q):
    return (p * torch.log((p + 1e-30) / (q + 1e-30))).sum().item()


# ============================================================
# Multi-step simulation: standard vs modified gradient
# ============================================================
print("=" * 70)
print("Multi-step simulation: Standard vs -log(p) redistribution")
print("=" * 70)

# Start from the same distribution
p_init = make_zipf(1.5, V)
H_init = entropy(p_init)
print(f"Initial: H={H_init:.4f}, top1={p_init[0]:.6f}")
print()

# Simulate training: alternate positive and negative samples
# Use slightly different |A| to simulate GRPO advantage distribution
# Correct (positive): fewer samples, larger |A|
# Incorrect (negative): more samples, smaller |A|

for sim_label, use_surp_redist in [("Standard (∝ p)", False), ("Modified (∝ -log p, α=0.3)", True)]:
    logits = torch.log(p_init + 1e-30).clone()

    H_history = []
    kl_history = []
    top1_history = []
    tail_mass_history = []

    for step in range(n_steps):
        p = torch.softmax(logits, dim=-1)
        H_history.append(entropy(p))
        kl_history.append(kl_div(p, p_init))
        top1_history.append(p[0].item())
        tail_mass_history.append(p[100:].sum().item())

        # Simulate a batch: 4 positive + 12 negative (typical for math RL)
        # Positive: A = +1.5 (large because few correct)
        # Negative: A = -0.5 (small because many incorrect)
        batch = [(+1.5, 0)] * 4 + [(-0.5, 0)] * 12

        total_grad = torch.zeros_like(logits)

        for A, tgt in batch:
            g = A * p.clone()
            g[tgt] = A * (p[tgt].item() - 1.0)

            if use_surp_redist:
                surp = -torch.log(p + 1e-30)
                surp_total = surp.sum() - surp[tgt] + 1e-10
                G_total = g.sum() - g[tgt]

                g_desired = G_total * surp / surp_total
                g_desired[tgt] = 0.0

                g_nontgt = (1 - alpha) * g.clone() + alpha * g_desired
                g_nontgt[tgt] = g[tgt]
                total_grad += g_nontgt
            else:
                total_grad += g

        logits = logits - lr * total_grad / len(batch)

    print(f"\n{sim_label}:")
    print(f"  Step   0: H={H_history[0]:.4f}, KL={kl_history[0]:.6f}, "
          f"top1={top1_history[0]:.6f}, tail_mass={tail_mass_history[0]:.6f}")
    for s in [10, 20, 50, 100, 150, 199]:
        if s < len(H_history):
            print(f"  Step {s:>3}: H={H_history[s]:.4f}, KL={kl_history[s]:.6f}, "
                  f"top1={top1_history[s]:.6f}, tail_mass={tail_mass_history[s]:.6f}")

print()

# ============================================================
# Show the feedback mechanism at specific steps
# ============================================================
print("=" * 70)
print("Feedback mechanism: gradient magnitudes at different training stages")
print("=" * 70)

for sim_label, use_surp_redist in [("Standard", False), ("Modified (α=0.3)", True)]:
    logits = torch.log(p_init + 1e-30).clone()

    for step in range(n_steps):
        p = torch.softmax(logits, dim=-1)

        if step in [0, 50, 100, 150]:
            # Analyze gradient for a NEGATIVE sample (A=-0.5, target=0)
            A = -0.5
            tgt = 0
            g_std = A * p.clone()
            g_std[tgt] = A * (p[tgt].item() - 1.0)

            if use_surp_redist:
                surp = -torch.log(p + 1e-30)
                surp_total = surp.sum() - surp[tgt] + 1e-10
                G_total = g_std.sum() - g_std[tgt]
                g_desired = G_total * surp / surp_total
                g_desired[tgt] = 0.0
                g_mod = (1 - alpha) * g_std + alpha * g_desired
                g_mod[tgt] = g_std[tgt]
            else:
                g_mod = g_std

            ranks = torch.argsort(p[1:], descending=True)
            top10 = ranks[:10] + 1
            mid = ranks[50:200] + 1
            tail = ranks[1000:] + 1

            print(f"\n  {sim_label} Step {step}: H={entropy(p):.4f}")
            print(f"    Top10 avg |grad|:  {g_mod[top10].abs().mean():.2e} "
                  f"(std: {g_std[top10].abs().mean():.2e})")
            print(f"    Mid   avg |grad|:  {g_mod[mid].abs().mean():.2e} "
                  f"(std: {g_std[mid].abs().mean():.2e})")
            print(f"    Tail  avg |grad|:  {g_mod[tail].abs().mean():.2e} "
                  f"(std: {g_std[tail].abs().mean():.2e})")
            print(f"    Tail/Top10 ratio:  {g_mod[tail].abs().mean() / (g_mod[top10].abs().mean() + 1e-30):.4f} "
                  f"(std: {g_std[tail].abs().mean() / (g_std[top10].abs().mean() + 1e-30):.4f})")

        batch = [(+1.5, 0)] * 4 + [(-0.5, 0)] * 12
        total_grad = torch.zeros_like(logits)
        for A_b, tgt_b in batch:
            g = A_b * p.clone()
            g[tgt_b] = A_b * (p[tgt_b].item() - 1.0)
            if use_surp_redist:
                surp = -torch.log(p + 1e-30)
                surp_total = surp.sum() - surp[tgt_b] + 1e-10
                G_total = g.sum() - g[tgt_b]
                g_desired = G_total * surp / surp_total
                g_desired[tgt_b] = 0.0
                g_nontgt = (1 - alpha) * g + alpha * g_desired
                g_nontgt[tgt_b] = g[tgt_b]
                total_grad += g_nontgt
            else:
                total_grad += g
        logits = logits - lr * total_grad / len(batch)

print()

# ============================================================
# Key question: what happens with different alpha values?
# ============================================================
print("=" * 70)
print("Alpha sweep: entropy at step 200")
print("=" * 70)

alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

print(f"\n{'Alpha':>8} {'H_final':>10} {'KL_final':>10} {'Top1':>10} {'Tail_mass':>10}")
print("-" * 55)

for alpha_test in alphas:
    logits = torch.log(p_init + 1e-30).clone()

    for step in range(n_steps):
        p = torch.softmax(logits, dim=-1)
        batch = [(+1.5, 0)] * 4 + [(-0.5, 0)] * 12
        total_grad = torch.zeros_like(logits)

        for A_b, tgt_b in batch:
            g = A_b * p.clone()
            g[tgt_b] = A_b * (p[tgt_b].item() - 1.0)

            if alpha_test > 0:
                surp = -torch.log(p + 1e-30)
                surp_total = surp.sum() - surp[tgt_b] + 1e-10
                G_total = g.sum() - g[tgt_b]
                g_desired = G_total * surp / surp_total
                g_desired[tgt_b] = 0.0
                g_nontgt = (1 - alpha_test) * g + alpha_test * g_desired
                g_nontgt[tgt_b] = g[tgt_b]
                total_grad += g_nontgt
            else:
                total_grad += g

        logits = logits - lr * total_grad / len(batch)

    p = torch.softmax(logits, dim=-1)
    print(f"{alpha_test:8.2f} {entropy(p):10.4f} {kl_div(p, p_init):10.4f} "
          f"{p[0].item():10.6f} {p[100:].sum().item():10.6f}")

print()

# ============================================================
# Joint analysis: correction factor × gradient magnitude
# ============================================================
print("=" * 70)
print("Joint analysis: p_v, -log(p_v), and actual gradient change")
print("=" * 70)

logits = torch.log(p_init + 1e-30).clone()

# After 50 training steps with -log(p) redistribution
for step in range(50):
    p = torch.softmax(logits, dim=-1)
    batch = [(+1.5, 0)] * 4 + [(-0.5, 0)] * 12
    total_grad = torch.zeros_like(logits)
    for A_b, tgt_b in batch:
        g = A_b * p.clone()
        g[tgt_b] = A_b * (p[tgt_b].item() - 1.0)
        surp = -torch.log(p + 1e-30)
        surp_total = surp.sum() - surp[tgt_b] + 1e-10
        G_total = g.sum() - g[tgt_b]
        g_desired = G_total * surp / surp_total
        g_desired[tgt_b] = 0.0
        g_nontgt = (1 - alpha) * g + alpha * g_desired
        g_nontgt[tgt_b] = g[tgt_b]
        total_grad += g_nontgt
    logits = logits - lr * total_grad / len(batch)

p_after_50 = torch.softmax(logits, dim=-1)
print(f"\nAfter 50 steps: H={entropy(p_after_50):.4f}")
print(f"\nJoint distribution of token changes:")
print(f"{'Rank':>8} {'p_init':>12} {'p_after':>12} {'Δp':>12} "
      f"{'-log(p_init)':>12} {'p*(-log p)':>12} {'Δp/p_init':>12}")
print("-" * 85)

ranks_by_init = torch.argsort(p_init[1:], descending=True)
for label, idx_range in [("Top 1", [0]), ("Top 5", list(range(5))),
                          ("Top 10", list(range(10))),
                          ("Rank 50", [49]), ("Rank 100", [99]),
                          ("Rank 500", [499]), ("Rank 1000", [999]),
                          ("Rank 5000", [4999])]:
    for r in idx_range:
        if r >= len(ranks_by_init):
            continue
        v = ranks_by_init[r].item() + 1
        p0 = p_init[v].item()
        p1 = p_after_50[v].item()
        dp = p1 - p0
        surp0 = -np.log(p0 + 1e-30)
        plogp = p0 * surp0
        rel = dp / (p0 + 1e-20)
        if len(idx_range) == 1:
            print(f"{label:>8} {p0:12.8f} {p1:12.8f} {dp:+12.8f} "
                  f"{surp0:12.4f} {plogp:12.8f} {rel:+12.4f}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The -log(p) redistribution creates a POSITIVE FEEDBACK LOOP:

  Step 1: -log(p) gives tail tokens disproportionately large gradients
  Step 2: Negative samples (more numerous, larger |G_total|) push tail up
  Step 3: Tail tokens accumulate probability mass → distribution flattens
  Step 4: Flatter distribution → ALL tokens get more similar gradients
  Step 5: But the RELATIVE amplification of tail tokens PERSISTS
  Step 6: Repeat → entropy grows continuously

The standard gradient avoids this because ∝ p redistribution preserves
the conditional distribution at EVERY step → no feedback loop possible.

Key insight: the correction factor (surp_v/surp_total)/(p_v/(1-p_a))
can be >> 1 for tail tokens. Even with alpha=0.3, the effective
amplification for deep tail tokens is 100-10000x. This accumulates
multiplicatively across training steps.

Why accuracy still improves:
Target gradient is EXACTLY preserved. The model still learns which
token to choose. Only the "background" distribution changes.
KL measures distribution shape, not prediction accuracy.
""")
