"""
Verify: In standard softmax policy gradient, what proportion governs
the probability mass redistribution among non-target tokens?

Setup:
  Loss L = -A * log(π_a)   where a is the selected token, A is advantage.
  
  Logit gradient: ∂L/∂z_v = A * p_v  (v ≠ a),  ∂L/∂z_a = -A*(1-p_a)
  
  After one GD step with learning rate η:
    z_v_new = z_v - η * ∂L/∂z_v
    p_new = softmax(z_new)

We measure Δp_v = p_v_new - p_v for each non-target token v, and fit
against candidate proportions: p, -p*log(p), -log(p), uniform.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

TRAINED_PATH = "verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json"
BASELINE_PATH = "verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json"
OUTPUT_DIR = "verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy"

V = 5000

def make_zipf(s, V):
    k = np.arange(1, V + 1, dtype=np.float64)
    logp = -s * np.log(k)
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return p

def entropy_bits(p):
    m = p > 1e-30
    return -np.sum(p[m] * np.log2(p[m]))

def find_s(target_H, V, tol=0.005):
    if target_H < 0.01:
        return 100.0
    lo, hi = 0.001, 80.0
    for _ in range(120):
        mid = (lo + hi) / 2
        H = entropy_bits(make_zipf(mid, V))
        if H > target_H:
            lo = mid
        else:
            hi = mid
        if abs(H - target_H) < tol:
            break
    return mid

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def simulate_one_step(p, target_idx, A, eta):
    """Simulate one gradient step and return actual probability changes."""
    z = np.log(p + 1e-30)
    
    grad = A * p.copy()
    grad[target_idx] = -A * (1.0 - p[target_idx])
    
    z_new = z - eta * grad
    p_new = softmax(z_new)
    
    delta_p = p_new - p
    return delta_p, p_new


def compute_redistribution_fit(p, target_idx, A, eta):
    """
    Compute actual Δp and fit against candidate proportions.
    Returns share of each non-target token vs candidate shares.
    """
    delta_p, _ = simulate_one_step(p, target_idx, A, eta)
    
    mask = np.ones(len(p), dtype=bool)
    mask[target_idx] = False
    p_others = p[mask]
    dp_others = delta_p[mask]
    
    abs_dp = np.abs(dp_others)
    total_abs_dp = abs_dp.sum()
    if total_abs_dp < 1e-30:
        return None
    
    actual_share = abs_dp / total_abs_dp
    
    # Candidate proportions (normalized to sum to 1)
    cand_p = p_others / p_others.sum()
    
    neg_plogp = -p_others * np.log(p_others + 1e-30)
    cand_plogp = neg_plogp / neg_plogp.sum()
    
    neg_logp = -np.log(p_others + 1e-30)
    cand_logp = neg_logp / neg_logp.sum()
    
    cand_uniform = np.ones_like(p_others) / len(p_others)
    
    def r_squared(actual, predicted):
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        if ss_tot < 1e-30:
            return 1.0
        return 1.0 - ss_res / ss_tot
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    
    def kl_div(actual, predicted):
        m = actual > 1e-30
        return np.sum(actual[m] * np.log(actual[m] / (predicted[m] + 1e-30)))
    
    return {
        "actual_share": actual_share,
        "cand_p": cand_p,
        "cand_plogp": cand_plogp,
        "cand_logp": cand_logp,
        "cand_uniform": cand_uniform,
        "R2_p": r_squared(actual_share, cand_p),
        "R2_plogp": r_squared(actual_share, cand_plogp),
        "R2_logp": r_squared(actual_share, cand_logp),
        "R2_uniform": r_squared(actual_share, cand_uniform),
        "cos_p": cosine_sim(actual_share, cand_p),
        "cos_plogp": cosine_sim(actual_share, cand_plogp),
        "cos_logp": cosine_sim(actual_share, cand_logp),
        "cos_uniform": cosine_sim(actual_share, cand_uniform),
        "KL_p": kl_div(actual_share, cand_p),
        "KL_plogp": kl_div(actual_share, cand_plogp),
        "KL_logp": kl_div(actual_share, cand_logp),
        "KL_uniform": kl_div(actual_share, cand_uniform),
    }


# Build distributions
print("Building distributions from entropy grid...")
H_grid = np.concatenate([
    np.arange(0.125, 3.0, 0.125),
    np.arange(3.0, 7.5, 0.5)
])
dist_cache = {}
for H in H_grid:
    H_key = round(H, 3)
    s = find_s(H, V)
    dist_cache[H_key] = make_zipf(s, V)
grid_keys = sorted(dist_cache.keys())
print(f"  Built {len(dist_cache)} distributions")

# Load real entropy distribution
print("Loading real token entropies...")
def load_entropies(path):
    data = json.load(open(path))
    ents = []
    for r in data["results"]:
        for resp in r["responses"]:
            for step in resp["entropy_analysis"]["steps"]:
                ents.extend(step["token_entropies"])
    return np.array(ents)

trained_ent = load_entropies(TRAINED_PATH)
baseline_ent = load_entropies(BASELINE_PATH)

def snap(val):
    return min(grid_keys, key=lambda k: abs(k - val))

# ═══════════════════════════════════════════════════════════════════
# Part 1: Numerical verification at different entropy levels + step sizes
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART 1: R² fit of Δp redistribution to candidate proportions")
print("=" * 90)

etas = [1e-4, 1e-3, 1e-2, 0.1]

for A_label, A_val in [("POSITIVE (A=+1)", 1.0), ("NEGATIVE (A=-1)", -1.0)]:
    print(f"\n{'─'*90}")
    print(f"  {A_label}: selected token probability {'increases' if A_val > 0 else 'decreases'}")
    print(f"  Other tokens {'lose' if A_val > 0 else 'gain'} probability mass")
    print(f"{'─'*90}")
    
    for eta in etas:
        print(f"\n  η = {eta:.0e}:")
        print(f"  {'H (bits)':>8} | {'p_top1':>6} | {'R²(p)':>8} | {'R²(-plogp)':>10} | "
              f"{'R²(-logp)':>9} | {'R²(unif)':>8} | {'cos(p)':>8} | {'cos(-plogp)':>11}")
        print(f"  {'-'*85}")
        
        for H_ex in [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
            H_key = snap(H_ex)
            p = dist_cache[H_key]
            target_idx = 0  # top-1 token
            
            res = compute_redistribution_fit(p, target_idx, A_val, eta)
            if res is None:
                continue
            
            print(f"  {H_key:>8.3f} | {p[0]:>6.4f} | {res['R2_p']:>8.6f} | "
                  f"{res['R2_plogp']:>10.6f} | {res['R2_logp']:>9.6f} | "
                  f"{res['R2_uniform']:>8.6f} | {res['cos_p']:>8.6f} | "
                  f"{res['cos_plogp']:>11.6f}")

# ═══════════════════════════════════════════════════════════════════
# Part 2: Weighted average across real data
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("PART 2: Weighted average R² across ALL real token entropies")
print("=" * 90)

eta = 1e-3
for label, ent_arr in [("Trained (26%)", trained_ent), ("Baseline (4%)", baseline_ent)]:
    nonzero = ent_arr[ent_arr > 0.01]
    freq = defaultdict(int)
    for e in nonzero:
        freq[snap(e)] += 1
    total = sum(freq.values())
    
    print(f"\n  {label}: {total} non-trivial tokens (η={eta})")
    
    for A_label, A_val in [("Positive (A=+1)", 1.0), ("Negative (A=-1)", -1.0)]:
        agg = defaultdict(float)
        for H_key, count in freq.items():
            w = count / total
            p = dist_cache[H_key]
            res = compute_redistribution_fit(p, 0, A_val, eta)
            if res is None:
                continue
            for k in ["R2_p", "R2_plogp", "R2_logp", "R2_uniform",
                       "cos_p", "cos_plogp", "cos_logp", "cos_uniform",
                       "KL_p", "KL_plogp", "KL_logp", "KL_uniform"]:
                agg[k] += w * res[k]
        
        print(f"\n    {A_label}:")
        print(f"      {'Candidate':>12} | {'R²':>10} | {'Cosine Sim':>10} | {'KL div':>10}")
        print(f"      {'-'*48}")
        for name, suf in [("∝ p", "p"), ("∝ -p·log(p)", "plogp"),
                           ("∝ -log(p)", "logp"), ("uniform", "uniform")]:
            print(f"      {name:>12} | {agg[f'R2_{suf}']:>10.6f} | "
                  f"{agg[f'cos_{suf}']:>10.6f} | {agg[f'KL_{suf}']:>10.6f}")

# ═══════════════════════════════════════════════════════════════════
# Part 3: Analytical decomposition
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("PART 3: Analytical decomposition of Δp_v for non-target tokens")
print("=" * 90)
print("""
Loss: L = -A · log(p_a)

Logit gradient:
  ∂L/∂z_a = -A · (1 - p_a)     [target token]
  ∂L/∂z_v =  A · p_v           [non-target v ≠ a]

Logit update: Δz_v = -η · A · p_v

Probability change (first order via Jacobian ∂p_i/∂z_j = p_i(δ_ij - p_j)):

  Δp_v ≈ p_v · (Δz_v - E_p[Δz])
       = p_v · (-η·A·p_v - μ)

  where μ = E_p[Δz] = Σ_j p_j · Δz_j = η·A·(p_a - S₂)
  and   S₂ = Σ_j p_j²

  ∴ Δp_v ≈ -η·A · p_v · (p_v + p_a - S₂)
          = -η·A · p_v · (p_v + p_a(1-p_a) - Σ_{j≠a} p_j²)
          
  For most tokens where p_v ≪ p_a(1-p_a):
    Δp_v ≈ -η·A · p_v · p_a · (1 - p_a)   [∝ p_v to leading order]

  For the runner-up where p_v is non-negligible:
    Δp_v ∝ p_v · (p_v + const)             [slightly super-linear in p_v]
""")

# Verify this analytical formula
print("  Numerical verification of analytical formula:")
print(f"  {'H':>6} | {'p_a':>6} | {'p_a(1-p_a)':>10} | {'S₂':>8} | "
      f"{'exact vs 1st-order':>20} | {'exact vs leading':>20}")
print(f"  {'-'*80}")

eta = 1e-3
A = 1.0
for H_ex in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    H_key = snap(H_ex)
    p = dist_cache[H_key]
    p_a = p[0]
    S2 = np.sum(p ** 2)
    
    # Exact
    dp_exact, _ = simulate_one_step(p, 0, A, eta)
    dp_others_exact = dp_exact[1:]
    
    # First-order: Δp_v = -η*A * p_v * (p_v + p_a - S2)
    p_oth = p[1:]
    dp_first = -eta * A * p_oth * (p_oth + p_a - S2)
    
    # Leading order: Δp_v ≈ -η*A * p_v * p_a * (1-p_a)
    dp_leading = -eta * A * p_oth * p_a * (1 - p_a)
    
    # R² comparison
    ss_tot = np.sum((dp_others_exact - dp_others_exact.mean()) ** 2)
    r2_first = 1 - np.sum((dp_others_exact - dp_first) ** 2) / ss_tot
    r2_leading = 1 - np.sum((dp_others_exact - dp_leading) ** 2) / ss_tot
    
    print(f"  {H_key:>6.3f} | {p_a:>6.4f} | {p_a*(1-p_a):>10.6f} | {S2:>8.6f} | "
          f"R²={r2_first:>16.10f} | R²={r2_leading:>16.10f}")


# ═══════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════

# Plot 1: Redistribution share comparison at different entropy levels
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Probability Mass Redistribution: Actual Share vs Candidate Proportions\n"
             "(Positive sample A=+1, η=1e-3, target = top-1 token)", fontsize=13)

H_examples = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
for idx, H_ex in enumerate(H_examples):
    ax = axes[idx // 3, idx % 3]
    H_key = snap(H_ex)
    p = dist_cache[H_key]
    
    res = compute_redistribution_fit(p, 0, 1.0, 1e-3)
    
    show = min(100, len(res["actual_share"]))
    ranks = np.arange(2, show + 2)
    
    ax.plot(ranks, res["actual_share"][:show], "k-", lw=2, label="Actual Δp share", alpha=0.9)
    ax.plot(ranks, res["cand_p"][:show], "r--", lw=1.5, label=f"∝ p  (R²={res['R2_p']:.6f})")
    ax.plot(ranks, res["cand_plogp"][:show], "b--", lw=1.5, label=f"∝ -p·log(p)  (R²={res['R2_plogp']:.6f})")
    ax.plot(ranks, res["cand_logp"][:show], "g--", lw=1.5, label=f"∝ -log(p)  (R²={res['R2_logp']:.6f})")
    
    ax.set_title(f"H = {H_ex:.2f} bits  (p₁ = {p[0]:.4f})")
    ax.set_xlabel("Token rank")
    ax.set_ylabel("Share of |Δp|")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/redistribution_proportion_verification.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_DIR}/redistribution_proportion_verification.png")


# Plot 2: R² vs entropy for each candidate
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("How Well Does Each Candidate Proportion Explain the Redistribution?", fontsize=13)

for A_label, A_val, ax in [("Positive (A=+1)", 1.0, ax2a), ("Negative (A=-1)", -1.0, ax2b)]:
    eta = 1e-3
    for name, suf, color, ls in [("∝ p", "p", "red", "-"),
                                   ("∝ -p·log(p)", "plogp", "blue", "--"),
                                   ("∝ -log(p)", "logp", "green", "-."),
                                   ("uniform", "uniform", "gray", ":")]:
        r2s = []
        for H_key in H_grid:
            H_k = round(H_key, 3)
            if H_k not in dist_cache:
                continue
            res = compute_redistribution_fit(dist_cache[H_k], 0, A_val, eta)
            if res is None:
                r2s.append(np.nan)
                continue
            r2s.append(res[f"R2_{suf}"])
        
        valid_H = [round(h, 3) for h in H_grid if round(h, 3) in dist_cache]
        ax.plot(valid_H[:len(r2s)], r2s, color=color, ls=ls, lw=2, label=name)
    
    ax.set_title(f"{A_label}")
    ax.set_xlabel("Token entropy H (bits)")
    ax.set_ylabel("R² (goodness of fit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/redistribution_R2_vs_entropy.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/redistribution_R2_vs_entropy.png")


# Plot 3: Scatter plot actual vs predicted at a representative entropy
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 4.5))
fig3.suptitle("Scatter: Actual Δp Share vs Candidate Proportion (H=1.0 bit, A=+1, η=1e-3)", fontsize=12)

H_key = snap(1.0)
p = dist_cache[H_key]
res = compute_redistribution_fit(p, 0, 1.0, 1e-3)

for i, (name, key, color) in enumerate([("∝ p", "cand_p", "red"),
                                          ("∝ -p·log(p)", "cand_plogp", "blue"),
                                          ("∝ -log(p)", "cand_logp", "green"),
                                          ("uniform", "cand_uniform", "gray")]):
    ax = axes3[i]
    show = min(500, len(res["actual_share"]))
    ax.scatter(res[key][:show], res["actual_share"][:show], s=3, alpha=0.5, color=color)
    
    lims = [0, max(res["actual_share"][:show].max(), res[key][:show].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="perfect fit")
    
    r2_key = "R2_" + key.replace("cand_", "")
    ax.set_title(f"{name}\nR²={res[r2_key]:.6f}")
    ax.set_xlabel(f"Candidate share")
    ax.set_ylabel("Actual |Δp| share")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/redistribution_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/redistribution_scatter.png")


# Plot 4: Step size sensitivity
fig4, axes4 = plt.subplots(1, 4, figsize=(20, 4.5))
fig4.suptitle("Effect of Step Size η on Redistribution Fit (H=1.0 bit, A=+1)", fontsize=12)

for i, eta in enumerate([1e-4, 1e-3, 1e-2, 0.1]):
    ax = axes4[i]
    H_key = snap(1.0)
    p = dist_cache[H_key]
    res = compute_redistribution_fit(p, 0, 1.0, eta)
    
    show = min(100, len(res["actual_share"]))
    ranks = np.arange(2, show + 2)
    
    ax.plot(ranks, res["actual_share"][:show], "k-", lw=2, label="Actual")
    ax.plot(ranks, res["cand_p"][:show], "r--", lw=1.5, label=f"∝p (R²={res['R2_p']:.4f})")
    ax.plot(ranks, res["cand_plogp"][:show], "b--", lw=1.5, label=f"-plogp (R²={res['R2_plogp']:.4f})")
    
    ax.set_title(f"η = {eta:.0e}")
    ax.set_xlabel("Token rank")
    ax.set_ylabel("Share")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/redistribution_stepsize_sensitivity.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/redistribution_stepsize_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════
# FINAL ANSWER
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Standard softmax policy gradient probability redistribution:

  ∂L/∂z_v = A · p_v   for non-target token v

  → Logit update: Δz_v = -η·A · p_v          (exactly ∝ p_v)
  → Prob change:  Δp_v ≈ -η·A · p_v · (p_v + p_a - S₂)
                       ≈ -η·A · p_v · p_a·(1-p_a)   for small p_v

ANSWER: The redistribution proportion is ∝ p_v (probability).

  - NOT ∝ -p·log(p) (entropy contribution)
  - NOT ∝ -log(p) (surprisal)
  - NOT uniform

This means:
  Positive samples (A > 0): high-probability runners-up lose the MOST mass
                             (they are NOT protected)
  Negative samples (A < 0): high-probability runners-up gain the MOST mass
                             (they are enhanced ∝ p)

The -p·log(p) proportion would arise if entropy regularization were added.
The SurprisalRedistribution modifies positive samples to use ∝ -log(p) instead.
""")
