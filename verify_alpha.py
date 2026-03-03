"""
Verify surprisal_alpha on real evaluation data.

For positive samples, the standard softmax gradient gives:
  g_v = A * p_v   (for non-target token v, redistribution ∝ p)

SurprisalRedistribution replaces this with:
  g_v = A * [alpha * (-log p_v)/C + (1-alpha) * p_v]   (blend toward ∝ -log p)

where C = sum_{j!=a} (-log p_j) / (1 - p_a), ensuring L1 gradient conservation.

This script builds Zipf distributions matching each real token's entropy,
then computes key metrics at different alpha values.
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

def compute_metrics(p, alpha):
    """Compute metrics for a distribution and alpha. Target = token 0."""
    p_a = p[0]
    p_oth = p[1:].copy()
    surp_oth = -np.log(p_oth + 1e-30)  # nats
    C = surp_oth.sum() / (1.0 - p_a + 1e-10)

    # Standard vs blended gradient for non-target tokens
    g_std = p_oth  # standard: ∝ p
    g_surp = surp_oth / C  # surprisal: ∝ -log(p)
    g_blend = alpha * g_surp + (1 - alpha) * g_std

    # 1. Cosine similarity → rotation angle
    dot = np.dot(g_std, g_blend)
    n1, n2 = np.linalg.norm(g_std), np.linalg.norm(g_blend)
    cos_sim = dot / (n1 * n2 + 1e-30)
    angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

    # 2. L1 conservation: sum(g_blend) / sum(g_std) — should be ≈ 1.0
    l1_ratio = g_blend.sum() / (g_std.sum() + 1e-30)

    # 3. L2 norm ratio: measures how spread out the gradient becomes
    l2_ratio = n2 / (n1 + 1e-30)

    # 4. Runner-up protection: gradient change for #2 token
    prot_2 = g_blend[0] / (g_std[0] + 1e-30)

    # 5. Top-10 aggregate: fraction of total gradient going to top-10
    top_k = min(10, len(p_oth))
    frac_std_top10 = g_std[:top_k].sum() / (g_std.sum() + 1e-30)
    frac_blend_top10 = g_blend[:top_k].sum() / (g_blend.sum() + 1e-30)

    # 6. Tail aggregate: fraction of total gradient going to bottom 90%
    n_tail = int(0.9 * len(p_oth))
    tail_start = len(p_oth) - n_tail
    frac_std_tail = g_std[tail_start:].sum() / (g_std.sum() + 1e-30)
    frac_blend_tail = g_blend[tail_start:].sum() / (g_blend.sum() + 1e-30)

    # 7. Gini coefficient of gradient (0=equal, 1=concentrated)
    def gini(x):
        x_sorted = np.sort(x)
        n = len(x_sorted)
        cum = np.cumsum(x_sorted)
        return 1 - 2 * cum.sum() / (n * cum[-1] + 1e-30)
    gini_std = gini(g_std)
    gini_blend = gini(g_blend)

    return {
        "angle": angle,
        "cos_sim": cos_sim,
        "l1_ratio": l1_ratio,
        "l2_ratio": l2_ratio,
        "prot_2": prot_2,
        "frac_std_top10": frac_std_top10,
        "frac_blend_top10": frac_blend_top10,
        "frac_std_tail": frac_std_tail,
        "frac_blend_tail": frac_blend_tail,
        "gini_std": gini_std,
        "gini_blend": gini_blend,
    }


print("Building Zipf distributions for entropy grid...")
H_grid = np.concatenate([
    np.arange(0.125, 3.0, 0.125),
    np.arange(3.0, 7.5, 0.5)
])
dist_cache = {}
for H in H_grid:
    H_key = round(H, 3)
    s = find_s(H, V)
    dist_cache[H_key] = make_zipf(s, V)
print(f"  Built {len(dist_cache)} distributions")

print("Loading evaluation data...")
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
print(f"  Trained: {len(trained_ent)} tokens, Baseline: {len(baseline_ent)} tokens")

grid_keys = sorted(dist_cache.keys())
def snap(val):
    return min(grid_keys, key=lambda k: abs(k - val))

alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# ─── Compute weighted summary ───
print("\nComputing metrics weighted by real token entropy distribution...\n")

all_summaries = {}
for label, ent_arr in [("Trained (26%)", trained_ent), ("Baseline (4%)", baseline_ent)]:
    nonzero = ent_arr[ent_arr > 0.01]
    freq = defaultdict(int)
    for e in nonzero:
        freq[snap(e)] += 1
    total = sum(freq.values())

    print(f"{'='*80}")
    print(f"  {label}  —  {total} non-trivial tokens (H > 0)")
    print(f"{'='*80}")
    print(f"{'α':>5} | {'angle°':>7} | {'cos_sim':>7} | {'L1 ratio':>8} | {'L2 ratio':>8} | "
          f"{'prot #2':>7} | {'top10 frac':>10} | {'tail frac':>10} | {'Gini':>5}")
    print("-" * 95)

    summary = {}
    for alpha in alphas:
        agg = defaultdict(float)
        for H_key, count in freq.items():
            w = count / total
            m = compute_metrics(dist_cache[H_key], alpha)
            for k, v in m.items():
                agg[k] += w * v

        print(f"{alpha:>5.1f} | {agg['angle']:>7.2f} | {agg['cos_sim']:>7.4f} | "
              f"{agg['l1_ratio']:>8.4f} | {agg['l2_ratio']:>8.4f} | "
              f"{agg['prot_2']:>7.4f} | "
              f"{agg['frac_blend_top10']:>10.4f} | "
              f"{agg['frac_blend_tail']:>10.4f} | "
              f"{agg['gini_blend']:>5.3f}")
        summary[alpha] = dict(agg)
    all_summaries[label] = summary

    print(f"\n  Standard (α=0): top-10 gets {summary[0]['frac_std_top10']*100:.1f}% of gradient, "
          f"tail(90%) gets {summary[0]['frac_std_tail']*100:.1f}%")
    for a in [0.3, 0.5, 1.0]:
        s = summary[a]
        angle = s['angle']
        top10_shift = (s['frac_blend_top10'] - summary[0]['frac_std_top10']) * 100
        tail_shift  = (s['frac_blend_tail'] - summary[0]['frac_std_tail']) * 100
        prot = (1 - s['prot_2']) * 100
        print(f"  α={a}: rotation {angle:.1f}°, #2 protected {prot:.1f}%, "
              f"top10 share {top10_shift:+.1f}pp, tail share {tail_shift:+.1f}pp")
    print()

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: Summary vs alpha
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("SurprisalRedistribution: Effect of α (weighted by real token entropy)", fontsize=14)

for label, ent_arr, color, ls in [("Trained (26%)", trained_ent, "tab:red", "-"),
                                   ("Baseline (4%)", baseline_ent, "tab:blue", "--")]:
    nonzero = ent_arr[ent_arr > 0.01]
    freq = defaultdict(int)
    for e in nonzero:
        freq[snap(e)] += 1
    total = sum(freq.values())

    vals = {k: [] for k in ["angle", "cos_sim", "l2_ratio", "prot_2",
                              "frac_blend_top10", "frac_blend_tail"]}
    for alpha in alphas:
        agg = defaultdict(float)
        for H_key, count in freq.items():
            w = count / total
            m = compute_metrics(dist_cache[H_key], alpha)
            for k, v in m.items():
                agg[k] += w * v
        for k in vals:
            vals[k].append(agg[k])

    axes[0,0].plot(alphas, vals["angle"], f"{ls}", color=color, marker="o", label=label)
    axes[0,1].plot(alphas, vals["cos_sim"], f"{ls}", color=color, marker="o", label=label)
    axes[0,2].plot(alphas, vals["l2_ratio"], f"{ls}", color=color, marker="o", label=label)
    axes[1,0].plot(alphas, vals["prot_2"], f"{ls}", color=color, marker="o", label=label)
    axes[1,1].plot(alphas, vals["frac_blend_top10"], f"{ls}", color=color, marker="o", label=label)
    axes[1,2].plot(alphas, vals["frac_blend_tail"], f"{ls}", color=color, marker="o", label=label)

titles = ["Gradient Rotation (°)", "Cosine Sim (std↔blend)", "L2 Norm Ratio",
          "#2 Protection (< 1 = protected)", "Top-10 Gradient Share", "Tail(90%) Gradient Share"]
ylabs  = ["rotation (°)", "cos similarity", "||g_new||/||g_std||",
          "correction factor", "fraction of total grad", "fraction of total grad"]
for i, ax in enumerate(axes.flat):
    ax.set_title(titles[i]); ax.set_ylabel(ylabs[i]); ax.set_xlabel("α")
    ax.legend(); ax.grid(True, alpha=0.3)

axes[1,0].axhline(1.0, color="black", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/surprisal_alpha_verification_summary.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/surprisal_alpha_verification_summary.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: Metrics vs token entropy, for selected alphas
# ══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle("Metrics vs Token Entropy at Different α", fontsize=14)

sel_alphas = [0.1, 0.3, 0.5, 1.0]
cmap = plt.cm.viridis
ca = {a: cmap(i / max(1, len(sel_alphas) - 1)) for i, a in enumerate(sel_alphas)}

H_plot = sorted(dist_cache.keys())
for alpha in sel_alphas:
    ms = [compute_metrics(dist_cache[h], alpha) for h in H_plot]
    c = ca[alpha]
    axes2[0,0].plot(H_plot, [m["angle"] for m in ms], color=c, label=f"α={alpha}")
    axes2[0,1].plot(H_plot, [m["cos_sim"] for m in ms], color=c, label=f"α={alpha}")
    axes2[0,2].plot(H_plot, [m["l2_ratio"] for m in ms], color=c, label=f"α={alpha}")
    axes2[1,0].plot(H_plot, [m["prot_2"] for m in ms], color=c, label=f"α={alpha}")
    axes2[1,1].plot(H_plot, [m["frac_blend_top10"] for m in ms], color=c, label=f"α={alpha}")
    axes2[1,2].plot(H_plot, [m["frac_blend_tail"] for m in ms], color=c, label=f"α={alpha}")

# Overlay entropy histogram on first subplot
ax_t = axes2[0,0].twinx()
bins = np.arange(0, 7.5, 0.125)
h_vals, _ = np.histogram(trained_ent[trained_ent > 0.01], bins=bins)
ax_t.bar(bins[:-1] + 0.0625, h_vals, width=0.12, alpha=0.15, color="gray")
ax_t.set_ylabel("token count", color="gray")

titles2 = ["Gradient Rotation (°)", "Cosine Sim", "L2 Norm Ratio",
           "#2 Protection", "Top-10 Gradient Share", "Tail Gradient Share"]
for i, ax in enumerate(axes2.flat):
    ax.set_title(titles2[i]); ax.set_xlabel("Token entropy H (bits)")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
axes2[1,0].axhline(1.0, color="black", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/surprisal_alpha_verification_vs_entropy.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/surprisal_alpha_verification_vs_entropy.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 3: Per-rank gradient redistribution profile at example H values
# ══════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle("Per-Token Gradient Profile: g_blend(rank) / g_std(rank)\n"
              "(< 1 = token protected, > 1 = token suppressed)", fontsize=14)

H_examples = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
for idx, H_ex in enumerate(H_examples):
    ax = axes3[idx // 3, idx % 3]
    H_key = snap(H_ex)
    p = dist_cache[H_key]
    p_a = p[0]; p_oth = p[1:]
    surp_oth = -np.log(p_oth + 1e-30)
    C = surp_oth.sum() / (1.0 - p_a + 1e-10)

    show = min(500, len(p_oth))
    ranks = np.arange(2, show + 2)
    g_std = p_oth[:show]

    for alpha in [0.1, 0.3, 0.5, 1.0]:
        g_blend = alpha * surp_oth[:show] / C + (1 - alpha) * p_oth[:show]
        ratio = g_blend / (g_std + 1e-30)
        ax.plot(ranks, ratio, label=f"α={alpha}", alpha=0.8)

    ax.axhline(1.0, color="black", ls="--", alpha=0.5)
    ax.set_title(f"H = {H_ex:.2f} bits  (p₁ = {p_a:.4f})")
    ax.set_xlabel("Token rank")
    ax.set_ylabel("g_blend / g_std")
    ax.set_ylim(0, 4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/surprisal_alpha_correction_profiles.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/surprisal_alpha_correction_profiles.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 4: Crossover rank analysis
# ══════════════════════════════════════════════════════════════════════════
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("Crossover Rank: Below this rank → protected, Above → suppressed", fontsize=13)

for alpha in sel_alphas:
    crossovers = []
    for H_key in H_plot:
        p = dist_cache[H_key]
        p_a = p[0]; p_oth = p[1:]
        surp_oth = -np.log(p_oth + 1e-30)
        C = surp_oth.sum() / (1.0 - p_a + 1e-10)
        g_std = p_oth
        g_blend = alpha * surp_oth / C + (1 - alpha) * p_oth
        ratio = g_blend / (g_std + 1e-30)
        cross_idx = np.where(ratio > 1.0)[0]
        crossover = cross_idx[0] + 2 if len(cross_idx) > 0 else len(p_oth) + 1
        crossovers.append(crossover)

    ax4a.plot(H_plot, crossovers, color=ca[alpha], label=f"α={alpha}")

ax4a.set_xlabel("Token entropy H (bits)")
ax4a.set_ylabel("Crossover rank")
ax4a.set_title("Crossover Rank vs Entropy")
ax4a.legend()
ax4a.grid(True, alpha=0.3)
ax4a.set_yscale("log")

# Protection strength at #2 and #5
for rank_offset, ls, marker in [(0, "-", "o"), (4, "--", "s")]:
    for alpha in sel_alphas:
        prots = []
        for H_key in H_plot:
            p = dist_cache[H_key]
            p_a = p[0]; p_oth = p[1:]
            surp_oth = -np.log(p_oth + 1e-30)
            C = surp_oth.sum() / (1.0 - p_a + 1e-10)
            g_std_r = p_oth[rank_offset]
            g_blend_r = alpha * surp_oth[rank_offset] / C + (1 - alpha) * p_oth[rank_offset]
            prots.append(g_blend_r / (g_std_r + 1e-30))

        lbl = f"α={alpha}, rank #{rank_offset+2}"
        ax4b.plot(H_plot, prots, ls=ls, marker=marker, markersize=3,
                  color=ca[alpha], label=lbl, alpha=0.8)

ax4b.axhline(1.0, color="black", ls="--", alpha=0.4)
ax4b.set_xlabel("Token entropy H (bits)")
ax4b.set_ylabel("g_blend / g_std")
ax4b.set_title("Protection Strength for #2 and #5 Tokens")
ax4b.legend(fontsize=7, ncol=2)
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/surprisal_alpha_crossover_analysis.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/surprisal_alpha_crossover_analysis.png")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("VERIFICATION CONCLUSIONS")
print("=" * 80)

for label in all_summaries:
    s = all_summaries[label]
    print(f"\n{label}:")
    print(f"  {'α':>4} | {'rotation':>8} | {'L1 conserv':>10} | {'#2 prot':>7} | {'top10→':>6} | {'tail→':>6}")
    print(f"  {'-'*55}")
    base_t10 = s[0]['frac_std_top10']
    base_tail = s[0]['frac_std_tail']
    for a in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        t10_chg = (s[a]['frac_blend_top10'] - base_t10) * 100
        tail_chg = (s[a]['frac_blend_tail'] - base_tail) * 100
        print(f"  {a:>4.1f} | {s[a]['angle']:>7.2f}° | {s[a]['l1_ratio']:>10.6f} | "
              f"{(1-s[a]['prot_2'])*100:>6.1f}% | {t10_chg:>+5.1f}% | {tail_chg:>+5.1f}%")

print("""
Key:
  rotation  = angle between standard and modified gradient direction
  L1 conserv = sum(g_blend)/sum(g_std), should be ≈1.0 (gradient magnitude conserved)
  #2 prot   = how much less gradient the runner-up token receives (% reduction)
  top10→    = change in top-10 tokens' share of total gradient (pp)
  tail→     = change in tail tokens' share of total gradient (pp)

RECOMMENDATION:
  α=0.3 provides meaningful runner-up protection with minimal gradient rotation.
  α=0.5 doubles the effect; suitable if training is stable at α=0.3.
  α=1.0 causes near-90° rotation at common entropy levels → training collapse.
""")
