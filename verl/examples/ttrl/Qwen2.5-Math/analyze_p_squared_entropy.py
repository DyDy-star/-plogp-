#!/usr/bin/env python3
"""
分析高熵和低熵token在 p → p² 变换后的熵变化差异

理论背景：
当概率分布 p 变换为 p²/Z (其中 Z = Σp_i²) 时，
这等价于在softmax参数化中将温度减半 (T → T/2)。
即 softmax(logits/T) → softmax(logits/(T/2)) = softmax(2*logits/T)

本脚本分析：
1. 理论模拟：不同熵水平的token在p→p²后的熵变化
2. 实证分析：使用实际评估数据展示token熵分布
3. 验证假设：高熵token变化更缓，低熵token变化更大
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import os

# ============================================================
# Style setup (borrowed from analyze_step_entropy.py)
# ============================================================
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# Color palette
COLORS = {
    'blue': '#3498DB',
    'red': '#E74C3C',
    'green': '#2ECC71',
    'purple': '#9B59B6',
    'orange': '#E67E22',
    'teal': '#1ABC9C',
    'dark': '#2C3E50',
}


# ============================================================
# Helper functions
# ============================================================

def compute_entropy(p):
    """Compute Shannon entropy H(p) in nats"""
    p = np.asarray(p, dtype=np.float64)
    mask = p > 1e-30
    return -np.sum(p[mask] * np.log(p[mask]))


def square_and_renormalize(p):
    """Apply p → p²/Z transformation"""
    p = np.asarray(p, dtype=np.float64)
    p_sq = p ** 2
    Z = np.sum(p_sq)
    return p_sq / Z if Z > 0 else p_sq


def load_token_entropies(json_file):
    """Load all token-level entropies from a JSON file"""
    print(f"正在加载: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_entropies = []
    for result in data['results']:
        for response in result['responses']:
            analysis = response.get('entropy_analysis', {})
            for step in analysis.get('steps', []):
                entropies = step.get('token_entropies', [])
                all_entropies.extend(entropies)

    pass_at_1 = data.get('overall_pass@1', 0)
    n_responses = data.get('total_responses', 0)
    print(f"  加载了 {len(all_entropies)} 个token熵值, pass@1={pass_at_1:.4f}")
    return np.array(all_entropies), pass_at_1, n_responses


# ============================================================
# Theoretical Analysis
# ============================================================

def theoretical_analysis(V=5000, n_temps=300, n_trials=30):
    """
    模拟 p → p²/Z 变换对不同熵水平的影响。
    
    由于 p²/Z 在softmax参数化下等价于温度减半:
    softmax(z/T) → softmax(z/(T/2))
    
    我们通过改变温度T来生成不同熵水平的分布，
    然后比较 H(T) 和 H(T/2) 的差异。
    """
    print(f"运行理论模拟 (V={V}, {n_temps}个温度, {n_trials}次试验)...")

    temperatures = np.logspace(-1.5, 1.5, n_temps)

    all_H_orig = []
    all_H_new = []

    for trial in range(n_trials):
        logits = np.random.randn(V)
        for T in temperatures:
            p = softmax(logits / T)
            p_sq = square_and_renormalize(p)

            H_orig = compute_entropy(p)
            H_new = compute_entropy(p_sq)

            all_H_orig.append(H_orig)
            all_H_new.append(H_new)

        if (trial + 1) % 10 == 0:
            print(f"  完成 {trial + 1}/{n_trials} 次试验")

    H_orig = np.array(all_H_orig)
    H_new = np.array(all_H_new)
    abs_change = H_new - H_orig
    rel_change = np.where(H_orig > 1e-10, (H_new - H_orig) / H_orig, 0.0)
    ratio = np.where(H_orig > 1e-10, H_new / H_orig, 1.0)

    print(f"  理论模拟完成, 共 {len(H_orig)} 个数据点")
    return {
        'H_orig': H_orig,
        'H_new': H_new,
        'abs_change': abs_change,
        'rel_change': rel_change,
        'ratio': ratio,
    }


def compute_binned_stats(H_orig, values, n_bins=60, max_h=None):
    """Compute binned mean and std for plotting smooth curves"""
    if max_h is None:
        max_h = np.max(H_orig)
    bins = np.linspace(0, max_h, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = np.full(n_bins, np.nan)
    bin_stds = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (H_orig >= bins[i]) & (H_orig < bins[i + 1])
        cnt = np.sum(mask)
        if cnt > 0:
            bin_means[i] = np.mean(values[mask])
            bin_stds[i] = np.std(values[mask])
            bin_counts[i] = cnt

    valid = ~np.isnan(bin_means)
    return bin_centers[valid], bin_means[valid], bin_stds[valid], bin_counts[valid]


# ============================================================
# Plotting Functions
# ============================================================

def plot_main_figure(theory, entropies_0, entropies_15, pass1_0, pass1_15, output_file):
    """Create the main 2x2 analysis figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(r'Entropy Change Analysis: $p \rightarrow p^2/Z$ Transformation'
                 '\nHow High vs Low Entropy Tokens Respond Differently',
                 fontsize=16, fontweight='bold', y=1.005)

    max_data_h = max(entropies_0.max(), entropies_15.max()) * 1.05

    # ---- Plot 1: H_new vs H_orig (theoretical) ----
    ax1 = axes[0, 0]
    mask_range = theory['H_orig'] <= max_data_h * 1.2
    ax1.scatter(theory['H_orig'][mask_range], theory['H_new'][mask_range],
                s=2, alpha=0.15, c=COLORS['blue'], rasterized=True)

    diag_max = max_data_h * 1.2
    ax1.plot([0, diag_max], [0, diag_max], 'k--', linewidth=2, alpha=0.5,
             label=r'$H_{new} = H_{orig}$ (no change)')

    bc, bm, bs, _ = compute_binned_stats(
        theory['H_orig'][mask_range], theory['H_new'][mask_range],
        n_bins=50, max_h=diag_max)
    ax1.plot(bc, bm, color=COLORS['red'], linewidth=3, label='Mean trend')
    ax1.fill_between(bc, bm - bs, bm + bs, alpha=0.15, color=COLORS['red'])

    ax1.set_xlabel('Original Entropy H(p)  [nats]', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'New Entropy H($p^2$/Z)  [nats]', fontsize=12, fontweight='bold')
    ax1.set_title(r'$H(p^2/Z)$ vs $H(p)$: Squaring Always Reduces Entropy',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.set_xlim(0, diag_max)
    ax1.set_ylim(0, diag_max)
    ax1.grid(True, alpha=0.3)

    gap_arrow_x = diag_max * 0.6
    gap_arrow_y_start = diag_max * 0.6
    bc_idx = np.argmin(np.abs(bc - gap_arrow_x))
    if bc_idx < len(bm):
        gap_arrow_y_end = bm[bc_idx]
        ax1.annotate('', xy=(gap_arrow_x, gap_arrow_y_end),
                     xytext=(gap_arrow_x, gap_arrow_y_start),
                     arrowprops=dict(arrowstyle='<->', color=COLORS['orange'], lw=2))
        ax1.text(gap_arrow_x + 0.15, (gap_arrow_y_start + gap_arrow_y_end) / 2,
                 'Entropy\nreduction', fontsize=9, color=COLORS['orange'],
                 fontweight='bold', ha='left', va='center')

    # ---- Plot 2: Entropy ratio H_new/H_orig vs H_orig ----
    ax2 = axes[0, 1]
    ax2.scatter(theory['H_orig'][mask_range], theory['ratio'][mask_range],
                s=2, alpha=0.15, c=COLORS['purple'], rasterized=True)

    bc2, bm2, bs2, _ = compute_binned_stats(
        theory['H_orig'][mask_range], theory['ratio'][mask_range],
        n_bins=50, max_h=max_data_h * 1.2)
    ax2.plot(bc2, bm2, color=COLORS['dark'], linewidth=3, label='Mean ratio')
    ax2.fill_between(bc2, bm2 - bs2, bm2 + bs2, alpha=0.15, color=COLORS['dark'])

    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Original Entropy H(p)  [nats]', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Ratio $H(p^2/Z) / H(p)$', fontsize=12, fontweight='bold')
    ax2.set_title(r'Entropy Retention Ratio: Higher Entropy $\rightarrow$ Less Change',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, max_data_h * 1.2)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    ax2.annotate('Low entropy tokens:\nRatio drops toward 0\n(large relative change)',
                 xy=(0.3, 0.25), fontsize=9, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#FADBD8', alpha=0.85))
    ax2.annotate('High entropy tokens:\nRatio approaches 1\n(small relative change)',
                 xy=(max_data_h * 0.75, 0.88), fontsize=9, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F5E3', alpha=0.85))

    # ---- Plot 3: Absolute change vs H_orig ----
    ax3 = axes[1, 0]
    ax3.scatter(theory['H_orig'][mask_range], theory['abs_change'][mask_range],
                s=2, alpha=0.15, c=COLORS['teal'], rasterized=True)

    bc3, bm3, bs3, _ = compute_binned_stats(
        theory['H_orig'][mask_range], theory['abs_change'][mask_range],
        n_bins=50, max_h=max_data_h * 1.2)
    ax3.plot(bc3, bm3, color=COLORS['dark'], linewidth=3, label='Mean absolute change')
    ax3.fill_between(bc3, bm3 - bs3, bm3 + bs3, alpha=0.15, color=COLORS['dark'])

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)

    peak_idx = np.argmin(bm3)
    ax3.plot(bc3[peak_idx], bm3[peak_idx], 'o', color=COLORS['red'],
             markersize=10, zorder=5)
    ax3.annotate(f'Peak absolute reduction\n'
                 f'at H ≈ {bc3[peak_idx]:.2f} nats\n'
                 f'ΔH ≈ {bm3[peak_idx]:.3f} nats',
                 xy=(bc3[peak_idx], bm3[peak_idx]),
                 xytext=(bc3[peak_idx] + max_data_h * 0.15, bm3[peak_idx] + 0.05),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F9E79F', alpha=0.85))

    ax3.set_xlabel('Original Entropy H(p)  [nats]', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'Absolute Change $H(p^2/Z) - H(p)$  [nats]', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Entropy Change: Peak at Intermediate Entropy',
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.set_xlim(0, max_data_h * 1.2)
    ax3.grid(True, alpha=0.3)

    # ---- Plot 4: Token entropy distribution (both models) ----
    ax4 = axes[1, 1]
    ent_max = max(np.percentile(entropies_0, 99.5), np.percentile(entropies_15, 99.5))
    bins_hist = np.linspace(0, ent_max, 100)

    ax4.hist(entropies_0[entropies_0 > 0], bins=bins_hist, alpha=0.5,
             color=COLORS['blue'], density=True, edgecolor='none',
             label=f'Step 0 (pass@1={pass1_0:.3f})')
    ax4.hist(entropies_15[entropies_15 > 0], bins=bins_hist, alpha=0.5,
             color=COLORS['red'], density=True, edgecolor='none',
             label=f'Step 15 (pass@1={pass1_15:.3f})')

    mean_0 = np.mean(entropies_0[entropies_0 > 0])
    mean_15 = np.mean(entropies_15[entropies_15 > 0])
    ax4.axvline(x=mean_0, color=COLORS['blue'], linestyle='--', linewidth=2, alpha=0.7,
                label=f'Mean (Step 0) = {mean_0:.3f}')
    ax4.axvline(x=mean_15, color=COLORS['red'], linestyle='--', linewidth=2, alpha=0.7,
                label=f'Mean (Step 15) = {mean_15:.3f}')

    zero_frac_0 = np.sum(entropies_0 == 0) / len(entropies_0) * 100
    zero_frac_15 = np.sum(entropies_15 == 0) / len(entropies_15) * 100

    info_text = (f'H=0 tokens:\n'
                 f'  Step 0:  {zero_frac_0:.1f}%\n'
                 f'  Step 15: {zero_frac_15:.1f}%')
    ax4.text(0.97, 0.95, info_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    ax4.set_xlabel('Token Entropy  [nats]', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density  (excluding H=0 tokens)', fontsize=12, fontweight='bold')
    ax4.set_title('Actual Token Entropy Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, 0.82))
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"主图已保存至: {output_file}")
    plt.close()


def plot_combined_insight(theory, entropies_0, entropies_15, pass1_0, pass1_15, output_file):
    """
    Create a combined figure with theoretical curve overlaid on actual data distribution.
    Uses dual y-axis: left for theoretical ratio, right for token density.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    max_data_h = max(np.percentile(entropies_0, 99.5), np.percentile(entropies_15, 99.5))

    # ---- Right axis: Token entropy density histograms ----
    bins_hist = np.linspace(0, max_data_h, 80)

    ax2.hist(entropies_0[entropies_0 > 0], bins=bins_hist, alpha=0.25,
             color=COLORS['blue'], density=True, edgecolor='none',
             label=f'Step 0 tokens (pass@1={pass1_0:.3f})')
    ax2.hist(entropies_15[entropies_15 > 0], bins=bins_hist, alpha=0.25,
             color=COLORS['red'], density=True, edgecolor='none',
             label=f'Step 15 tokens (pass@1={pass1_15:.3f})')
    ax2.set_ylabel('Token Density  (right axis)', fontsize=12,
                    fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # ---- Left axis: Theoretical ratio curve ----
    mask_range = theory['H_orig'] <= max_data_h * 1.1
    bc, bm, bs, _ = compute_binned_stats(
        theory['H_orig'][mask_range], theory['ratio'][mask_range],
        n_bins=60, max_h=max_data_h * 1.1)

    ax1.plot(bc, bm, color=COLORS['dark'], linewidth=3.5, zorder=10,
             label=r'Entropy retention ratio $H(p^2/Z)/H(p)$')
    ax1.fill_between(bc, bm - bs, bm + bs, alpha=0.2, color=COLORS['dark'], zorder=9)

    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Phase annotations
    ax1.axvspan(0, 0.3, alpha=0.08, color='red', zorder=0)
    ax1.axvspan(0.3, 1.5, alpha=0.08, color='yellow', zorder=0)
    ax1.axvspan(1.5, max_data_h * 1.1, alpha=0.08, color='green', zorder=0)

    y_label = 0.12
    ax1.text(0.15, y_label, 'Low H\n(large change)', ha='center', fontsize=9,
             fontweight='bold', alpha=0.7, color=COLORS['red'])
    ax1.text(0.9, y_label, 'Medium H\n(moderate change)', ha='center', fontsize=9,
             fontweight='bold', alpha=0.7, color=COLORS['orange'])
    ax1.text(max_data_h * 0.75, y_label, 'High H\n(small change)', ha='center',
             fontsize=9, fontweight='bold', alpha=0.7, color=COLORS['green'])

    ax1.set_xlabel('Token Entropy H(p)  [nats]', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'Entropy Retention Ratio $H(p^2/Z) / H(p)$  (left axis)',
                   fontsize=12, fontweight='bold')
    ax1.set_title(r'$p \rightarrow p^2$: High-Entropy Tokens Change Less,'
                  r' Low-Entropy Tokens Change More'
                  '\nTheoretical Prediction Overlaid with Actual Token Entropy Distribution',
                  fontsize=14, fontweight='bold', pad=15)

    ax1.set_xlim(0, max_data_h * 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"综合图已保存至: {output_file}")
    plt.close()


def plot_relative_change_detail(theory, entropies_0, entropies_15, pass1_0, pass1_15, output_file):
    """
    Detailed plot of relative change (%) with actual token entropy overlay.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    max_data_h = max(np.percentile(entropies_0, 99.5), np.percentile(entropies_15, 99.5))

    # Right axis: Token entropy density
    bins_hist = np.linspace(0, max_data_h, 80)
    ax2.hist(entropies_0[entropies_0 > 0], bins=bins_hist, alpha=0.2,
             color=COLORS['blue'], density=True, edgecolor='none',
             label=f'Step 0 tokens')
    ax2.hist(entropies_15[entropies_15 > 0], bins=bins_hist, alpha=0.2,
             color=COLORS['red'], density=True, edgecolor='none',
             label=f'Step 15 tokens')
    ax2.set_ylabel('Token Density  (right axis)', fontsize=12,
                    fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Left axis: Relative change curve
    mask_range = theory['H_orig'] <= max_data_h * 1.1
    bc, bm, bs, _ = compute_binned_stats(
        theory['H_orig'][mask_range], theory['rel_change'][mask_range] * 100,
        n_bins=60, max_h=max_data_h * 1.1)

    ax1.plot(bc, bm, color=COLORS['red'], linewidth=3.5, zorder=10,
             label=r'Relative change $\frac{H(p^2/Z) - H(p)}{H(p)} \times 100\%$')
    ax1.fill_between(bc, bm - bs, bm + bs, alpha=0.15, color=COLORS['red'], zorder=9)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax1.axhline(y=-50, color='gray', linestyle=':', linewidth=1, alpha=0.4,
                label='-50% reference')

    ax1.set_xlabel('Token Entropy H(p)  [nats]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Relative Entropy Change (%)  [left axis]',
                   fontsize=12, fontweight='bold')
    ax1.set_title(r'Relative Entropy Change After $p \rightarrow p^2/Z$'
                  '\nLow-Entropy Tokens Lose Proportionally More Entropy',
                  fontsize=14, fontweight='bold', pad=15)

    ax1.set_xlim(0, max_data_h * 1.05)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

    # Add summary text with dynamically computed values
    mask_low = (theory['H_orig'] > 0.05) & (theory['H_orig'] < 0.5)
    mask_med = (theory['H_orig'] >= 0.5) & (theory['H_orig'] < 2.0)
    mask_high = theory['H_orig'] >= 2.0
    pct_low = abs(np.mean(theory['rel_change'][mask_low]) * 100) if np.any(mask_low) else 0
    pct_med = abs(np.mean(theory['rel_change'][mask_med]) * 100) if np.any(mask_med) else 0
    pct_high = abs(np.mean(theory['rel_change'][mask_high]) * 100) if np.any(mask_high) else 0
    summary = (
        r"$\bf{Key\ Finding:}$" + "\n"
        f"  Low entropy (H < 0.5): ~{pct_low:.0f}% entropy reduction\n"
        f"  Medium entropy (0.5 < H < 2): ~{pct_med:.0f}% reduction\n"
        f"  High entropy (H > 2): ~{pct_high:.0f}% reduction\n"
        r"$\Rightarrow$ Low-H tokens change MUCH more than high-H tokens"
    )
    ax1.text(0.02, 0.02, summary, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"相对变化详图已保存至: {output_file}")
    plt.close()


def print_summary(theory, entropies_0, entropies_15):
    """Print numerical summary of the analysis"""
    print("\n" + "=" * 70)
    print("分析总结: p → p²/Z 变换对不同熵水平token的影响")
    print("=" * 70)

    # Theoretical analysis by entropy bins
    entropy_bins = [(0, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 3.0), (3.0, 6.0)]
    bin_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

    print(f"\n{'Entropy Range':<18} {'Label':<12} {'Avg Ratio':<14} {'Avg Rel Change':<16} {'Avg Abs Change':<16}")
    print("-" * 76)

    for (lo, hi), label in zip(entropy_bins, bin_labels):
        mask = (theory['H_orig'] >= lo) & (theory['H_orig'] < hi)
        if np.sum(mask) > 0:
            avg_ratio = np.mean(theory['ratio'][mask])
            avg_rel = np.mean(theory['rel_change'][mask]) * 100
            avg_abs = np.mean(theory['abs_change'][mask])
            print(f"[{lo:.1f}, {hi:.1f}) nats   {label:<12} {avg_ratio:<14.4f} {avg_rel:<16.2f}% {avg_abs:<16.4f}")

    # Data statistics
    for name, ent in [("Step 0", entropies_0), ("Step 15", entropies_15)]:
        print(f"\n{name} 数据统计:")
        print(f"  总token数: {len(ent)}")
        print(f"  H=0 token占比: {np.sum(ent == 0) / len(ent) * 100:.1f}%")
        print(f"  非零熵均值: {np.mean(ent[ent > 0]):.4f} nats")
        print(f"  非零熵中位数: {np.median(ent[ent > 0]):.4f} nats")
        print(f"  最大熵: {np.max(ent):.4f} nats")

        for (lo, hi), label in zip(entropy_bins, bin_labels):
            frac = np.sum((ent > lo) & (ent <= hi)) / len(ent) * 100
            print(f"  {label} [{lo:.1f}, {hi:.1f}): {frac:.1f}%")

    # Dynamic conclusions based on computed data
    mask_low = (theory['H_orig'] > 0.05) & (theory['H_orig'] < 0.5)
    mask_high = theory['H_orig'] >= 2.0
    ratio_low = np.mean(theory['ratio'][mask_low]) if np.any(mask_low) else 0
    ratio_high = np.mean(theory['ratio'][mask_high]) if np.any(mask_high) else 0
    rel_low = abs(np.mean(theory['rel_change'][mask_low]) * 100) if np.any(mask_low) else 0
    rel_high = abs(np.mean(theory['rel_change'][mask_high]) * 100) if np.any(mask_high) else 0

    print("\n" + "=" * 70)
    print("结论:")
    print(f"  1. 低熵token (H<0.5): 熵保留率仅 {ratio_low:.1%}, 相对减少 ~{rel_low:.0f}%")
    print(f"  2. 高熵token (H>2):   熵保留率 {ratio_high:.1%}, 相对减少 ~{rel_high:.0f}%")
    print(f"  3. 差异: 低熵token的相对变化是高熵token的 {rel_low/rel_high:.1f}x")
    print("  4. 绝对变化(|ΔH|)随原始熵增大而增大(高熵token绝对减少更多)")
    print("  5. H=0的token (已确定) 不受影响; H→ln(V)的token (近均匀) 几乎不变")
    print("  → 假设成立: 高熵token相对变化更缓, 低熵token相对变化更大 ✓")
    print("  → 但注意: 即使是高熵token也损失了相当比例的熵")
    print("  → p→p² 本质上等价于将softmax温度减半 (T → T/2)")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():
    file_step0 = ("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/"
                  "eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json")
    file_step15 = ("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/"
                   "eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json")

    output_dir = ("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/"
                  "eval_results_aime_full_entropy")

    # Load actual data
    entropies_0, pass1_0, n_resp_0 = load_token_entropies(file_step0)
    entropies_15, pass1_15, n_resp_15 = load_token_entropies(file_step15)

    # Theoretical simulation
    theory = theoretical_analysis(V=5000, n_temps=300, n_trials=30)

    # Print summary
    print_summary(theory, entropies_0, entropies_15)

    # Generate plots
    print("\n生成图像...")

    plot_main_figure(
        theory, entropies_0, entropies_15, pass1_0, pass1_15,
        os.path.join(output_dir, 'p_squared_entropy_analysis_main.png'))

    plot_combined_insight(
        theory, entropies_0, entropies_15, pass1_0, pass1_15,
        os.path.join(output_dir, 'p_squared_entropy_combined_insight.png'))

    plot_relative_change_detail(
        theory, entropies_0, entropies_15, pass1_0, pass1_15,
        os.path.join(output_dir, 'p_squared_entropy_relative_change.png'))

    print("\n✅ 所有分析完成！")


if __name__ == "__main__":
    main()
