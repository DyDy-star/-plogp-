#!/usr/bin/env python3
"""
分析熵增/熵减时的分步熵值和增量分布，与高斯分布和均匀分布的关系。

分析内容：
1. 每步熵值的分布 vs 高斯 / 均匀
2. 步间熵增量(velocity)的分布 vs 高斯 / 均匀
3. 熵增步和熵减步分别分析
4. 正确 vs 错误样本对比
5. 正态性检验 (Shapiro-Wilk, D'Agostino-Pearson)
6. 与均匀分布的 KS 检验

绘图风格借鉴 analyze_entropy_results.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# ─── 配置 ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

DATA_DIR = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy"
FILES = {
    "Checkpoint (step240)": os.path.join(DATA_DIR, "aime_eval_full_entropy_20260204_043201.json"),
    "Base (Qwen2.5-Math-1.5B)": os.path.join(DATA_DIR, "aime_eval_full_entropy_20260204_041649.json"),
}
OUTPUT_DIR = DATA_DIR
N_STEPS_FILTER = 15  # 只分析15步的样本


# ─── 数据加载 ─────────────────────────────────────────────
def load_data(filepath):
    """加载 JSON，提取15步样本的 step entropies"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = []
    for result in data['results']:
        for resp in result['responses']:
            analysis = resp['entropy_analysis']
            n_steps = analysis['overall_stats']['n_steps']
            if n_steps != N_STEPS_FILTER:
                continue

            step_entropies = [s['mean_entropy'] for s in analysis['steps']]
            is_correct = resp['is_correct']

            # 计算步间增量 (velocity)
            velocities = [step_entropies[i+1] - step_entropies[i]
                          for i in range(len(step_entropies)-1)]

            samples.append({
                'step_entropies': step_entropies,
                'velocities': velocities,
                'is_correct': is_correct,
            })
    return samples


def compute_stats(values, label=""):
    """计算分布统计量和正态性检验"""
    arr = np.array(values)
    if len(arr) < 8:
        return None

    stats = {
        'n': len(arr),
        'mean': np.mean(arr),
        'std': np.std(arr),
        'median': np.median(arr),
        'skewness': float(skewness(arr)),
        'kurtosis': float(kurtosis_excess(arr)),
        'min': np.min(arr),
        'max': np.max(arr),
    }

    # Shapiro-Wilk 正态性检验 (最多5000个样本)
    subset = arr[:5000] if len(arr) > 5000 else arr
    try:
        from scipy.stats import shapiro, kstest, normaltest
        sw_stat, sw_p = shapiro(subset)
        stats['shapiro_W'] = sw_stat
        stats['shapiro_p'] = sw_p

        # D'Agostino-Pearson 检验
        if len(subset) >= 20:
            da_stat, da_p = normaltest(subset)
            stats['dagostino_stat'] = da_stat
            stats['dagostino_p'] = da_p

        # KS 检验 vs 均匀分布 (归一化到 [0,1])
        if arr.max() > arr.min():
            normed = (arr - arr.min()) / (arr.max() - arr.min())
            ks_stat, ks_p = kstest(normed, 'uniform')
            stats['ks_uniform_stat'] = ks_stat
            stats['ks_uniform_p'] = ks_p

        # KS 检验 vs 正态分布
        standardized = (arr - np.mean(arr)) / (np.std(arr) + 1e-10)
        ks_norm_stat, ks_norm_p = kstest(standardized, 'norm')
        stats['ks_normal_stat'] = ks_norm_stat
        stats['ks_normal_p'] = ks_norm_p

    except ImportError:
        pass  # scipy not available

    return stats


def skewness(arr):
    """计算偏度"""
    n = len(arr)
    if n < 3:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return np.mean(((arr - m) / s) ** 3)


def kurtosis_excess(arr):
    """计算超额峰度 (正态分布=0)"""
    n = len(arr)
    if n < 4:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return np.mean(((arr - m) / s) ** 4) - 3.0


# ─── 绘图函数 ─────────────────────────────────────────────
def plot_distribution_with_fits(ax, values, title, color='steelblue', show_fits=True):
    """绘制直方图 + 高斯拟合 + 均匀分布参考线"""
    arr = np.array(values)
    if len(arr) < 5:
        ax.text(0.5, 0.5, f'数据不足 (n={len(arr)})', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # 直方图
    n_bins = min(50, max(15, len(arr) // 20))
    counts, bins, patches = ax.hist(arr, bins=n_bins, density=True, alpha=0.6,
                                     color=color, edgecolor='white', linewidth=0.5,
                                     label=f'数据 (n={len(arr)})')

    if show_fits:
        x = np.linspace(arr.min() - 0.1 * (arr.max() - arr.min()),
                        arr.max() + 0.1 * (arr.max() - arr.min()), 300)

        # 高斯拟合
        mu, sigma = np.mean(arr), np.std(arr)
        if sigma > 0:
            gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, gaussian, 'r-', lw=2, label=f'高斯 (μ={mu:.4f}, σ={sigma:.4f})')

        # 均匀分布参考线
        data_range = arr.max() - arr.min()
        if data_range > 0:
            uniform_height = 1.0 / data_range
            ax.axhline(y=uniform_height, color='orange', linestyle='--', lw=1.5,
                       label=f'均匀分布 (h={uniform_height:.2f})')

    ax.set_title(title)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_qq(ax, values, title, color='steelblue'):
    """Q-Q 图 (与标准正态对比)"""
    arr = np.array(values)
    if len(arr) < 5:
        ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    sorted_data = np.sort(arr)
    n = len(sorted_data)
    # 理论正态分位数
    theoretical = np.array([(i - 0.5) / n for i in range(1, n + 1)])

    try:
        from scipy.stats import norm
        theoretical_q = norm.ppf(theoretical)
    except ImportError:
        # 手动计算近似
        theoretical_q = np.sqrt(2) * erfinv(2 * theoretical - 1)

    # 标准化数据
    mu, sigma = np.mean(arr), np.std(arr)
    if sigma > 0:
        standardized = (sorted_data - mu) / sigma
    else:
        standardized = sorted_data - mu

    ax.scatter(theoretical_q, standardized, s=3, alpha=0.5, color=color)
    lim = max(abs(theoretical_q.min()), abs(theoretical_q.max()), abs(standardized.min()), abs(standardized.max()))
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, label='y=x (完美正态)')
    ax.set_xlabel('理论正态分位数')
    ax.set_ylabel('标准化数据分位数')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


def erfinv(x):
    """近似 erfinv (Winitzki 2008)"""
    a = 0.147
    ln_part = np.log(1 - x ** 2)
    term1 = (2 / (np.pi * a)) + ln_part / 2
    result = np.sign(x) * np.sqrt(np.sqrt(term1 ** 2 - ln_part / a) - term1)
    return result


# ─── 主分析 ─────────────────────────────────────────────
def analyze_model(model_name, samples):
    """分析单个模型的熵分布"""
    print(f"\n{'='*70}")
    print(f"模型: {model_name}  (15步样本数: {len(samples)})")
    print(f"{'='*70}")

    correct = [s for s in samples if s['is_correct']]
    incorrect = [s for s in samples if not s['is_correct']]
    print(f"  正确: {len(correct)},  错误: {len(incorrect)}")

    # ── 收集所有数据 ──
    all_step_entropies = {i: [] for i in range(N_STEPS_FILTER)}
    all_velocities_pos = []  # 熵增步
    all_velocities_neg = []  # 熵减步
    all_velocities_all = []

    per_step_velocities = {i: [] for i in range(N_STEPS_FILTER - 1)}

    for group_name, group in [("正确", correct), ("错误", incorrect)]:
        step_ent = {i: [] for i in range(N_STEPS_FILTER)}
        vel_pos = []
        vel_neg = []
        vel_all = []
        step_vel = {i: [] for i in range(N_STEPS_FILTER - 1)}

        for s in group:
            for i, h in enumerate(s['step_entropies']):
                step_ent[i].append(h)
                all_step_entropies[i].append(h)
            for i, v in enumerate(s['velocities']):
                vel_all.append(v)
                all_velocities_all.append(v)
                step_vel[i].append(v)
                per_step_velocities[i].append(v)
                if v > 0:
                    vel_pos.append(v)
                    all_velocities_pos.append(v)
                elif v < 0:
                    vel_neg.append(v)
                    all_velocities_neg.append(v)

        # 打印统计
        print(f"\n  [{group_name}样本] 速度(增量)统计:")
        print(f"    全部增量:  n={len(vel_all)}, mean={np.mean(vel_all):.5f}, std={np.std(vel_all):.5f}")
        if vel_pos:
            print(f"    熵增(v>0): n={len(vel_pos)}, mean={np.mean(vel_pos):.5f}, std={np.std(vel_pos):.5f}")
            stats = compute_stats(vel_pos)
            if stats and 'shapiro_p' in stats:
                print(f"      偏度={stats['skewness']:.3f}, 峰度={stats['kurtosis']:.3f}")
                print(f"      Shapiro-Wilk p={stats['shapiro_p']:.2e} ({'正态' if stats['shapiro_p']>0.05 else '非正态'})")
                if 'ks_normal_p' in stats:
                    print(f"      KS vs 正态  p={stats['ks_normal_p']:.2e}")
                if 'ks_uniform_p' in stats:
                    print(f"      KS vs 均匀  p={stats['ks_uniform_p']:.2e}")
        if vel_neg:
            print(f"    熵减(v<0): n={len(vel_neg)}, mean={np.mean(vel_neg):.5f}, std={np.std(vel_neg):.5f}")
            stats = compute_stats(vel_neg)
            if stats and 'shapiro_p' in stats:
                print(f"      偏度={stats['skewness']:.3f}, 峰度={stats['kurtosis']:.3f}")
                print(f"      Shapiro-Wilk p={stats['shapiro_p']:.2e} ({'正态' if stats['shapiro_p']>0.05 else '非正态'})")
                if 'ks_normal_p' in stats:
                    print(f"      KS vs 正态  p={stats['ks_normal_p']:.2e}")
                if 'ks_uniform_p' in stats:
                    print(f"      KS vs 均匀  p={stats['ks_uniform_p']:.2e}")

    return {
        'correct': correct,
        'incorrect': incorrect,
        'all_step_entropies': all_step_entropies,
        'all_velocities_pos': all_velocities_pos,
        'all_velocities_neg': all_velocities_neg,
        'all_velocities_all': all_velocities_all,
        'per_step_velocities': per_step_velocities,
    }


def plot_model_analysis(model_name, info, output_path):
    """为一个模型生成完整分析图"""
    correct = info['correct']
    incorrect = info['incorrect']

    fig = plt.figure(figsize=(24, 28))
    fig.suptitle(f'{model_name} — 熵增/熵减分布 vs 高斯 vs 均匀', fontsize=16, fontweight='bold', y=0.98)

    # ── Row 1: 全部增量的分布 + Q-Q ──
    ax1 = fig.add_subplot(6, 4, 1)
    plot_distribution_with_fits(ax1, info['all_velocities_all'], '全部增量分布 (所有样本)', 'steelblue')

    ax2 = fig.add_subplot(6, 4, 2)
    plot_qq(ax2, info['all_velocities_all'], 'Q-Q: 全部增量 vs 正态', 'steelblue')

    ax3 = fig.add_subplot(6, 4, 3)
    plot_distribution_with_fits(ax3, info['all_velocities_pos'], '熵增增量 (v>0)', '#d62728')

    ax4 = fig.add_subplot(6, 4, 4)
    plot_distribution_with_fits(ax4, info['all_velocities_neg'], '熵减增量 (v<0)', '#1f77b4')

    # ── Row 2: 正确 vs 错误的增量 ──
    correct_vel = [v for s in correct for v in s['velocities']]
    incorrect_vel = [v for s in incorrect for v in s['velocities']]
    correct_vel_pos = [v for v in correct_vel if v > 0]
    correct_vel_neg = [v for v in correct_vel if v < 0]
    incorrect_vel_pos = [v for v in incorrect_vel if v > 0]
    incorrect_vel_neg = [v for v in incorrect_vel if v < 0]

    ax5 = fig.add_subplot(6, 4, 5)
    plot_distribution_with_fits(ax5, correct_vel_pos, '正确样本 - 熵增增量', '#2ca02c')
    ax6 = fig.add_subplot(6, 4, 6)
    plot_distribution_with_fits(ax6, correct_vel_neg, '正确样本 - 熵减增量', '#2ca02c')
    ax7 = fig.add_subplot(6, 4, 7)
    plot_distribution_with_fits(ax7, incorrect_vel_pos, '错误样本 - 熵增增量', '#d62728')
    ax8 = fig.add_subplot(6, 4, 8)
    plot_distribution_with_fits(ax8, incorrect_vel_neg, '错误样本 - 熵减增量', '#d62728')

    # ── Row 3: Q-Q 图 (正确 vs 错误, 增 vs 减) ──
    ax9 = fig.add_subplot(6, 4, 9)
    plot_qq(ax9, correct_vel_pos, 'Q-Q: 正确-熵增 vs 正态', '#2ca02c')
    ax10 = fig.add_subplot(6, 4, 10)
    plot_qq(ax10, correct_vel_neg, 'Q-Q: 正确-熵减 vs 正态', '#2ca02c')
    ax11 = fig.add_subplot(6, 4, 11)
    plot_qq(ax11, incorrect_vel_pos, 'Q-Q: 错误-熵增 vs 正态', '#d62728')
    ax12 = fig.add_subplot(6, 4, 12)
    plot_qq(ax12, incorrect_vel_neg, 'Q-Q: 错误-熵减 vs 正态', '#d62728')

    # ── Row 4: 分步熵值分布 (每个步骤位置) ──
    ax13 = fig.add_subplot(6, 4, 13)
    step_means_correct = []
    step_means_incorrect = []
    step_stds_correct = []
    step_stds_incorrect = []
    for i in range(N_STEPS_FILTER):
        c_ent = [s['step_entropies'][i] for s in correct]
        ic_ent = [s['step_entropies'][i] for s in incorrect]
        step_means_correct.append(np.mean(c_ent) if c_ent else 0)
        step_means_incorrect.append(np.mean(ic_ent) if ic_ent else 0)
        step_stds_correct.append(np.std(c_ent) if c_ent else 0)
        step_stds_incorrect.append(np.std(ic_ent) if ic_ent else 0)

    x = np.arange(N_STEPS_FILTER)
    ax13.errorbar(x, step_means_correct, yerr=step_stds_correct,
                  fmt='o-', color='#2ca02c', capsize=3, label='正确', markersize=4)
    ax13.errorbar(x, step_means_incorrect, yerr=step_stds_incorrect,
                  fmt='s-', color='#d62728', capsize=3, label='错误', markersize=4)
    ax13.set_xlabel('步骤')
    ax13.set_ylabel('平均熵')
    ax13.set_title('各步骤熵值 (正确 vs 错误)')
    ax13.legend(fontsize=8)
    ax13.grid(True, alpha=0.3)

    # ── Row 4 cont: 分步增量分布 (每个步骤位置) ──
    ax14 = fig.add_subplot(6, 4, 14)
    vel_means_correct = []
    vel_means_incorrect = []
    vel_stds_correct = []
    vel_stds_incorrect = []
    for i in range(N_STEPS_FILTER - 1):
        c_vel = [s['velocities'][i] for s in correct]
        ic_vel = [s['velocities'][i] for s in incorrect]
        vel_means_correct.append(np.mean(c_vel) if c_vel else 0)
        vel_means_incorrect.append(np.mean(ic_vel) if ic_vel else 0)
        vel_stds_correct.append(np.std(c_vel) if c_vel else 0)
        vel_stds_incorrect.append(np.std(ic_vel) if ic_vel else 0)

    x_vel = np.arange(N_STEPS_FILTER - 1)
    ax14.errorbar(x_vel, vel_means_correct, yerr=vel_stds_correct,
                  fmt='o-', color='#2ca02c', capsize=3, label='正确', markersize=4)
    ax14.errorbar(x_vel, vel_means_incorrect, yerr=vel_stds_incorrect,
                  fmt='s-', color='#d62728', capsize=3, label='错误', markersize=4)
    ax14.axhline(y=0, color='gray', linestyle='-', lw=0.8)
    ax14.set_xlabel('步骤转换 (i→i+1)')
    ax14.set_ylabel('平均增量')
    ax14.set_title('各步骤位置的增量 (正确 vs 错误)')
    ax14.legend(fontsize=8)
    ax14.grid(True, alpha=0.3)

    # ── Row 4 cont: 各步骤位置的增量分布箱线图 ──
    ax15 = fig.add_subplot(6, 4, 15)
    bp_data = [info['per_step_velocities'][i] for i in range(N_STEPS_FILTER - 1)]
    bp = ax15.boxplot(bp_data, positions=range(N_STEPS_FILTER - 1),
                      widths=0.6, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#9ecae1')
        patch.set_alpha(0.7)
    ax15.axhline(y=0, color='gray', linestyle='-', lw=0.8)
    ax15.set_xlabel('步骤转换 (i→i+1)')
    ax15.set_ylabel('增量')
    ax15.set_title('各步骤位置增量的箱线图')
    ax15.grid(True, alpha=0.3)

    # ── Row 4 cont: 熵增/熵减比例 per step ──
    ax16 = fig.add_subplot(6, 4, 16)
    increase_ratio = []
    for i in range(N_STEPS_FILTER - 1):
        vels = info['per_step_velocities'][i]
        n_pos = sum(1 for v in vels if v > 0)
        ratio = n_pos / len(vels) if vels else 0
        increase_ratio.append(ratio)
    ax16.bar(range(N_STEPS_FILTER - 1), increase_ratio, color='#ff7f0e', alpha=0.7)
    ax16.axhline(y=0.5, color='gray', linestyle='--', lw=1)
    ax16.set_xlabel('步骤转换 (i→i+1)')
    ax16.set_ylabel('熵增比例')
    ax16.set_title('各步骤位置的熵增比例')
    ax16.set_ylim(0, 1)
    ax16.grid(True, alpha=0.3)

    # ── Row 5: 各步骤的熵值分布直方图 (选择几个代表性步骤) ──
    representative_steps = [0, 3, 7, 11, 14]
    for idx, step_i in enumerate(representative_steps[:4]):
        ax = fig.add_subplot(6, 4, 17 + idx)
        step_vals = [s['step_entropies'][step_i] for s in correct + incorrect]
        c_vals = [s['step_entropies'][step_i] for s in correct]
        ic_vals = [s['step_entropies'][step_i] for s in incorrect]
        if c_vals and ic_vals:
            bins = np.linspace(0, max(np.max(c_vals), np.max(ic_vals)) * 1.1, 30)
            ax.hist(c_vals, bins=bins, density=True, alpha=0.5, color='#2ca02c', label='正确')
            ax.hist(ic_vals, bins=bins, density=True, alpha=0.5, color='#d62728', label='错误')
            # 高斯拟合 (全部)
            mu, sigma = np.mean(step_vals), np.std(step_vals)
            if sigma > 0:
                x_fit = np.linspace(0, bins[-1], 200)
                gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fit - mu) / sigma) ** 2)
                ax.plot(x_fit, gauss, 'k--', lw=1.5, label='高斯拟合')
        ax.set_title(f'步骤 {step_i} 的熵值分布')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # ── Row 6: 分步增量的分布 (选择几个代表性转换) ──
    representative_trans = [0, 4, 8, 12]
    for idx, trans_i in enumerate(representative_trans):
        ax = fig.add_subplot(6, 4, 21 + idx)
        c_vels = [s['velocities'][trans_i] for s in correct]
        ic_vels = [s['velocities'][trans_i] for s in incorrect]
        all_vels = c_vels + ic_vels
        if all_vels:
            bins = np.linspace(min(all_vels), max(all_vels), 30)
            if c_vels:
                ax.hist(c_vels, bins=bins, density=True, alpha=0.5, color='#2ca02c', label='正确')
            if ic_vels:
                ax.hist(ic_vels, bins=bins, density=True, alpha=0.5, color='#d62728', label='错误')
            # 高斯拟合
            mu, sigma = np.mean(all_vels), np.std(all_vels)
            if sigma > 0:
                x_fit = np.linspace(bins[0], bins[-1], 200)
                gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fit - mu) / sigma) ** 2)
                ax.plot(x_fit, gauss, 'k--', lw=1.5, label='高斯拟合')
        ax.set_title(f'转换 {trans_i}→{trans_i+1} 的增量分布')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n  图表已保存: {output_path}")


def print_summary_table(all_info):
    """打印正态性检验汇总表"""
    print(f"\n{'='*90}")
    print("正态性和均匀性检验汇总")
    print(f"{'='*90}")
    print(f"{'类别':<25} {'n':>6} {'均值':>10} {'标准差':>10} {'偏度':>8} {'峰度':>8} {'SW p':>10} {'KS-N p':>10} {'KS-U p':>10}")
    print("-" * 90)

    for model_name, info in all_info.items():
        print(f"\n  [{model_name}]")
        for label, values in [
            ("全部增量", info['all_velocities_all']),
            ("熵增增量(v>0)", info['all_velocities_pos']),
            ("熵减增量(v<0)", info['all_velocities_neg']),
        ]:
            stats = compute_stats(values, label)
            if stats:
                sw_p = f"{stats.get('shapiro_p', float('nan')):.2e}"
                ks_n = f"{stats.get('ks_normal_p', float('nan')):.2e}"
                ks_u = f"{stats.get('ks_uniform_p', float('nan')):.2e}"
                print(f"  {label:<23} {stats['n']:>6} {stats['mean']:>10.5f} {stats['std']:>10.5f} "
                      f"{stats['skewness']:>8.3f} {stats['kurtosis']:>8.3f} {sw_p:>10} {ks_n:>10} {ks_u:>10}")

        # 分正确/错误
        for group_name, group in [("正确", info['correct']), ("错误", info['incorrect'])]:
            vel = [v for s in group for v in s['velocities']]
            vel_pos = [v for v in vel if v > 0]
            vel_neg = [v for v in vel if v < 0]
            for sub_label, sub_vals in [(f"{group_name}-全部", vel),
                                         (f"{group_name}-熵增", vel_pos),
                                         (f"{group_name}-熵减", vel_neg)]:
                stats = compute_stats(sub_vals)
                if stats:
                    sw_p = f"{stats.get('shapiro_p', float('nan')):.2e}"
                    ks_n = f"{stats.get('ks_normal_p', float('nan')):.2e}"
                    ks_u = f"{stats.get('ks_uniform_p', float('nan')):.2e}"
                    print(f"    {sub_label:<21} {stats['n']:>6} {stats['mean']:>10.5f} {stats['std']:>10.5f} "
                          f"{stats['skewness']:>8.3f} {stats['kurtosis']:>8.3f} {sw_p:>10} {ks_n:>10} {ks_u:>10}")


def print_per_step_normality(all_info):
    """分步骤位置的正态性检验"""
    print(f"\n{'='*90}")
    print("分步骤位置的增量正态性检验 (Shapiro-Wilk)")
    print(f"{'='*90}")

    for model_name, info in all_info.items():
        print(f"\n  [{model_name}]")
        print(f"  {'步骤':>4} {'n':>6} {'均值':>10} {'标准差':>10} {'偏度':>8} {'SW p':>10} {'结论':>8}")
        print("  " + "-" * 65)
        for i in range(N_STEPS_FILTER - 1):
            vels = info['per_step_velocities'][i]
            stats = compute_stats(vels)
            if stats:
                sw_p = stats.get('shapiro_p', float('nan'))
                conclusion = "正态" if sw_p > 0.05 else "非正态"
                print(f"  {i:>2}→{i+1:<2} {stats['n']:>6} {stats['mean']:>10.5f} {stats['std']:>10.5f} "
                      f"{stats['skewness']:>8.3f} {sw_p:>10.2e} {conclusion:>8}")


# ─── 主函数 ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("熵增/熵减分布 vs 高斯 vs 均匀 分析")
    print("=" * 70)

    all_info = {}

    for model_name, filepath in FILES.items():
        if not os.path.exists(filepath):
            print(f"  文件不存在: {filepath}")
            continue

        samples = load_data(filepath)
        info = analyze_model(model_name, samples)
        all_info[model_name] = info

        # 生成图表
        safe_name = model_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        output_path = os.path.join(OUTPUT_DIR, f"entropy_gaussian_analysis_{safe_name}.png")
        plot_model_analysis(model_name, info, output_path)

    # 汇总表
    if all_info:
        print_summary_table(all_info)
        print_per_step_normality(all_info)

    # ── 两个模型对比图 ──
    if len(all_info) == 2:
        plot_comparison(all_info)

    print(f"\n{'='*70}")
    print("分析完成！")
    print(f"{'='*70}")


def plot_comparison(all_info):
    """两个模型的增量分布对比"""
    models = list(all_info.keys())
    colors = ['#1f77b4', '#ff7f0e']

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    fig.suptitle('两模型对比 — 增量分布 vs 高斯 vs 均匀', fontsize=14, fontweight='bold')

    for col, (model_name, info) in enumerate(all_info.items()):
        # Row 1: 全部增量, 熵增, 熵减
        ax = axes[0, col * 2]
        plot_distribution_with_fits(ax, info['all_velocities_pos'],
                                    f'{model_name[:15]}\n熵增增量', colors[col])
        ax = axes[0, col * 2 + 1]
        plot_distribution_with_fits(ax, info['all_velocities_neg'],
                                    f'{model_name[:15]}\n熵减增量', colors[col])

        # Row 2: Q-Q
        ax = axes[1, col * 2]
        plot_qq(ax, info['all_velocities_pos'],
                f'{model_name[:15]}\nQ-Q 熵增 vs 正态', colors[col])
        ax = axes[1, col * 2 + 1]
        plot_qq(ax, info['all_velocities_neg'],
                f'{model_name[:15]}\nQ-Q 熵减 vs 正态', colors[col])

    # Row 3: 正确 vs 错误 叠加
    for col, (model_name, info) in enumerate(all_info.items()):
        c_vel = [v for s in info['correct'] for v in s['velocities']]
        ic_vel = [v for s in info['incorrect'] for v in s['velocities']]

        ax = axes[2, col * 2]
        if c_vel and ic_vel:
            all_v = c_vel + ic_vel
            bins = np.linspace(min(all_v), max(all_v), 40)
            ax.hist(c_vel, bins=bins, density=True, alpha=0.5, color='#2ca02c', label=f'正确 (n={len(c_vel)})')
            ax.hist(ic_vel, bins=bins, density=True, alpha=0.5, color='#d62728', label=f'错误 (n={len(ic_vel)})')
            ax.legend(fontsize=7)
        ax.set_title(f'{model_name[:15]}\n全部增量 (正确 vs 错误)')
        ax.grid(True, alpha=0.3)

        # 增量的 std 对比 per sample
        ax = axes[2, col * 2 + 1]
        c_stds = [np.std(s['velocities']) for s in info['correct']]
        ic_stds = [np.std(s['velocities']) for s in info['incorrect']]
        if c_stds and ic_stds:
            bins = np.linspace(0, max(max(c_stds), max(ic_stds)), 30)
            ax.hist(c_stds, bins=bins, density=True, alpha=0.5, color='#2ca02c', label=f'正确 std(v)')
            ax.hist(ic_stds, bins=bins, density=True, alpha=0.5, color='#d62728', label=f'错误 std(v)')
            ax.legend(fontsize=7)
        ax.set_title(f'{model_name[:15]}\n样本级 std(velocity)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(OUTPUT_DIR, "entropy_gaussian_comparison.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n  对比图已保存: {output_path}")


if __name__ == "__main__":
    main()
