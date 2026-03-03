#!/usr/bin/env python3
"""
分析熵增/熵减方向下的 KL Forward、KL Reverse、JS Divergence 差异

按以下维度拆分:
  - 熵变方向: 熵增 (entropy_delta > 0) vs 熵减 (entropy_delta <= 0)
  - 正确性: Correct vs Wrong
  - 模型: Base vs Trained

绘图风格借鉴 analyze_entropy_results.py

使用方法:
    python analyze_entropy_direction_divergence.py \
        --base  eval_results_aime_full_entropy/aime_eval_full_entropy_20260206_131750.json \
        --trained eval_results_aime_full_entropy/aime_eval_full_entropy_20260206_131034.json
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ========================================
# 绘图风格（借鉴 analyze_entropy_results.py）
# ========================================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 颜色方案
COLORS = {
    'trained_correct':  '#2ECC71',  # 绿色
    'trained_wrong':    '#E74C3C',  # 红色
    'base_correct':     '#3498DB',  # 蓝色
    'base_wrong':       '#F39C12',  # 橙色
}

LABELS = {
    'trained_correct':  'Trained Correct',
    'trained_wrong':    'Trained Wrong',
    'base_correct':     'Base Correct',
    'base_wrong':       'Base Wrong',
}


# ========================================
# 数据加载
# ========================================

def load_transition_data(json_file, n_steps=15):
    """
    加载 step_transitions，按正确/错误分组。
    每个 sample 保留完整的 transition list。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct, wrong = [], []

    for r in data['results']:
        for resp in r['responses']:
            ea = resp.get('entropy_analysis', {})
            stats = ea.get('overall_stats', {})
            trans = ea.get('step_transitions', [])

            if stats.get('n_steps', 0) != n_steps:
                continue
            if len(trans) != n_steps - 1:
                continue

            sample = {
                'transitions': trans,
            }

            if resp.get('is_correct', False):
                correct.append(sample)
            else:
                wrong.append(sample)

    return correct, wrong


def split_by_entropy_direction(samples):
    """
    将样本的所有 transitions 按 entropy_delta 方向拆分。
    
    返回:
        increase: dict of lists  (熵增时的各指标)
        decrease: dict of lists  (熵减时的各指标)
    """
    increase = defaultdict(list)  # entropy_delta > 0
    decrease = defaultdict(list)  # entropy_delta <= 0

    for sample in samples:
        for t in sample['transitions']:
            delta = t['entropy_delta']
            target = increase if delta > 0 else decrease
            target['kl_forward'].append(t['kl_forward'])
            target['kl_reverse'].append(t['kl_reverse'])
            target['js'].append(t['js_divergence'])
            target['top10_overlap'].append(t['top10_overlap'])
            target['cosine_sim'].append(t['cosine_sim'])
            target['entropy_delta'].append(abs(delta))

    return dict(increase), dict(decrease)


def split_by_entropy_direction_per_sample(samples):
    """
    与 split_by_entropy_direction 类似，但返回 per-sample 聚合值。
    每个样本: 计算该样本内熵增/熵减 transition 各指标的均值。
    """
    inc_samples = []  # per-sample mean for entropy-increasing transitions
    dec_samples = []  # per-sample mean for entropy-decreasing transitions

    for sample in samples:
        inc_kl_fwd, inc_kl_rev, inc_js = [], [], []
        dec_kl_fwd, dec_kl_rev, dec_js = [], [], []

        for t in sample['transitions']:
            if t['entropy_delta'] > 0:
                inc_kl_fwd.append(t['kl_forward'])
                inc_kl_rev.append(t['kl_reverse'])
                inc_js.append(t['js_divergence'])
            else:
                dec_kl_fwd.append(t['kl_forward'])
                dec_kl_rev.append(t['kl_reverse'])
                dec_js.append(t['js_divergence'])

        if inc_kl_fwd:
            inc_samples.append({
                'kl_forward': np.mean(inc_kl_fwd),
                'kl_reverse': np.mean(inc_kl_rev),
                'js': np.mean(inc_js),
                'n_transitions': len(inc_kl_fwd),
            })
        if dec_kl_fwd:
            dec_samples.append({
                'kl_forward': np.mean(dec_kl_fwd),
                'kl_reverse': np.mean(dec_kl_rev),
                'js': np.mean(dec_js),
                'n_transitions': len(dec_kl_fwd),
            })

    return inc_samples, dec_samples


# ========================================
# 打印统计分析
# ========================================

def print_statistics(label, correct, wrong):
    """打印一个模型的详细统计"""
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")

    for cw_label, samples in [("Correct", correct), ("Wrong", wrong)]:
        inc, dec = split_by_entropy_direction(samples)
        print(f"\n  [{cw_label}] (n_samples={len(samples)})")

        for dir_label, data in [("熵增 (Δ>0)", inc), ("熵减 (Δ≤0)", dec)]:
            if not data.get('kl_forward'):
                print(f"    {dir_label}: 无数据")
                continue
            n = len(data['kl_forward'])
            print(f"    {dir_label} ({n} transitions):")
            print(f"      KL Forward:  {np.mean(data['kl_forward']):>8.4f} ± {np.std(data['kl_forward']):.4f}")
            print(f"      KL Reverse:  {np.mean(data['kl_reverse']):>8.4f} ± {np.std(data['kl_reverse']):.4f}")
            print(f"      JS Diverg:   {np.mean(data['js']):>8.4f} ± {np.std(data['js']):.4f}")
            print(f"      Top10 Ovlp:  {np.mean(data['top10_overlap']):>8.4f} ± {np.std(data['top10_overlap']):.4f}")
            print(f"      |ΔEntropy|:  {np.mean(data['entropy_delta']):>8.4f} ± {np.std(data['entropy_delta']):.4f}")

            # KL 不对称性分析
            asym = np.array(data['kl_forward']) - np.array(data['kl_reverse'])
            print(f"      KL 不对称性 (Fwd-Rev): {np.mean(asym):>+8.4f} ± {np.std(asym):.4f}")
            fwd_dom = np.sum(np.array(data['kl_forward']) > np.array(data['kl_reverse']))
            print(f"      KL_Fwd > KL_Rev 比例:  {fwd_dom}/{n} ({fwd_dom/n*100:.1f}%)")


# ========================================
# 绘图
# ========================================

def plot_main_figure(base_c, base_w, trained_c, trained_w, output_dir):
    """
    主图: 2x3 布局
    行: 熵增 vs 熵减
    列: KL Forward, KL Reverse, JS Divergence
    每个子图: 4组 boxplot (Trained Correct/Wrong, Base Correct/Wrong)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Distribution Divergence: Entropy Increase vs Decrease Direction',
        fontsize=16, fontweight='bold', y=1.01
    )

    metrics = ['kl_forward', 'kl_reverse', 'js']
    metric_labels = [
        'KL Forward  $D_{KL}(P_i \\| P_{i+1})$',
        'KL Reverse  $D_{KL}(P_{i+1} \\| P_i)$',
        'JS Divergence  $D_{JS}(P_i, P_{i+1})$',
    ]
    ylabels = ['KL Divergence (nats)', 'KL Divergence (nats)', 'JS Divergence (nats)']

    groups_data = {
        'trained_correct': trained_c,
        'trained_wrong':   trained_w,
        'base_correct':    base_c,
        'base_wrong':      base_w,
    }
    group_names = list(groups_data.keys())
    group_short = ['T-C', 'T-W', 'B-C', 'B-W']

    for row_idx, (dir_label, dir_filter) in enumerate([
        ('Entropy Increase (Δ>0)', lambda d: d > 0),
        ('Entropy Decrease (Δ≤0)', lambda d: d <= 0),
    ]):
        for col_idx, (metric, mlabel, yl) in enumerate(zip(metrics, metric_labels, ylabels)):
            ax = axes[row_idx][col_idx]

            box_data = []
            box_colors = []
            box_labels = []

            for gname in group_names:
                samples = groups_data[gname]
                vals = []
                for s in samples:
                    for t in s['transitions']:
                        if dir_filter(t['entropy_delta']):
                            vals.append(t[metric if metric != 'js' else 'js_divergence'])
                box_data.append(vals)
                box_colors.append(COLORS[gname])
                box_labels.append(LABELS[gname])

            if all(len(d) == 0 for d in box_data):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            bp = ax.boxplot(
                box_data,
                positions=range(len(box_data)),
                widths=0.6,
                patch_artist=True,
                showfliers=False,  # 不显示outlier以保持清晰
                showmeans=True,
                meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=5),
            )

            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            # 标注均值数值
            for i, vals in enumerate(box_data):
                if vals:
                    mean_val = np.mean(vals)
                    ax.text(i, ax.get_ylim()[1] * 0.02 + mean_val, f'{mean_val:.2f}',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax.set_xticks(range(len(group_short)))
            ax.set_xticklabels(group_short, fontsize=10, fontweight='bold')
            ax.set_ylabel(yl, fontsize=10)
            ax.set_title(f'{dir_label}\n{mlabel}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = Path(output_dir) / 'entropy_direction_divergence_boxplot.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  图1已保存: {out_path}")
    plt.close()


def plot_asymmetry_figure(base_c, base_w, trained_c, trained_w, output_dir):
    """
    图2: 2x2 布局 - KL不对称性分析
    [0,0] 熵增时 KL_Fwd vs KL_Rev scatter（Trained）
    [0,1] 熵增时 KL_Fwd vs KL_Rev scatter（Base）
    [1,0] 熵减时 KL_Fwd vs KL_Rev scatter（Trained）
    [1,1] 熵减时 KL_Fwd vs KL_Rev scatter（Base）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        'KL Asymmetry Analysis: Forward vs Reverse KL',
        fontsize=16, fontweight='bold', y=1.01
    )

    configs = [
        (0, 0, 'Entropy Increase (Δ>0) — Trained', trained_c, trained_w,
         COLORS['trained_correct'], COLORS['trained_wrong'], lambda d: d > 0),
        (0, 1, 'Entropy Increase (Δ>0) — Base', base_c, base_w,
         COLORS['base_correct'], COLORS['base_wrong'], lambda d: d > 0),
        (1, 0, 'Entropy Decrease (Δ≤0) — Trained', trained_c, trained_w,
         COLORS['trained_correct'], COLORS['trained_wrong'], lambda d: d <= 0),
        (1, 1, 'Entropy Decrease (Δ≤0) — Base', base_c, base_w,
         COLORS['base_correct'], COLORS['base_wrong'], lambda d: d <= 0),
    ]

    for row, col, title, c_samples, w_samples, c_color, w_color, dir_filter in configs:
        ax = axes[row][col]

        for samples, color, label in [
            (c_samples, c_color, 'Correct'),
            (w_samples, w_color, 'Wrong'),
        ]:
            kl_fwd, kl_rev = [], []
            for s in samples:
                for t in s['transitions']:
                    if dir_filter(t['entropy_delta']):
                        kl_fwd.append(t['kl_forward'])
                        kl_rev.append(t['kl_reverse'])

            if kl_fwd:
                ax.scatter(kl_fwd, kl_rev, c=color, alpha=0.15, s=10, label=label)
                # 标注均值点
                ax.scatter([np.mean(kl_fwd)], [np.mean(kl_rev)],
                          c=color, s=120, marker='*', edgecolors='black',
                          linewidths=1, zorder=5,
                          label=f'{label} mean ({np.mean(kl_fwd):.1f}, {np.mean(kl_rev):.1f})')

        # 对角线 (KL_Fwd == KL_Rev)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, linewidth=1, label='Fwd = Rev')

        ax.set_xlabel('KL Forward (nats)', fontsize=10)
        ax.set_ylabel('KL Reverse (nats)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = Path(output_dir) / 'entropy_direction_kl_asymmetry.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  图2已保存: {out_path}")
    plt.close()


def plot_summary_bars(base_c, base_w, trained_c, trained_w, output_dir):
    """
    图3: 2x2 汇总柱状图
    [0,0] 各组在熵增/熵减时的 KL Forward 均值
    [0,1] 各组在熵增/熵减时的 KL Reverse 均值
    [1,0] 各组在熵增/熵减时的 JS Divergence 均值
    [1,1] 各组在熵增/熵减时的 KL不对称性 (Fwd - Rev) 均值
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Summary: Divergence Metrics by Entropy Direction',
        fontsize=16, fontweight='bold', y=1.01
    )

    groups = {
        'trained_correct': trained_c,
        'trained_wrong':   trained_w,
        'base_correct':    base_c,
        'base_wrong':      base_w,
    }
    group_names = list(groups.keys())
    group_short = ['T-Correct', 'T-Wrong', 'B-Correct', 'B-Wrong']

    def get_metric_values(samples, metric_key, dir_filter):
        vals = []
        for s in samples:
            for t in s['transitions']:
                if dir_filter(t['entropy_delta']):
                    if metric_key == 'asymmetry':
                        vals.append(t['kl_forward'] - t['kl_reverse'])
                    elif metric_key == 'js':
                        vals.append(t['js_divergence'])
                    else:
                        vals.append(t[metric_key])
        return vals

    metrics_config = [
        ('kl_forward', 'KL Forward', 'KL Divergence (nats)'),
        ('kl_reverse', 'KL Reverse', 'KL Divergence (nats)'),
        ('js',         'JS Divergence', 'JS Divergence (nats)'),
        ('asymmetry',  'KL Asymmetry (Fwd - Rev)', 'Δ KL (nats)'),
    ]

    x = np.arange(len(group_names))
    bar_width = 0.35

    for idx, (metric_key, mtitle, ylabel) in enumerate(metrics_config):
        ax = axes[idx // 2][idx % 2]

        inc_means, inc_sems = [], []
        dec_means, dec_sems = [], []

        for gname in group_names:
            # 熵增
            vals_inc = get_metric_values(groups[gname], metric_key, lambda d: d > 0)
            inc_means.append(np.mean(vals_inc) if vals_inc else 0)
            inc_sems.append(np.std(vals_inc) / np.sqrt(len(vals_inc)) if len(vals_inc) > 1 else 0)
            # 熵减
            vals_dec = get_metric_values(groups[gname], metric_key, lambda d: d <= 0)
            dec_means.append(np.mean(vals_dec) if vals_dec else 0)
            dec_sems.append(np.std(vals_dec) / np.sqrt(len(vals_dec)) if len(vals_dec) > 1 else 0)

        bars1 = ax.bar(x - bar_width/2, inc_means, bar_width, yerr=inc_sems,
                       color=[COLORS[g] for g in group_names], alpha=0.8,
                       edgecolor='black', linewidth=0.8, capsize=3,
                       label='Entropy ↑')
        bars2 = ax.bar(x + bar_width/2, dec_means, bar_width, yerr=dec_sems,
                       color=[COLORS[g] for g in group_names], alpha=0.4,
                       edgecolor='black', linewidth=0.8, capsize=3, hatch='///',
                       label='Entropy ↓')

        # 数值标注
        for bar, val in zip(bars1, inc_means):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        for bar, val in zip(bars2, dec_means):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        if metric_key == 'asymmetry':
            ax.axhline(y=0, color='black', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(group_short, fontsize=9, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(mtitle, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 自定义图例: 实心=熵增, 斜线=熵减
        inc_patch = mpatches.Patch(facecolor='gray', alpha=0.8, edgecolor='black', label='Entropy ↑ (Increase)')
        dec_patch = mpatches.Patch(facecolor='gray', alpha=0.4, edgecolor='black', hatch='///', label='Entropy ↓ (Decrease)')
        ax.legend(handles=[inc_patch, dec_patch], fontsize=8, loc='best')

    # 底部信息
    info = (f"T-Correct: {len(trained_c)} | T-Wrong: {len(trained_w)} | "
            f"B-Correct: {len(base_c)} | B-Wrong: {len(base_w)} | "
            f"15-step samples only")
    fig.text(0.5, -0.01, info, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path = Path(output_dir) / 'entropy_direction_summary_bars.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  图3已保存: {out_path}")
    plt.close()


def plot_per_sample_distribution(base_c, base_w, trained_c, trained_w, output_dir):
    """
    图4: 2x2 - 每样本聚合值的分布 (histogram)
    比较: 熵增时的 mean JS vs 熵减时的 mean JS（per sample）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Per-Sample Distribution: Mean JS Divergence by Entropy Direction',
        fontsize=16, fontweight='bold', y=1.01
    )

    configs = [
        (0, 0, 'Trained Correct', trained_c, COLORS['trained_correct']),
        (0, 1, 'Trained Wrong', trained_w, COLORS['trained_wrong']),
        (1, 0, 'Base Correct', base_c, COLORS['base_correct']),
        (1, 1, 'Base Wrong', base_w, COLORS['base_wrong']),
    ]

    for row, col, title, samples, color in configs:
        ax = axes[row][col]
        inc_samples, dec_samples = split_by_entropy_direction_per_sample(samples)

        inc_js = [s['js'] for s in inc_samples] if inc_samples else []
        dec_js = [s['js'] for s in dec_samples] if dec_samples else []

        bins = np.linspace(0, max(
            max(inc_js) if inc_js else 0.5,
            max(dec_js) if dec_js else 0.5
        ) * 1.1, 25)

        if inc_js:
            ax.hist(inc_js, bins=bins, alpha=0.6, color=color,
                    edgecolor='black', linewidth=0.5,
                    label=f'Entropy ↑ (n={len(inc_js)}, μ={np.mean(inc_js):.3f})')
        if dec_js:
            ax.hist(dec_js, bins=bins, alpha=0.4, color=color, hatch='///',
                    edgecolor='black', linewidth=0.5,
                    label=f'Entropy ↓ (n={len(dec_js)}, μ={np.mean(dec_js):.3f})')

        ax.set_xlabel('Mean JS Divergence (per sample)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{title} (n={len(samples)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = Path(output_dir) / 'entropy_direction_js_histogram.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  图4已保存: {out_path}")
    plt.close()


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(description="分析熵增/熵减方向下的散度指标")
    parser.add_argument('--base', required=True, help='Base model eval JSON')
    parser.add_argument('--trained', required=True, help='Trained model eval JSON')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.trained).parent)

    # 加载数据
    print("Loading data...")
    base_c, base_w = load_transition_data(args.base)
    trained_c, trained_w = load_transition_data(args.trained)

    print(f"  Trained: {len(trained_c)} correct, {len(trained_w)} wrong")
    print(f"  Base:    {len(base_c)} correct, {len(base_w)} wrong")

    # 统计分析
    print_statistics("Trained Model", trained_c, trained_w)
    print_statistics("Base Model", base_c, base_w)

    # 跨模型对比分析
    print(f"\n{'='*70}")
    print(" 跨模型/正确性 对比")
    print(f"{'='*70}")

    for gname, samples in [
        ("Trained Correct", trained_c), ("Trained Wrong", trained_w),
        ("Base Correct", base_c), ("Base Wrong", base_w),
    ]:
        inc, dec = split_by_entropy_direction(samples)
        inc_js = np.mean(inc['js']) if inc.get('js') else 0
        dec_js = np.mean(dec['js']) if dec.get('js') else 0
        inc_n = len(inc.get('js', []))
        dec_n = len(dec.get('js', []))
        inc_ratio = inc_n / (inc_n + dec_n) * 100 if (inc_n + dec_n) > 0 else 0
        print(f"\n  {gname}:")
        print(f"    熵增 transitions: {inc_n} ({inc_ratio:.1f}%), mean JS = {inc_js:.4f}")
        print(f"    熵减 transitions: {dec_n} ({100-inc_ratio:.1f}%), mean JS = {dec_js:.4f}")
        if inc_js > 0 and dec_js > 0:
            print(f"    JS ratio (↑/↓): {inc_js/dec_js:.3f}")

    # 绘图
    print(f"\n{'='*70}")
    print(" 生成可视化图表")
    print(f"{'='*70}")

    plot_main_figure(base_c, base_w, trained_c, trained_w, output_dir)
    plot_asymmetry_figure(base_c, base_w, trained_c, trained_w, output_dir)
    plot_summary_bars(base_c, base_w, trained_c, trained_w, output_dir)
    plot_per_sample_distribution(base_c, base_w, trained_c, trained_w, output_dir)

    print(f"\n{'='*70}")
    print("分析完成！")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
