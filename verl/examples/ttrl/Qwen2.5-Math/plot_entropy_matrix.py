#!/usr/bin/env python3
"""绘制熵矩阵图：筛选15步样本，按正确性着色行，按组内中位数二值化熵"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from collections import defaultdict
import seaborn as sns

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


def load_and_filter(json_file, n_steps=15):
    """加载JSON文件并筛选出指定步骤数的样本
    
    Returns:
        list of dict: 每个元素包含 'is_correct', 'entropies' (list of mean_entropy per step)
    """
    print(f"正在读取文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    samples = []
    
    for result in results:
        for response in result.get('responses', []):
            ea = response.get('entropy_analysis', {})
            stats = ea.get('overall_stats', {})
            steps = ea.get('steps', [])
            
            if stats.get('n_steps') != n_steps:
                continue
            if len(steps) != n_steps:
                continue
            
            # 按step_index排序，提取mean_entropy
            steps_sorted = sorted(steps, key=lambda s: s['step_index'])
            entropies = [s['mean_entropy'] for s in steps_sorted]
            
            samples.append({
                'is_correct': response.get('is_correct', False),
                'entropies': entropies,
                'question_id': result.get('id', ''),
                'response_id': response.get('response_id', 0),
            })
    
    print(f"  筛选出 {len(samples)} 个 {n_steps} 步样本 (正确: {sum(1 for s in samples if s['is_correct'])}, "
          f"错误: {sum(1 for s in samples if not s['is_correct'])})")
    return samples


def plot_entropy_matrix(samples, title_suffix="", output_file="entropy_matrix.png"):
    """绘制熵二值化矩阵图
    
    Args:
        samples: load_and_filter 返回的样本列表
        title_suffix: 标题后缀
        output_file: 输出文件路径
    """
    if not samples:
        print("没有样本可绘制")
        return
    
    n_samples = len(samples)
    n_steps = len(samples[0]['entropies'])
    
    # 构建熵矩阵
    entropy_matrix = np.array([s['entropies'] for s in samples])
    is_correct = np.array([s['is_correct'] for s in samples])
    question_ids = [s['question_id'] for s in samples]
    
    # 按 question_id 分组，计算组内每步中位数
    qid_groups = defaultdict(list)
    for i, qid in enumerate(question_ids):
        qid_groups[qid].append(i)
    
    # 每个样本使用其所属组的中位数进行二值化
    binary_matrix = np.zeros((n_samples, n_steps), dtype=int)
    sample_medians = np.zeros((n_samples, n_steps))  # 记录每个样本使用的中位数
    
    print(f"\n按 question_id 分组: {len(qid_groups)} 组")
    for qid, indices in qid_groups.items():
        group_entropies = entropy_matrix[indices]
        group_medians = np.median(group_entropies, axis=0)
        for i in indices:
            binary_matrix[i] = (entropy_matrix[i] < group_medians).astype(int)
            sample_medians[i] = group_medians
    
    # 打印各组中位数的统计（取所有组中位数的平均作为参考）
    all_group_medians = []
    for qid, indices in qid_groups.items():
        group_entropies = entropy_matrix[indices]
        all_group_medians.append(np.median(group_entropies, axis=0))
    avg_group_medians = np.mean(all_group_medians, axis=0)
    print(f"各组中位数的平均值 (参考):")
    for i, med in enumerate(avg_group_medians):
        print(f"  Step {i}: {med:.6f}")
    
    # 按正确性排序：正确的在上面，错误的在下面
    # 在每组内按照二值化行的sum排序（更多1的在上面）
    correct_indices = np.where(is_correct)[0]
    incorrect_indices = np.where(~is_correct)[0]
    
    # 对正确和错误分别按行sum排序
    correct_sums = binary_matrix[correct_indices].sum(axis=1)
    correct_sorted = correct_indices[np.argsort(-correct_sums)]
    
    incorrect_sums = binary_matrix[incorrect_indices].sum(axis=1)
    incorrect_sorted = incorrect_indices[np.argsort(-incorrect_sums)]
    
    # 合并排序索引：正确在上，错误在下
    sorted_indices = np.concatenate([correct_sorted, incorrect_sorted])
    
    binary_sorted = binary_matrix[sorted_indices]
    is_correct_sorted = is_correct[sorted_indices]
    
    n_correct = len(correct_sorted)
    n_incorrect = len(incorrect_sorted)
    
    # ============ 绘图 ============
    fig_height = max(8, n_samples * 0.05 + 3)
    fig, (ax_label, ax_main) = plt.subplots(
        1, 2, figsize=(16, fig_height),
        gridspec_kw={'width_ratios': [0.8, 15], 'wspace': 0.02}
    )
    
    # === 左侧：正确/错误颜色条 ===
    label_colors = np.zeros((n_samples, 1, 3))
    for i in range(n_samples):
        if is_correct_sorted[i]:
            label_colors[i, 0] = [0.2, 0.78, 0.35]  # 绿色
        else:
            label_colors[i, 0] = [0.91, 0.3, 0.24]   # 红色
    
    ax_label.imshow(label_colors, aspect='auto', interpolation='nearest')
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    ax_label.set_ylabel('Samples (sorted by correctness)', fontsize=12, fontweight='bold')
    
    # 添加分隔线
    if n_correct > 0 and n_incorrect > 0:
        ax_label.axhline(y=n_correct - 0.5, color='black', linewidth=2)
        ax_main.axhline(y=n_correct - 0.5, color='black', linewidth=2, linestyle='--')
    
    # 标注正确/错误的数量
    if n_correct > 0:
        ax_label.text(0, n_correct / 2, f'Correct\n({n_correct})', 
                     ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    if n_incorrect > 0:
        ax_label.text(0, n_correct + n_incorrect / 2, f'Wrong\n({n_incorrect})', 
                     ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # === 右侧：二值化矩阵热力图 ===
    # 创建自定义colormap: 0 -> 浅色(高熵/>=中位数), 1 -> 深色(低熵/<中位数)
    # 使用蓝白配色：0=白色(高熵), 1=深蓝(低熵)
    cmap = ListedColormap(['#FFFFFF', '#2C3E50'])
    
    # 为正确和错误的行使用不同的颜色方案
    # 正确的行: 0=浅绿, 1=深绿
    # 错误的行: 0=浅红, 1=深红
    rgb_matrix = np.zeros((n_samples, n_steps, 3))
    
    for i in range(n_samples):
        for j in range(n_steps):
            if is_correct_sorted[i]:
                if binary_sorted[i, j] == 1:  # 低熵(< 中位数)
                    rgb_matrix[i, j] = [0.18, 0.55, 0.34]  # 深绿
                else:  # 高熵(>= 中位数)
                    rgb_matrix[i, j] = [0.78, 0.94, 0.82]  # 浅绿
            else:
                if binary_sorted[i, j] == 1:  # 低熵(< 中位数)
                    rgb_matrix[i, j] = [0.75, 0.22, 0.17]  # 深红
                else:  # 高熵(>= 中位数)
                    rgb_matrix[i, j] = [0.96, 0.78, 0.76]  # 浅红
    
    ax_main.imshow(rgb_matrix, aspect='auto', interpolation='nearest')
    
    # 设置X轴
    ax_main.set_xticks(range(n_steps))
    ax_main.set_xticklabels([f'Step {i}' for i in range(n_steps)], fontsize=9, rotation=45, ha='right')
    ax_main.set_xlabel('Reasoning Steps', fontsize=12, fontweight='bold')
    
    # 设置Y轴 - 不显示每个样本的标签（太多了）
    ax_main.set_yticks([])
    
    # 标题
    fig.suptitle(f'Entropy Binary Matrix (15-Step Samples){title_suffix}\n'
                 f'Dark = Low Entropy (< group median) → Label 1  |  Light = High Entropy (≥ group median) → Label 0',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 图例
    legend_patches = [
        mpatches.Patch(facecolor=[0.18, 0.55, 0.34], edgecolor='black', label='Correct & Low Entropy (1)'),
        mpatches.Patch(facecolor=[0.78, 0.94, 0.82], edgecolor='black', label='Correct & High Entropy (0)'),
        mpatches.Patch(facecolor=[0.75, 0.22, 0.17], edgecolor='black', label='Wrong & Low Entropy (1)'),
        mpatches.Patch(facecolor=[0.96, 0.78, 0.76], edgecolor='black', label='Wrong & High Entropy (0)'),
    ]
    ax_main.legend(handles=legend_patches, loc='upper right', fontsize=9,
                   bbox_to_anchor=(1.0, -0.08), ncol=4, frameon=True, 
                   fancybox=True, shadow=True)
    
    # 在底部添加中位数信息（显示各组中位数的平均值作为参考）
    median_text = f"Avg Group Medians ({len(qid_groups)} groups): " + " | ".join([f"S{i}={avg_group_medians[i]:.4f}" for i in range(n_steps)])
    fig.text(0.5, 0.01, median_text, ha='center', fontsize=7, 
             style='italic', color='gray')
    
    # 添加统计信息框
    info_text = (f"Total Samples: {n_samples}\n"
                 f"Correct: {n_correct} ({n_correct/n_samples*100:.1f}%)\n"
                 f"Wrong: {n_incorrect} ({n_incorrect/n_samples*100:.1f}%)")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_main.text(1.01, 0.98, info_text, transform=ax_main.transAxes,
                fontsize=9, verticalalignment='top', bbox=props)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存至: {output_file}")
    plt.close()
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"  总样本数: {n_samples}")
    print(f"  正确样本数: {n_correct} ({n_correct/n_samples*100:.1f}%)")
    print(f"  错误样本数: {n_incorrect} ({n_incorrect/n_samples*100:.1f}%)")
    
    # 分析正确 vs 错误的二值化分布差异
    if n_correct > 0 and n_incorrect > 0:
        correct_matrix = binary_sorted[:n_correct]
        incorrect_matrix = binary_sorted[n_correct:]
        
        print(f"\n  每步低熵比例 (label=1):")
        print(f"  {'Step':<8} {'Correct':<12} {'Wrong':<12} {'Diff':<12}")
        print(f"  {'-'*44}")
        for j in range(n_steps):
            c_ratio = correct_matrix[:, j].mean()
            w_ratio = incorrect_matrix[:, j].mean()
            print(f"  {j:<8} {c_ratio:<12.4f} {w_ratio:<12.4f} {c_ratio - w_ratio:<12.4f}")


def main():
    # 两个JSON文件路径
    file1 = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260204_043201.json"
    file2 = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260204_041649.json"
    
    output_dir = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy"
    
    # 加载和筛选数据
    print("=" * 60)
    print("加载文件1 (043201 - 训练后模型)")
    print("=" * 60)
    samples_1 = load_and_filter(file1)
    
    print()
    print("=" * 60)
    print("加载文件2 (041649 - 基线模型)")
    print("=" * 60)
    samples_2 = load_and_filter(file2)
    
    # 分别绘制两个文件的矩阵图
    print("\n" + "=" * 60)
    print("绘制文件1矩阵图")
    print("=" * 60)
    plot_entropy_matrix(
        samples_1,
        title_suffix="\n[Model: 043201 (Trained)]",
        output_file=f"{output_dir}/entropy_matrix_043201.png"
    )
    
    print("\n" + "=" * 60)
    print("绘制文件2矩阵图")
    print("=" * 60)
    plot_entropy_matrix(
        samples_2,
        title_suffix="\n[Model: 041649 (Baseline)]",
        output_file=f"{output_dir}/entropy_matrix_041649.png"
    )
    
    # 合并两个文件绘制对比图
    print("\n" + "=" * 60)
    print("绘制合并对比图")
    print("=" * 60)
    
    fig, axes_pairs = plt.subplots(1, 4, figsize=(24, max(8, max(len(samples_1), len(samples_2)) * 0.05 + 3)),
                                    gridspec_kw={'width_ratios': [0.6, 7, 0.6, 7], 'wspace': 0.05})
    
    for idx, (samples, label, ax_lbl, ax_mat) in enumerate([
        (samples_1, "043201 (Trained)", axes_pairs[0], axes_pairs[1]),
        (samples_2, "041649 (Baseline)", axes_pairs[2], axes_pairs[3]),
    ]):
        n_samples = len(samples)
        n_steps = len(samples[0]['entropies'])
        
        entropy_matrix = np.array([s['entropies'] for s in samples])
        is_correct = np.array([s['is_correct'] for s in samples])
        question_ids = [s['question_id'] for s in samples]
        
        # 按 question_id 分组，计算组内每步中位数进行二值化
        qid_groups = defaultdict(list)
        for i, qid in enumerate(question_ids):
            qid_groups[qid].append(i)
        
        binary_matrix = np.zeros((n_samples, n_steps), dtype=int)
        for qid, indices in qid_groups.items():
            group_entropies = entropy_matrix[indices]
            group_medians = np.median(group_entropies, axis=0)
            for i in indices:
                binary_matrix[i] = (entropy_matrix[i] < group_medians).astype(int)
        
        correct_indices = np.where(is_correct)[0]
        incorrect_indices = np.where(~is_correct)[0]
        
        correct_sums = binary_matrix[correct_indices].sum(axis=1)
        correct_sorted = correct_indices[np.argsort(-correct_sums)]
        
        incorrect_sums = binary_matrix[incorrect_indices].sum(axis=1)
        incorrect_sorted = incorrect_indices[np.argsort(-incorrect_sums)]
        
        sorted_indices = np.concatenate([correct_sorted, incorrect_sorted])
        binary_sorted = binary_matrix[sorted_indices]
        is_correct_sorted = is_correct[sorted_indices]
        
        n_correct = len(correct_sorted)
        n_incorrect = len(incorrect_sorted)
        
        # 左侧颜色条
        label_colors = np.zeros((n_samples, 1, 3))
        for i in range(n_samples):
            if is_correct_sorted[i]:
                label_colors[i, 0] = [0.2, 0.78, 0.35]
            else:
                label_colors[i, 0] = [0.91, 0.3, 0.24]
        
        ax_lbl.imshow(label_colors, aspect='auto', interpolation='nearest')
        ax_lbl.set_xticks([])
        ax_lbl.set_yticks([])
        if n_correct > 0:
            ax_lbl.text(0, n_correct / 2, f'{n_correct}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        if n_incorrect > 0:
            ax_lbl.text(0, n_correct + n_incorrect / 2, f'{n_incorrect}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        if n_correct > 0 and n_incorrect > 0:
            ax_lbl.axhline(y=n_correct - 0.5, color='black', linewidth=2)
            ax_mat.axhline(y=n_correct - 0.5, color='black', linewidth=2, linestyle='--')
        
        # 矩阵
        rgb_matrix = np.zeros((n_samples, n_steps, 3))
        for i in range(n_samples):
            for j in range(n_steps):
                if is_correct_sorted[i]:
                    if binary_sorted[i, j] == 1:
                        rgb_matrix[i, j] = [0.18, 0.55, 0.34]
                    else:
                        rgb_matrix[i, j] = [0.78, 0.94, 0.82]
                else:
                    if binary_sorted[i, j] == 1:
                        rgb_matrix[i, j] = [0.75, 0.22, 0.17]
                    else:
                        rgb_matrix[i, j] = [0.96, 0.78, 0.76]
        
        ax_mat.imshow(rgb_matrix, aspect='auto', interpolation='nearest')
        ax_mat.set_xticks(range(n_steps))
        ax_mat.set_xticklabels([f'S{i}' for i in range(n_steps)], fontsize=8, rotation=45, ha='right')
        ax_mat.set_yticks([])
        ax_mat.set_title(f'{label}\n(N={n_samples}, Correct={n_correct}, Wrong={n_incorrect})',
                        fontsize=11, fontweight='bold')
    
    # 图例
    legend_patches = [
        mpatches.Patch(facecolor=[0.18, 0.55, 0.34], edgecolor='black', label='Correct & Low Entropy (1)'),
        mpatches.Patch(facecolor=[0.78, 0.94, 0.82], edgecolor='black', label='Correct & High Entropy (0)'),
        mpatches.Patch(facecolor=[0.75, 0.22, 0.17], edgecolor='black', label='Wrong & Low Entropy (1)'),
        mpatches.Patch(facecolor=[0.96, 0.78, 0.76], edgecolor='black', label='Wrong & High Entropy (0)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', fontsize=10, ncol=4,
              frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Entropy Binary Matrix Comparison (15-Step Samples)\n'
                 'Dark = Low Entropy (< group median) → 1  |  Light = High Entropy (≥ group median) → 0',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_combined = f"{output_dir}/entropy_matrix_comparison.png"
    plt.savefig(output_combined, dpi=300, bbox_inches='tight')
    print(f"\n合并对比图已保存至: {output_combined}")
    plt.close()
    
    print("\n✅ 所有矩阵图绘制完成！")


if __name__ == "__main__":
    main()
