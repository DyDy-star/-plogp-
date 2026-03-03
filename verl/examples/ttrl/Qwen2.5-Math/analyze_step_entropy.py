#!/usr/bin/env python3
"""分析JSON文件中每个步骤的平均熵，并以15步平均熵作为基线绘制图像"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


def analyze_entropy(json_file, ylim=None):
    """分析JSON文件中的熵数据
    
    Args:
        json_file: JSON文件路径
        ylim: 可选，固定的Y轴范围 (min, max)
    """
    print(f"正在读取文件: {json_file}")
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"文件读取完成")
    
    # 获取results
    results = data.get('results', [])
    print(f"共有 {len(results)} 个问题")
    
    # 统计每个步骤的熵值
    step_entropies = defaultdict(list)
    
    total_responses = 0
    responses_with_entropy = 0
    
    # 遍历所有问题
    for result_idx, result in enumerate(results):
        responses = result.get('responses', [])
        
        # 遍历每个response
        for response in responses:
            total_responses += 1
            entropy_analysis = response.get('entropy_analysis', {})
            
            if not entropy_analysis:
                continue
                
            steps = entropy_analysis.get('steps', [])
            if not steps:
                continue
            
            responses_with_entropy += 1
            
            # 遍历每个步骤
            for step in steps:
                step_index = step.get('step_index')
                mean_entropy = step.get('mean_entropy')
                
                if step_index is not None and mean_entropy is not None:
                    step_entropies[step_index].append(mean_entropy)
    
    print(f"总响应数: {total_responses}")
    print(f"包含熵分析的响应数: {responses_with_entropy}")
    
    if not step_entropies:
        print("错误：未找到熵数据")
        return
    
    # 计算每个步骤的平均熵和标准差
    steps = sorted(step_entropies.keys())
    avg_entropies = []
    std_entropies = []
    sample_counts = []
    
    print(f"\n总共有 {len(steps)} 个步骤")
    print("\n每个步骤的统计信息：")
    print(f"{'步骤':<8} {'平均熵':<12} {'标准差':<12} {'样本数':<10}")
    print("-" * 50)
    
    for step in steps:
        entropies = step_entropies[step]
        avg_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)
        avg_entropies.append(avg_entropy)
        std_entropies.append(std_entropy)
        sample_counts.append(len(entropies))
        
        print(f"{step:<8} {avg_entropy:<12.4f} {std_entropy:<12.4f} {len(entropies):<10}")
    
    # 计算前15步的平均熵作为基线
    if len(avg_entropies) >= 15:
        baseline = np.mean(avg_entropies[:15])
        baseline_steps = 15
        print(f"\n前15步的平均熵（基线）: {baseline:.4f}")
    else:
        baseline = np.mean(avg_entropies)
        baseline_steps = len(avg_entropies)
        print(f"\n所有步骤的平均熵（基线）: {baseline:.4f}")
        print(f"注意：总步骤数 ({len(avg_entropies)}) 少于15步")
    
    # 绘制图像
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('AIME Dataset: Step-by-Step Entropy Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 图1：平均熵随步骤变化（带误差条）
    ax1 = axes[0, 0]
    ax1.errorbar(steps, avg_entropies, yerr=std_entropies, 
                 fmt='o-', linewidth=2.5, markersize=8, capsize=4,
                 label='Average Entropy ± Std Dev', color='#3498DB', alpha=0.8)
    ax1.axhline(y=baseline, color='#E74C3C', linestyle='--', linewidth=2.5, 
                label=f'Baseline (First {baseline_steps} steps = {baseline:.4f})')
    
    # 添加阶段背景
    ax1.axvspan(-0.5, 2.5, alpha=0.1, color='red')
    ax1.axvspan(2.5, 10.5, alpha=0.1, color='yellow')
    ax1.axvspan(10.5, max(steps) + 0.5, alpha=0.1, color='green')
    
    ax1.set_xlabel('Reasoning Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')
    ax1.set_title('Average Entropy per Step (All Responses)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xticks(steps)
    ax1.set_xlim(-0.5, max(steps) + 0.5)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签（仅偶数步骤）
    for i, step in enumerate(steps):
        if i % 2 == 0:
            ax1.text(step, avg_entropies[i] + 0.01, f'{avg_entropies[i]:.3f}',
                    ha='center', va='bottom', fontsize=8, color='#3498DB', fontweight='bold')
    
    # 应用固定Y轴范围（如果指定）
    if ylim is not None:
        ax1.set_ylim(ylim)
    
    # 图2：与基线的相对差异（百分比）
    ax2 = axes[0, 1]
    relative_diff_pct = [(avg - baseline) / baseline * 100 for avg in avg_entropies]
    colors = ['#2ECC71' if diff <= 0 else '#E74C3C' for diff in relative_diff_pct]
    bars = ax2.bar(steps, relative_diff_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Reasoning Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Difference (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Entropy Difference from Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(steps)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (step, diff) in enumerate(zip(steps, relative_diff_pct)):
        if abs(diff) > 5:  # 只标注显著的差值
            ax2.text(i, diff + (2 if diff > 0 else -2), f'{diff:.1f}%', 
                    ha='center', va='bottom' if diff > 0 else 'top', fontsize=8)
    
    # 图3：样本数分布
    ax3 = axes[1, 0]
    ax3.bar(steps, sample_counts, alpha=0.7, color='#9B59B6', edgecolor='black', linewidth=1)
    ax3.set_xlabel('Reasoning Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax3.set_title('Sample Count per Step', fontsize=14, fontweight='bold')
    ax3.set_xticks(steps)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 图4：熵值的趋势线
    ax4 = axes[1, 1]
    sns.lineplot(x=steps, y=avg_entropies, marker='o', linewidth=2.5,
                 markersize=8, color='#3498DB', label='Average Entropy', ax=ax4)
    ax4.fill_between(steps, 
                     np.array(avg_entropies) - np.array(std_entropies),
                     np.array(avg_entropies) + np.array(std_entropies),
                     alpha=0.3, color='#3498DB', label='Std Dev Range')
    ax4.axhline(y=baseline, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Reasoning Step', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')
    ax4.set_title('Entropy Trend with Variability', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.set_xticks(steps)
    ax4.grid(True, alpha=0.3)
    
    # 应用固定Y轴范围（如果指定）
    if ylim is not None:
        ax4.set_ylim(ylim)
    
    # 添加整体信息文本
    info_text = f"Dataset Info:\n"
    info_text += f"• Test Samples: {len(results)}\n"
    info_text += f"• Total Responses: {total_responses}\n"
    info_text += f"• Valid Entropy Analysis: {responses_with_entropy}\n"
    info_text += f"• Number of Steps: {len(steps)}\n"
    info_text += f"• Baseline (First {baseline_steps} steps): {baseline:.4f}"
    
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 保存图像（带固定Y轴标记）
    if ylim is not None:
        output_file = json_file.replace('.json', f'_entropy_analysis_fixed_ylim.png')
    else:
        output_file = json_file.replace('.json', '_entropy_analysis.png')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存至: {output_file}")
    plt.close()
    
    # 统计信息
    print(f"\n统计摘要：")
    print(f"- 最小平均熵: {min(avg_entropies):.4f} (步骤 {steps[np.argmin(avg_entropies)]})")
    print(f"- 最大平均熵: {max(avg_entropies):.4f} (步骤 {steps[np.argmax(avg_entropies)]})")
    print(f"- 全局平均熵: {np.mean(avg_entropies):.4f}")
    print(f"- 全局标准差: {np.std(avg_entropies):.4f}")
    
    return steps, avg_entropies, baseline, std_entropies


def plot_simple_entropy(json_file, steps, avg_entropies, baseline, std_entropies, ylim=None):
    """绘制简洁版熵分析图
    
    Args:
        json_file: JSON文件路径
        steps: 步骤列表
        avg_entropies: 平均熵列表
        baseline: 基线值
        std_entropies: 标准差列表
        ylim: 可选，固定的Y轴范围 (min, max)
    """
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 绘制主线和误差带
    sns.lineplot(x=steps, y=avg_entropies, marker='o', linewidth=3, 
                 markersize=10, color='#3498DB', label='Average Entropy', ax=ax)
    
    # 填充标准差范围
    ax.fill_between(steps, 
                    np.array(avg_entropies) - np.array(std_entropies),
                    np.array(avg_entropies) + np.array(std_entropies),
                    alpha=0.2, color='#3498DB', label='Std Dev Range')
    
    # 基线
    ax.axhline(y=baseline, color='#E74C3C', linestyle='--', linewidth=2.5, 
               label=f'Baseline (First 15 steps = {baseline:.4f})', alpha=0.8)
    
    # 添加阶段背景
    ax.axvspan(-0.5, 2.5, alpha=0.1, color='red')
    ax.axvspan(2.5, 10.5, alpha=0.1, color='yellow')
    ax.axvspan(10.5, max(steps) + 0.5, alpha=0.1, color='green')
    
    # 添加阶段标签（如果使用固定Y轴则使用固定值，否则使用当前Y轴范围）
    if ylim is not None:
        label_y = ylim[1] * 0.95
    else:
        label_y = ax.get_ylim()[1] * 0.95
    ax.text(1, label_y, 'Early', ha='center', fontsize=10, 
            fontweight='bold', alpha=0.7)
    ax.text(6.5, label_y, 'Middle', ha='center', fontsize=10, 
            fontweight='bold', alpha=0.7)
    ax.text(min(12.5, (10.5 + max(steps)) / 2), label_y, 'Late', 
            ha='center', fontsize=10, fontweight='bold', alpha=0.7)
    
    # 添加数值标签（仅偶数步骤）
    for i, step in enumerate(steps):
        if i % 2 == 0:
            ax.text(step, avg_entropies[i] + 0.015, f'{avg_entropies[i]:.3f}',
                   ha='center', va='bottom', fontsize=9, color='#3498DB', fontweight='bold')
    
    ax.set_xlabel('Reasoning Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Entropy', fontsize=14, fontweight='bold')
    ax.set_title('AIME Dataset: Entropy Evolution Across Reasoning Steps', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xticks(steps)
    ax.set_xlim(-0.5, max(steps) + 0.5)
    ax.grid(True, alpha=0.3)
    
    # 应用固定Y轴范围（如果指定）
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # 保存简洁版（带固定Y轴标记）
    if ylim is not None:
        simple_output = json_file.replace('.json', f'_entropy_analysis_simple_fixed_ylim.png')
    else:
        simple_output = json_file.replace('.json', '_entropy_analysis_simple.png')
    plt.tight_layout()
    plt.savefig(simple_output, dpi=300, bbox_inches='tight')
    print(f"简洁版图像已保存至: {simple_output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="分析JSON文件中每个步骤的平均熵")
    parser.add_argument("json_file", type=str, help="评估结果JSON文件路径")
    parser.add_argument("--fixed-ylim", action="store_true", 
                        help="使用固定的Y轴范围 (0, 1.0) 以便对比不同模型")
    parser.add_argument("--ylim-max", type=float, default=1.0,
                        help="固定Y轴的最大值 (默认: 1.0)")
    parser.add_argument("--ylim-min", type=float, default=0.0,
                        help="固定Y轴的最小值 (默认: 0.0)")
    
    args = parser.parse_args()
    
    # 构建Y轴范围参数
    ylim = None
    if args.fixed_ylim:
        ylim = (args.ylim_min, args.ylim_max)
        print(f"使用固定Y轴范围: {ylim}")
    
    # 主要分析
    result = analyze_entropy(args.json_file, ylim=ylim)
    
    if result:
        steps, avg_entropies, baseline, std_entropies = result
        
        # 生成简洁版图像
        print("\n生成简洁版图像...")
        plot_simple_entropy(args.json_file, steps, avg_entropies, baseline, std_entropies, ylim=ylim)
        
        print("\n✅ 所有分析完成！")


if __name__ == "__main__":
    main()



