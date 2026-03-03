#!/usr/bin/env python3
"""分析强化学习前后熵方差的变化"""

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


def extract_step_entropies(json_file):
    """从JSON文件中提取每个步骤的熵值"""
    print(f"正在读取文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"共有 {len(results)} 个问题")
    
    # 统计每个步骤的熵值
    step_entropies = defaultdict(list)
    
    total_responses = 0
    responses_with_entropy = 0
    
    # 遍历所有问题
    for result in results:
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
    
    return step_entropies


def analyze_variance_changes(rl_file, baseline_file):
    """分析强化学习前后方差的变化"""
    
    print("=" * 80)
    print("分析强化学习训练后的熵数据")
    print("=" * 80)
    rl_step_entropies = extract_step_entropies(rl_file)
    
    print("\n" + "=" * 80)
    print("分析基线模型的熵数据")
    print("=" * 80)
    baseline_step_entropies = extract_step_entropies(baseline_file)
    
    # 计算前15步的统计数据
    print("\n" + "=" * 80)
    print("计算前15步的统计数据")
    print("=" * 80)
    
    # 获取前15步的数据
    rl_first15_entropies = []
    baseline_first15_entropies = []
    
    for step in range(15):
        if step in rl_step_entropies:
            rl_first15_entropies.extend(rl_step_entropies[step])
        if step in baseline_step_entropies:
            baseline_first15_entropies.extend(baseline_step_entropies[step])
    
    # 计算前15步的平均熵的方差
    rl_variance_15 = np.var(rl_first15_entropies)
    baseline_variance_15 = np.var(baseline_first15_entropies)
    
    print(f"\n【前15步平均熵的方差】")
    print(f"基线模型方差: {baseline_variance_15:.6f}")
    print(f"强化学习后方差: {rl_variance_15:.6f}")
    print(f"方差变化: {rl_variance_15 - baseline_variance_15:.6f}")
    print(f"方差变化率: {(rl_variance_15 - baseline_variance_15) / baseline_variance_15 * 100:.2f}%")
    
    # 计算每个步骤的平均熵（用于确定高熵和低熵步骤）
    rl_step_means = {}
    baseline_step_means = {}
    
    for step in range(15):
        if step in rl_step_entropies:
            rl_step_means[step] = np.mean(rl_step_entropies[step])
        if step in baseline_step_entropies:
            baseline_step_means[step] = np.mean(baseline_step_entropies[step])
    
    # 计算前15步的整体平均熵作为阈值
    rl_threshold = np.mean([rl_step_means[s] for s in range(15) if s in rl_step_means])
    baseline_threshold = np.mean([baseline_step_means[s] for s in range(15) if s in baseline_step_means])
    
    print(f"\n【阈值（前15步整体平均熵）】")
    print(f"基线模型阈值: {baseline_threshold:.6f}")
    print(f"强化学习后阈值: {rl_threshold:.6f}")
    
    # 分类高熵和低熵步骤
    rl_high_entropy_steps = []
    rl_low_entropy_steps = []
    baseline_high_entropy_steps = []
    baseline_low_entropy_steps = []
    
    for step in range(15):
        # 强化学习模型
        if step in rl_step_entropies and step in rl_step_means:
            if rl_step_means[step] > rl_threshold:
                rl_high_entropy_steps.extend(rl_step_entropies[step])
            else:
                rl_low_entropy_steps.extend(rl_step_entropies[step])
        
        # 基线模型
        if step in baseline_step_entropies and step in baseline_step_means:
            if baseline_step_means[step] > baseline_threshold:
                baseline_high_entropy_steps.extend(baseline_step_entropies[step])
            else:
                baseline_low_entropy_steps.extend(baseline_step_entropies[step])
    
    # 计算高熵步骤的方差
    rl_high_variance = np.var(rl_high_entropy_steps) if rl_high_entropy_steps else 0
    baseline_high_variance = np.var(baseline_high_entropy_steps) if baseline_high_entropy_steps else 0
    
    print(f"\n【高熵步骤的方差（步骤平均熵 > 前15步整体平均熵）】")
    print(f"基线模型高熵步骤数: {len(baseline_high_entropy_steps)}")
    print(f"强化学习后高熵步骤数: {len(rl_high_entropy_steps)}")
    print(f"基线模型高熵方差: {baseline_high_variance:.6f}")
    print(f"强化学习后高熵方差: {rl_high_variance:.6f}")
    print(f"高熵方差变化: {rl_high_variance - baseline_high_variance:.6f}")
    if baseline_high_variance > 0:
        print(f"高熵方差变化率: {(rl_high_variance - baseline_high_variance) / baseline_high_variance * 100:.2f}%")
    
    # 计算低熵步骤的方差
    rl_low_variance = np.var(rl_low_entropy_steps) if rl_low_entropy_steps else 0
    baseline_low_variance = np.var(baseline_low_entropy_steps) if baseline_low_entropy_steps else 0
    
    print(f"\n【低熵步骤的方差（步骤平均熵 ≤ 前15步整体平均熵）】")
    print(f"基线模型低熵步骤数: {len(baseline_low_entropy_steps)}")
    print(f"强化学习后低熵步骤数: {len(rl_low_entropy_steps)}")
    print(f"基线模型低熵方差: {baseline_low_variance:.6f}")
    print(f"强化学习后低熵方差: {rl_low_variance:.6f}")
    print(f"低熵方差变化: {rl_low_variance - baseline_low_variance:.6f}")
    if baseline_low_variance > 0:
        print(f"低熵方差变化率: {(rl_low_variance - baseline_low_variance) / baseline_low_variance * 100:.2f}%")
    
    # 绘制对比图
    print("\n" + "=" * 80)
    print("绘制对比图")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Variance Analysis: Before vs After Reinforcement Learning', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 图1：前15步方差对比
    ax1 = axes[0, 0]
    categories = ['Baseline', 'After RL']
    variances = [baseline_variance_15, rl_variance_15]
    colors = ['#3498DB', '#E74C3C']
    bars = ax1.bar(categories, variances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax1.set_title('Variance of First 15 Steps', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加变化率标签
    change_rate = (rl_variance_15 - baseline_variance_15) / baseline_variance_15 * 100
    ax1.text(0.5, max(variances) * 0.9, f'Change: {change_rate:+.2f}%',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 图2：高熵步骤方差对比
    ax2 = axes[0, 1]
    high_variances = [baseline_high_variance, rl_high_variance]
    bars = ax2.bar(categories, high_variances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Variance of High Entropy Steps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, var in zip(bars, high_variances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    if baseline_high_variance > 0:
        high_change_rate = (rl_high_variance - baseline_high_variance) / baseline_high_variance * 100
        ax2.text(0.5, max(high_variances) * 0.9, f'Change: {high_change_rate:+.2f}%',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 图3：低熵步骤方差对比
    ax3 = axes[1, 0]
    low_variances = [baseline_low_variance, rl_low_variance]
    bars = ax3.bar(categories, low_variances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax3.set_title('Variance of Low Entropy Steps', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, var in zip(bars, low_variances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    if baseline_low_variance > 0:
        low_change_rate = (rl_low_variance - baseline_low_variance) / baseline_low_variance * 100
        ax3.text(0.5, max(low_variances) * 0.9, f'Change: {low_change_rate:+.2f}%',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 图4：综合对比（高熵、低熵、整体）
    ax4 = axes[1, 1]
    x = np.arange(3)
    width = 0.35
    
    baseline_vars = [baseline_variance_15, baseline_high_variance, baseline_low_variance]
    rl_vars = [rl_variance_15, rl_high_variance, rl_low_variance]
    
    bars1 = ax4.bar(x - width/2, baseline_vars, width, label='Baseline', 
                    color='#3498DB', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, rl_vars, width, label='After RL',
                    color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax4.set_title('Comprehensive Variance Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['All 15 Steps', 'High Entropy', 'Low Entropy'])
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加信息文本
    info_text = f"Dataset Info:\n"
    info_text += f"• High Entropy: step avg > overall avg\n"
    info_text += f"• Low Entropy: step avg <= overall avg\n"
    info_text += f"• Baseline Threshold: {baseline_threshold:.4f}\n"
    info_text += f"• RL Threshold: {rl_threshold:.4f}\n"
    info_text += f"• Overall Variance Reduction: {change_rate:.2f}%"
    
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 保存图像
    output_file = '/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_full_entropy/variance_change_analysis.png'
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存至: {output_file}")
    plt.close()
    
    # 输出详细的每个步骤的统计
    print("\n" + "=" * 80)
    print("每个步骤的详细统计")
    print("=" * 80)
    print(f"\n{'步骤':<6} {'基线平均':<12} {'RL平均':<12} {'基线方差':<12} {'RL方差':<12} {'类型(基线)':<12} {'类型(RL)':<12}")
    print("-" * 90)
    
    for step in range(15):
        baseline_mean = baseline_step_means.get(step, 0)
        rl_mean = rl_step_means.get(step, 0)
        
        baseline_var = np.var(baseline_step_entropies.get(step, [])) if step in baseline_step_entropies else 0
        rl_var = np.var(rl_step_entropies.get(step, [])) if step in rl_step_entropies else 0
        
        baseline_type = "高熵" if baseline_mean > baseline_threshold else "低熵"
        rl_type = "高熵" if rl_mean > rl_threshold else "低熵"
        
        print(f"{step:<6} {baseline_mean:<12.4f} {rl_mean:<12.4f} {baseline_var:<12.4f} {rl_var:<12.4f} {baseline_type:<12} {rl_type:<12}")
    
    # 保存详细结果到文本文件
    result_file = '/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_full_entropy/variance_change_analysis.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("强化学习前后熵方差变化分析\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【前15步平均熵的方差】\n")
        f.write(f"基线模型方差: {baseline_variance_15:.6f}\n")
        f.write(f"强化学习后方差: {rl_variance_15:.6f}\n")
        f.write(f"方差变化: {rl_variance_15 - baseline_variance_15:.6f}\n")
        f.write(f"方差变化率: {(rl_variance_15 - baseline_variance_15) / baseline_variance_15 * 100:.2f}%\n\n")
        
        f.write("【阈值（前15步整体平均熵）】\n")
        f.write(f"基线模型阈值: {baseline_threshold:.6f}\n")
        f.write(f"强化学习后阈值: {rl_threshold:.6f}\n\n")
        
        f.write("【高熵步骤的方差】\n")
        f.write(f"基线模型高熵步骤数: {len(baseline_high_entropy_steps)}\n")
        f.write(f"强化学习后高熵步骤数: {len(rl_high_entropy_steps)}\n")
        f.write(f"基线模型高熵方差: {baseline_high_variance:.6f}\n")
        f.write(f"强化学习后高熵方差: {rl_high_variance:.6f}\n")
        f.write(f"高熵方差变化: {rl_high_variance - baseline_high_variance:.6f}\n")
        if baseline_high_variance > 0:
            f.write(f"高熵方差变化率: {(rl_high_variance - baseline_high_variance) / baseline_high_variance * 100:.2f}%\n\n")
        
        f.write("【低熵步骤的方差】\n")
        f.write(f"基线模型低熵步骤数: {len(baseline_low_entropy_steps)}\n")
        f.write(f"强化学习后低熵步骤数: {len(rl_low_entropy_steps)}\n")
        f.write(f"基线模型低熵方差: {baseline_low_variance:.6f}\n")
        f.write(f"强化学习后低熵方差: {rl_low_variance:.6f}\n")
        f.write(f"低熵方差变化: {rl_low_variance - baseline_low_variance:.6f}\n")
        if baseline_low_variance > 0:
            f.write(f"低熵方差变化率: {(rl_low_variance - baseline_low_variance) / baseline_low_variance * 100:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("每个步骤的详细统计\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'步骤':<6} {'基线平均':<12} {'RL平均':<12} {'基线方差':<12} {'RL方差':<12} {'类型(基线)':<12} {'类型(RL)':<12}\n")
        f.write("-" * 90 + "\n")
        
        for step in range(15):
            baseline_mean = baseline_step_means.get(step, 0)
            rl_mean = rl_step_means.get(step, 0)
            
            baseline_var = np.var(baseline_step_entropies.get(step, [])) if step in baseline_step_entropies else 0
            rl_var = np.var(rl_step_entropies.get(step, [])) if step in rl_step_entropies else 0
            
            baseline_type = "高熵" if baseline_mean > baseline_threshold else "低熵"
            rl_type = "高熵" if rl_mean > rl_threshold else "低熵"
            
            f.write(f"{step:<6} {baseline_mean:<12.4f} {rl_mean:<12.4f} {baseline_var:<12.4f} {rl_var:<12.4f} {baseline_type:<12} {rl_type:<12}\n")
    
    print(f"\n详细结果已保存至: {result_file}")
    print("\n✅ 分析完成！")


def main():
    parser = argparse.ArgumentParser(description="分析强化学习前后熵方差的变化")
    parser.add_argument("--rl_file", type=str, required=True, help="强化学习训练后的JSON文件路径")
    parser.add_argument("--baseline_file", type=str, required=True, help="基线模型的JSON文件路径")
    
    args = parser.parse_args()
    
    analyze_variance_changes(args.rl_file, args.baseline_file)


if __name__ == "__main__":
    main()

