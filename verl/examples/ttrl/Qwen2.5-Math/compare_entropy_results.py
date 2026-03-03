#!/usr/bin/env python3
"""
比较两种熵计算方式的评估结果

用法:
    python compare_entropy_results.py \
        --old eval_results_entropy/result.json \
        --new eval_results_full_entropy/result.json
"""

import json
import argparse
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(file_path: str) -> Dict:
    """加载评估结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_entropies(results: Dict, entropy_type: str = "token") -> Dict:
    """
    提取熵统计信息
    
    Args:
        results: 评估结果字典
        entropy_type: "token" 或 "step"
    """
    all_token_entropies = []
    all_step_mean_entropies = []
    all_early_avg = []
    all_late_avg = []
    all_entropy_rewards = []
    
    for sample in results.get("results", []):
        for response in sample.get("responses", []):
            entropy_analysis = response.get("entropy_analysis", {})
            
            # 步骤级统计
            steps = entropy_analysis.get("steps", [])
            for step in steps:
                step_mean = step.get("mean_entropy")
                if step_mean is not None:
                    all_step_mean_entropies.append(step_mean)
                
                # Token级别
                token_entropies = step.get("token_entropies", [])
                all_token_entropies.extend(token_entropies)
            
            # 整体统计
            overall_stats = entropy_analysis.get("overall_stats", {})
            early_avg = overall_stats.get("early_avg_entropy")
            late_avg = overall_stats.get("late_avg_entropy")
            entropy_reward = overall_stats.get("entropy_reward")
            
            if early_avg is not None:
                all_early_avg.append(early_avg)
            if late_avg is not None:
                all_late_avg.append(late_avg)
            if entropy_reward is not None:
                all_entropy_rewards.append(entropy_reward)
    
    return {
        "token_entropies": all_token_entropies,
        "step_mean_entropies": all_step_mean_entropies,
        "early_avg_entropies": all_early_avg,
        "late_avg_entropies": all_late_avg,
        "entropy_rewards": all_entropy_rewards,
    }


def print_statistics(data: Dict, name: str):
    """打印统计信息"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    for key, values in data.items():
        if not values:
            continue
        
        values = np.array(values)
        print(f"\n{key}:")
        print(f"  数量: {len(values)}")
        print(f"  均值: {np.mean(values):.4f}")
        print(f"  标准差: {np.std(values):.4f}")
        print(f"  最小值: {np.min(values):.4f}")
        print(f"  最大值: {np.max(values):.4f}")
        print(f"  中位数: {np.median(values):.4f}")


def compare_results(old_data: Dict, new_data: Dict):
    """对比两种方法的结果"""
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")
    
    comparisons = [
        ("Token级平均熵", "token_entropies"),
        ("步骤级平均熵", "step_mean_entropies"),
        ("前12步平均熵", "early_avg_entropies"),
        ("后3步平均熵", "late_avg_entropies"),
        ("熵奖励 (前12-后3)", "entropy_rewards"),
    ]
    
    for name, key in comparisons:
        old_values = np.array(old_data.get(key, []))
        new_values = np.array(new_data.get(key, []))
        
        if len(old_values) == 0 or len(new_values) == 0:
            continue
        
        old_mean = np.mean(old_values)
        new_mean = np.mean(new_values)
        ratio = new_mean / old_mean if old_mean != 0 else float('inf')
        diff = new_mean - old_mean
        
        print(f"\n{name}:")
        print(f"  旧版 (top-5): {old_mean:.4f}")
        print(f"  新版 (完整): {new_mean:.4f}")
        print(f"  差异: {diff:+.4f} ({(ratio-1)*100:+.1f}%)")
        print(f"  新版/旧版比值: {ratio:.2f}x")


def plot_comparison(old_data: Dict, new_data: Dict, output_dir: str):
    """绘制对比图"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体（如果可用）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 1. Token级熵分布对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) Token级熵直方图
    ax = axes[0, 0]
    old_token = old_data.get("token_entropies", [])
    new_token = new_data.get("token_entropies", [])
    
    ax.hist(old_token, bins=50, alpha=0.5, label='Top-5 Approx', color='red')
    ax.hist(new_token, bins=50, alpha=0.5, label='Full Vocab', color='blue')
    ax.set_xlabel('Token Entropy (nats)')
    ax.set_ylabel('Frequency')
    ax.set_title('Token-Level Entropy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (2) 步骤级熵对比
    ax = axes[0, 1]
    old_step = old_data.get("step_mean_entropies", [])
    new_step = new_data.get("step_mean_entropies", [])
    
    ax.hist(old_step, bins=50, alpha=0.5, label='Top-5 Approx', color='red')
    ax.hist(new_step, bins=50, alpha=0.5, label='Full Vocab', color='blue')
    ax.set_xlabel('Step Mean Entropy (nats)')
    ax.set_ylabel('Frequency')
    ax.set_title('Step-Level Mean Entropy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (3) 前12步 vs 后3步对比
    ax = axes[1, 0]
    x = ['Early (12 steps)', 'Late (3 steps)']
    old_vals = [
        np.mean(old_data.get("early_avg_entropies", [])),
        np.mean(old_data.get("late_avg_entropies", []))
    ]
    new_vals = [
        np.mean(new_data.get("early_avg_entropies", [])),
        np.mean(new_data.get("late_avg_entropies", []))
    ]
    
    x_pos = np.arange(len(x))
    width = 0.35
    
    ax.bar(x_pos - width/2, old_vals, width, label='Top-5 Approx', color='red', alpha=0.7)
    ax.bar(x_pos + width/2, new_vals, width, label='Full Vocab', color='blue', alpha=0.7)
    ax.set_ylabel('Mean Entropy (nats)')
    ax.set_title('Early vs Late Step Entropy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # (4) 熵奖励分布对比
    ax = axes[1, 1]
    old_reward = old_data.get("entropy_rewards", [])
    new_reward = new_data.get("entropy_rewards", [])
    
    ax.hist(old_reward, bins=50, alpha=0.5, label='Top-5 Approx', color='red')
    ax.hist(new_reward, bins=50, alpha=0.5, label='Full Vocab', color='blue')
    ax.set_xlabel('Entropy Reward (early - late)')
    ax.set_ylabel('Frequency')
    ax.set_title('Entropy Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plot_path = output_path / "entropy_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存到: {plot_path}")
    
    # 2. 散点图：新版 vs 旧版
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (1) 步骤级熵散点图
    ax = axes[0]
    old_step = np.array(old_data.get("step_mean_entropies", []))
    new_step = np.array(new_data.get("step_mean_entropies", []))
    
    if len(old_step) == len(new_step):
        ax.scatter(old_step, new_step, alpha=0.3, s=10)
        
        # 添加y=x参考线
        max_val = max(np.max(old_step), np.max(new_step))
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
        
        # 拟合线性回归
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(old_step, new_step)
        x_line = np.array([0, max_val])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'b-', linewidth=2, 
                label=f'Linear fit (R²={r_value**2:.3f})')
        
        ax.set_xlabel('Top-5 Approx Entropy')
        ax.set_ylabel('Full Vocab Entropy')
        ax.set_title('Step-Level Entropy Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # (2) 熵奖励散点图
    ax = axes[1]
    old_reward = np.array(old_data.get("entropy_rewards", []))
    new_reward = np.array(new_data.get("entropy_rewards", []))
    
    if len(old_reward) == len(new_reward):
        ax.scatter(old_reward, new_reward, alpha=0.3, s=20)
        
        # 添加y=x参考线
        min_val = min(np.min(old_reward), np.min(new_reward))
        max_val = max(np.max(old_reward), np.max(new_reward))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        # 拟合线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(old_reward, new_reward)
        x_line = np.array([min_val, max_val])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'b-', linewidth=2, 
                label=f'Linear fit (R²={r_value**2:.3f})')
        
        ax.set_xlabel('Top-5 Approx Entropy Reward')
        ax.set_ylabel('Full Vocab Entropy Reward')
        ax.set_title('Entropy Reward Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plot_path = output_path / "entropy_correlation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"相关性图已保存到: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="比较两种熵计算方式的评估结果")
    parser.add_argument("--old", type=str, required=True, help="旧版结果文件（top-5近似）")
    parser.add_argument("--new", type=str, required=True, help="新版结果文件（完整词表）")
    parser.add_argument("--output_dir", type=str, default="./comparison_plots", help="输出目录")
    
    args = parser.parse_args()
    
    print("加载评估结果...")
    old_results = load_results(args.old)
    new_results = load_results(args.new)
    
    print(f"\n旧版 (top-5近似): {args.old}")
    print(f"  Pass@1: {old_results.get('overall_pass@1', 'N/A')}")
    print(f"  测试样本数: {old_results.get('n_test_samples', 'N/A')}")
    
    print(f"\n新版 (完整词表): {args.new}")
    print(f"  Pass@1: {new_results.get('overall_pass@1', 'N/A')}")
    print(f"  测试样本数: {new_results.get('n_test_samples', 'N/A')}")
    print(f"  熵类型: {new_results.get('entropy_type', 'N/A')}")
    
    print("\n提取熵统计信息...")
    old_data = extract_entropies(old_results)
    new_data = extract_entropies(new_results)
    
    # 打印统计信息
    print_statistics(old_data, "旧版评估（top-5近似）")
    print_statistics(new_data, "新版评估（完整词表）")
    
    # 对比分析
    compare_results(old_data, new_data)
    
    # 绘制对比图
    print("\n生成对比图...")
    plot_comparison(old_data, new_data, args.output_dir)
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)
    
    # 关键发现总结
    print("\n关键发现:")
    old_mean = np.mean(old_data.get("token_entropies", []))
    new_mean = np.mean(new_data.get("token_entropies", []))
    ratio = new_mean / old_mean if old_mean != 0 else 0
    
    print(f"1. 完整词表的熵值约为top-5近似的 {ratio:.2f} 倍")
    print(f"2. 旧版平均token熵: {old_mean:.4f} nats")
    print(f"3. 新版平均token熵: {new_mean:.4f} nats")
    print(f"4. top-5近似低估了约 {(1-1/ratio)*100:.1f}% 的熵值")
    print(f"\n推荐: 使用新版（完整词表）以获得与训练一致的熵计算结果")


if __name__ == "__main__":
    main()

