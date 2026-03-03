#!/usr/bin/env python3
"""
分析评估结果中的步骤划分和策略熵信息

使用方法：
    python analyze_entropy_results.py results.json

功能：
1. 统计正确/错误响应的熵差异
2. 分析每个步骤的熵分布
3. 可视化熵随步骤的变化
4. 计算KL散度与正确性的相关性
"""

import json
import argparse
import numpy as np
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("警告: matplotlib未安装，将跳过可视化功能")
    print("      安装方法: pip install matplotlib")


def load_results(json_file: str) -> dict:
    """加载评估结果JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_entropy_by_correctness(results: list) -> dict:
    """分析正确和错误响应的熵差异"""
    correct_entropies = []
    incorrect_entropies = []
    correct_kl_divs = []
    incorrect_kl_divs = []
    correct_num_steps = []
    incorrect_num_steps = []
    
    for result in results:
        for resp in result['responses']:
            analysis = resp['entropy_analysis']
            overall_mean_entropy = analysis['overall_stats']['overall_mean_entropy']
            num_steps = analysis['overall_stats']['num_steps']
            
            # 计算平均KL散度
            avg_kl_div = np.mean([s['kl_div_uniform'] for s in analysis['steps']])
            
            if resp['is_correct']:
                correct_entropies.append(overall_mean_entropy)
                correct_kl_divs.append(avg_kl_div)
                correct_num_steps.append(num_steps)
            else:
                incorrect_entropies.append(overall_mean_entropy)
                incorrect_kl_divs.append(avg_kl_div)
                incorrect_num_steps.append(num_steps)
    
    return {
        'correct': {
            'entropies': correct_entropies,
            'kl_divs': correct_kl_divs,
            'num_steps': correct_num_steps,
        },
        'incorrect': {
            'entropies': incorrect_entropies,
            'kl_divs': incorrect_kl_divs,
            'num_steps': incorrect_num_steps,
        }
    }


def analyze_entropy_per_step(results: list) -> dict:
    """分析每个步骤位置的熵分布"""
    step_entropies = defaultdict(list)
    step_kl_divs = defaultdict(list)
    
    for result in results:
        for resp in result['responses']:
            analysis = resp['entropy_analysis']
            for step in analysis['steps']:
                step_idx = step['step_index']
                step_entropies[step_idx].append(step['mean_entropy'])
                step_kl_divs[step_idx].append(step['kl_div_uniform'])
    
    return {
        'entropies': step_entropies,
        'kl_divs': step_kl_divs,
    }


def print_statistics(data: dict):
    """打印统计信息"""
    print("\n" + "=" * 70)
    print("策略熵统计分析")
    print("=" * 70)
    
    # 整体统计
    correct = data['correct']
    incorrect = data['incorrect']
    
    print("\n1. 按正确性分类的熵分析")
    print("-" * 70)
    
    if correct['entropies']:
        print(f"\n正确响应 (n={len(correct['entropies'])}):")
        print(f"  平均熵: {np.mean(correct['entropies']):.4f} ± {np.std(correct['entropies']):.4f}")
        print(f"  熵范围: [{np.min(correct['entropies']):.4f}, {np.max(correct['entropies']):.4f}]")
        print(f"  平均KL散度: {np.mean(correct['kl_divs']):.4f} ± {np.std(correct['kl_divs']):.4f}")
        print(f"  平均步骤数: {np.mean(correct['num_steps']):.2f} ± {np.std(correct['num_steps']):.2f}")
    
    if incorrect['entropies']:
        print(f"\n错误响应 (n={len(incorrect['entropies'])}):")
        print(f"  平均熵: {np.mean(incorrect['entropies']):.4f} ± {np.std(incorrect['entropies']):.4f}")
        print(f"  熵范围: [{np.min(incorrect['entropies']):.4f}, {np.max(incorrect['entropies']):.4f}]")
        print(f"  平均KL散度: {np.mean(incorrect['kl_divs']):.4f} ± {np.std(incorrect['kl_divs']):.4f}")
        print(f"  平均步骤数: {np.mean(incorrect['num_steps']):.2f} ± {np.std(incorrect['num_steps']):.2f}")
    
    # 熵差异显著性
    if correct['entropies'] and incorrect['entropies']:
        entropy_diff = np.mean(correct['entropies']) - np.mean(incorrect['entropies'])
        print(f"\n熵差异 (正确 - 错误): {entropy_diff:.4f}")
        if abs(entropy_diff) > 0.1:
            print(f"  → {'正确响应熵更高' if entropy_diff > 0 else '错误响应熵更高'}")
        else:
            print(f"  → 熵差异不明显")
        
        kl_diff = np.mean(correct['kl_divs']) - np.mean(incorrect['kl_divs'])
        print(f"KL散度差异 (正确 - 错误): {kl_diff:.4f}")


def analyze_step_progression(results: list, max_steps: int = 10):
    """分析熵在推理步骤中的变化趋势"""
    print("\n2. 推理步骤中的熵变化")
    print("-" * 70)
    
    step_data = analyze_entropy_per_step(results)
    
    print(f"\n前{max_steps}个步骤的平均熵:")
    for step_idx in range(min(max_steps, len(step_data['entropies']))):
        if step_idx in step_data['entropies']:
            entropies = step_data['entropies'][step_idx]
            kl_divs = step_data['kl_divs'][step_idx]
            print(f"  步骤 {step_idx}: "
                  f"熵={np.mean(entropies):.4f} ± {np.std(entropies):.4f}, "
                  f"KL散度={np.mean(kl_divs):.4f} ± {np.std(kl_divs):.4f}, "
                  f"n={len(entropies)}")
    
    return step_data


def visualize_results(data: dict, step_data: dict, output_prefix: str = "entropy_analysis"):
    """可视化分析结果"""
    if not PLOT_AVAILABLE:
        print("\n提示: 安装matplotlib以启用可视化功能")
        return
    
    print("\n3. 生成可视化图表")
    print("-" * 70)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 正确vs错误的熵分布
    ax = axes[0, 0]
    if data['correct']['entropies'] and data['incorrect']['entropies']:
        ax.hist(data['correct']['entropies'], bins=20, alpha=0.5, label='正确', color='green')
        ax.hist(data['incorrect']['entropies'], bins=20, alpha=0.5, label='错误', color='red')
        ax.set_xlabel('平均策略熵')
        ax.set_ylabel('频数')
        ax.set_title('正确vs错误响应的熵分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. KL散度分布
    ax = axes[0, 1]
    if data['correct']['kl_divs'] and data['incorrect']['kl_divs']:
        ax.hist(data['correct']['kl_divs'], bins=20, alpha=0.5, label='正确', color='green')
        ax.hist(data['incorrect']['kl_divs'], bins=20, alpha=0.5, label='错误', color='red')
        ax.set_xlabel('平均KL散度')
        ax.set_ylabel('频数')
        ax.set_title('正确vs错误响应的KL散度分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. 熵随步骤的变化
    ax = axes[1, 0]
    max_steps = 15
    step_indices = list(range(min(max_steps, len(step_data['entropies']))))
    mean_entropies = []
    std_entropies = []
    
    for idx in step_indices:
        if idx in step_data['entropies']:
            entropies = step_data['entropies'][idx]
            mean_entropies.append(np.mean(entropies))
            std_entropies.append(np.std(entropies))
        else:
            mean_entropies.append(0)
            std_entropies.append(0)
    
    if mean_entropies:
        ax.plot(step_indices, mean_entropies, marker='o', label='平均熵')
        ax.fill_between(step_indices, 
                        np.array(mean_entropies) - np.array(std_entropies),
                        np.array(mean_entropies) + np.array(std_entropies),
                        alpha=0.3)
        ax.set_xlabel('步骤索引')
        ax.set_ylabel('平均策略熵')
        ax.set_title('熵随推理步骤的变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. 步骤数分布
    ax = axes[1, 1]
    if data['correct']['num_steps'] and data['incorrect']['num_steps']:
        ax.hist(data['correct']['num_steps'], bins=range(1, 20), alpha=0.5, label='正确', color='green')
        ax.hist(data['incorrect']['num_steps'], bins=range(1, 20), alpha=0.5, label='错误', color='red')
        ax.set_xlabel('推理步骤数')
        ax.set_ylabel('频数')
        ax.set_title('正确vs错误响应的步骤数分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = f"{output_prefix}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {output_file}")
    
    plt.close()


def find_interesting_cases(results: list, top_k: int = 3):
    """找出有趣的案例"""
    print("\n4. 有趣的案例分析")
    print("-" * 70)
    
    # 找出熵最高和最低的正确响应
    correct_responses = []
    for result in results:
        for resp in result['responses']:
            if resp['is_correct']:
                entropy = resp['entropy_analysis']['overall_stats']['overall_mean_entropy']
                correct_responses.append({
                    'question': result['prompt'][:80] + "...",
                    'entropy': entropy,
                    'num_steps': resp['entropy_analysis']['overall_stats']['num_steps'],
                    'response': resp['response'][:100] + "...",
                })
    
    if correct_responses:
        correct_responses.sort(key=lambda x: x['entropy'])
        
        print(f"\n熵最低的{min(top_k, len(correct_responses))}个正确响应:")
        for i, resp in enumerate(correct_responses[:top_k], 1):
            print(f"\n  案例 {i}:")
            print(f"    问题: {resp['question']}")
            print(f"    平均熵: {resp['entropy']:.4f}")
            print(f"    步骤数: {resp['num_steps']}")
        
        print(f"\n熵最高的{min(top_k, len(correct_responses))}个正确响应:")
        for i, resp in enumerate(correct_responses[-top_k:][::-1], 1):
            print(f"\n  案例 {i}:")
            print(f"    问题: {resp['question']}")
            print(f"    平均熵: {resp['entropy']:.4f}")
            print(f"    步骤数: {resp['num_steps']}")


def main():
    parser = argparse.ArgumentParser(description="分析评估结果中的步骤划分和策略熵信息")
    parser.add_argument("json_file", type=str, help="评估结果JSON文件路径")
    parser.add_argument("--output-prefix", type=str, default="entropy_analysis", 
                       help="输出图表的文件名前缀")
    parser.add_argument("--no-plot", action="store_true", help="不生成可视化图表")
    
    args = parser.parse_args()
    
    # 加载结果
    print(f"正在加载结果文件: {args.json_file}")
    try:
        data = load_results(args.json_file)
    except FileNotFoundError:
        print(f"错误: 文件不存在: {args.json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败: {e}")
        sys.exit(1)
    
    results = data['results']
    print(f"加载了 {len(results)} 个测试样本")
    
    # 整体信息
    print(f"Overall Pass@1: {data['overall_pass@1']:.4f}")
    print(f"总响应数: {data['total_responses']}")
    print(f"正确响应数: {data['total_correct_responses']}")
    
    # 分析熵按正确性分类
    correctness_data = analyze_entropy_by_correctness(results)
    print_statistics(correctness_data)
    
    # 分析步骤演进
    step_data = analyze_step_progression(results)
    
    # 可视化
    if not args.no_plot:
        visualize_results(correctness_data, step_data, args.output_prefix)
    
    # 找出有趣的案例
    find_interesting_cases(results)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

