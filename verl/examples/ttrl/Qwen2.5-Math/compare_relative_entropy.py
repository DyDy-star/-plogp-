#!/usr/bin/env python3
"""对比两个评估结果JSON文件的相对熵"""

import json
import numpy as np
from collections import defaultdict


def analyze_file_entropy(json_file):
    """分析单个JSON文件的熵数据并计算相对熵"""
    print(f"\n{'='*70}")
    print(f"分析文件: {json_file}")
    print(f"{'='*70}")
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取基本信息
    overall_pass1 = data.get('overall_pass@1', 0)
    n_test_samples = data.get('n_test_samples', 0)
    total_correct = data.get('total_correct_responses', 0)
    total_responses = data.get('total_responses', 0)
    
    print(f"Overall Pass@1: {overall_pass1:.4f}")
    print(f"正确率: {total_correct}/{total_responses} = {total_correct/total_responses:.4f}")
    
    # 统计每个步骤的熵值
    step_entropies = defaultdict(list)
    
    # 遍历所有问题和响应
    results = data.get('results', [])
    for result in results:
        responses = result.get('responses', [])
        
        for response in responses:
            entropy_analysis = response.get('entropy_analysis', {})
            
            if not entropy_analysis:
                continue
                
            steps = entropy_analysis.get('steps', [])
            if not steps:
                continue
            
            # 遍历每个步骤
            for step in steps:
                step_index = step.get('step_index')
                mean_entropy = step.get('mean_entropy')
                
                if step_index is not None and mean_entropy is not None:
                    step_entropies[step_index].append(mean_entropy)
    
    # 计算每个步骤的平均熵
    steps = sorted(step_entropies.keys())
    avg_entropies = []
    
    print(f"\n步骤级别的平均熵：")
    print(f"{'步骤':<8} {'平均熵':<12} {'样本数':<10}")
    print("-" * 35)
    
    for step in steps:
        entropies = step_entropies[step]
        avg_entropy = np.mean(entropies)
        avg_entropies.append(avg_entropy)
        print(f"{step:<8} {avg_entropy:<12.4f} {len(entropies):<10}")
    
    # 计算所有步骤的平均熵（基线）
    global_mean_entropy = np.mean(avg_entropies)
    print(f"\n所有步骤的平均熵（基线）: {global_mean_entropy:.4f}")
    
    # 计算每个步骤的相对差异 (relative difference) = (step_entropy - baseline) / baseline
    relative_differences = []
    high_entropy_relative_diffs = []
    low_entropy_relative_diffs = []
    high_entropy_values = []  # 存储高熵步骤的实际熵值
    low_entropy_values = []   # 存储低熵步骤的实际熵值
    
    for i, avg_entropy in enumerate(avg_entropies):
        if global_mean_entropy > 0:
            rel_diff = (avg_entropy - global_mean_entropy) / global_mean_entropy
        else:
            rel_diff = 0
        
        relative_differences.append(rel_diff)
        
        # 分类：高熵步骤（>= 平均熵）和低熵步骤（< 平均熵）
        if avg_entropy >= global_mean_entropy:
            high_entropy_relative_diffs.append(rel_diff)
            high_entropy_values.append(avg_entropy)
        else:
            low_entropy_relative_diffs.append(rel_diff)
            low_entropy_values.append(avg_entropy)
    
    # 计算高熵和低熵步骤的平均相对差异
    high_entropy_mean_rel_diff = np.mean(high_entropy_relative_diffs) if high_entropy_relative_diffs else 0
    low_entropy_mean_rel_diff = np.mean(low_entropy_relative_diffs) if low_entropy_relative_diffs else 0
    
    # 计算高熵和低熵步骤的平均熵值
    high_entropy_mean = np.mean(high_entropy_values) if high_entropy_values else 0
    low_entropy_mean = np.mean(low_entropy_values) if low_entropy_values else 0
    
    # 计算相对熵指标：高熵步骤的平均相对差异 - 低熵步骤的平均相对差异
    relative_entropy_metric = high_entropy_mean_rel_diff - low_entropy_mean_rel_diff
    
    # 计算绝对熵差指标：高熵步骤的平均熵 - 低熵步骤的平均熵
    absolute_entropy_diff = high_entropy_mean - low_entropy_mean
    
    # ========== 新增：计算各种比值指标 ==========
    n_steps = len(avg_entropies)
    
    # 1. 前7步和后8步的平均熵
    first_7_entropies = avg_entropies[:7] if n_steps >= 7 else avg_entropies
    last_8_entropies = avg_entropies[-8:] if n_steps >= 8 else avg_entropies
    first_7_mean_entropy = np.mean(first_7_entropies)
    last_8_mean_entropy = np.mean(last_8_entropies)
    
    # 2. 前7步和后8步的平均relative difference
    first_7_rel_diffs = relative_differences[:7] if n_steps >= 7 else relative_differences
    last_8_rel_diffs = relative_differences[-8:] if n_steps >= 8 else relative_differences
    first_7_mean_rel_diff = np.mean(first_7_rel_diffs)
    last_8_mean_rel_diff = np.mean(last_8_rel_diffs)
    
    # 计算比值（避免除以零）
    # 比值1：前7步平均熵 / 后8步平均熵
    entropy_ratio_first7_last8 = first_7_mean_entropy / last_8_mean_entropy if last_8_mean_entropy != 0 else float('inf')
    
    # 比值2：前7步平均relative difference / 后8步平均relative difference
    # 注意：relative difference可能为负，使用绝对值或直接比值
    rel_diff_ratio_first7_last8 = first_7_mean_rel_diff / last_8_mean_rel_diff if last_8_mean_rel_diff != 0 else float('inf')
    
    # 比值3：高熵步骤平均熵 / 低熵步骤平均熵
    high_low_entropy_ratio = high_entropy_mean / low_entropy_mean if low_entropy_mean != 0 else float('inf')
    
    # 比值4：高熵步骤平均relative difference / 低熵步骤平均relative difference
    # 注意：低熵步骤的relative difference通常为负值
    high_low_rel_diff_ratio = high_entropy_mean_rel_diff / abs(low_entropy_mean_rel_diff) if low_entropy_mean_rel_diff != 0 else float('inf')
    
    print(f"\n相对差异分析：")
    print(f"  高熵步骤数量（>= {global_mean_entropy:.4f}）: {len(high_entropy_relative_diffs)}")
    print(f"  高熵步骤的平均熵: {high_entropy_mean:.4f}")
    print(f"  高熵步骤的平均相对差异: {high_entropy_mean_rel_diff:.4f}")
    print(f"  低熵步骤数量（< {global_mean_entropy:.4f}）: {len(low_entropy_relative_diffs)}")
    print(f"  低熵步骤的平均熵: {low_entropy_mean:.4f}")
    print(f"  低熵步骤的平均相对差异: {low_entropy_mean_rel_diff:.4f}")
    print(f"  📊 绝对熵差指标（高熵步骤平均熵 - 低熵步骤平均熵）: {absolute_entropy_diff:.4f}")
    print(f"  🎯 相对熵指标（高熵平均相对差异 - 低熵平均相对差异）: {relative_entropy_metric:.4f}")
    
    print(f"\n📐 比值分析（前7步 vs 后8步）：")
    print(f"  前7步平均熵: {first_7_mean_entropy:.4f}")
    print(f"  后8步平均熵: {last_8_mean_entropy:.4f}")
    print(f"  🔢 比值1（前7步平均熵 / 后8步平均熵）: {entropy_ratio_first7_last8:.4f}")
    print(f"  前7步平均relative difference: {first_7_mean_rel_diff:.4f}")
    print(f"  后8步平均relative difference: {last_8_mean_rel_diff:.4f}")
    print(f"  🔢 比值2（前7步平均rel_diff / 后8步平均rel_diff）: {rel_diff_ratio_first7_last8:.4f}")
    
    print(f"\n📐 比值分析（高熵步骤 vs 低熵步骤）：")
    print(f"  🔢 比值3（高熵步骤平均熵 / 低熵步骤平均熵）: {high_low_entropy_ratio:.4f}")
    print(f"  🔢 比值4（高熵步骤平均rel_diff / |低熵步骤平均rel_diff|）: {high_low_rel_diff_ratio:.4f}")
    
    return {
        'file': json_file,
        'overall_pass1': overall_pass1,
        'total_correct': total_correct,
        'total_responses': total_responses,
        'accuracy': total_correct / total_responses if total_responses > 0 else 0,
        'global_mean_entropy': global_mean_entropy,
        'high_entropy_mean': high_entropy_mean,
        'low_entropy_mean': low_entropy_mean,
        'absolute_entropy_diff': absolute_entropy_diff,
        'high_entropy_mean_rel_diff': high_entropy_mean_rel_diff,
        'low_entropy_mean_rel_diff': low_entropy_mean_rel_diff,
        'relative_entropy_metric': relative_entropy_metric,
        'n_high_entropy_steps': len(high_entropy_relative_diffs),
        'n_low_entropy_steps': len(low_entropy_relative_diffs),
        'n_steps': len(steps),
        'step_entropies': avg_entropies,
        'relative_differences': relative_differences,
        # 新增比值指标
        'first_7_mean_entropy': first_7_mean_entropy,
        'last_8_mean_entropy': last_8_mean_entropy,
        'first_7_mean_rel_diff': first_7_mean_rel_diff,
        'last_8_mean_rel_diff': last_8_mean_rel_diff,
        'entropy_ratio_first7_last8': entropy_ratio_first7_last8,
        'rel_diff_ratio_first7_last8': rel_diff_ratio_first7_last8,
        'high_low_entropy_ratio': high_low_entropy_ratio,
        'high_low_rel_diff_ratio': high_low_rel_diff_ratio,
    }


def compare_files(file1, file2):
    """对比两个文件的相对熵"""
    
    # 分析两个文件
    result1 = analyze_file_entropy(file1)
    result2 = analyze_file_entropy(file2)
    
    # 打印对比结果
    print(f"\n{'='*70}")
    print(f"对比结果")
    print(f"{'='*70}")
    
    print(f"\n文件1: {file1.split('/')[-1]}")
    print(f"  - Pass@1: {result1['overall_pass1']:.4f}")
    print(f"  - 准确率: {result1['accuracy']:.4f}")
    print(f"  - 全局平均熵: {result1['global_mean_entropy']:.4f}")
    print(f"  - 高熵步骤平均熵: {result1['high_entropy_mean']:.4f}")
    print(f"  - 低熵步骤平均熵: {result1['low_entropy_mean']:.4f}")
    print(f"  - 📊 绝对熵差指标: {result1['absolute_entropy_diff']:.4f}")
    print(f"  - 高熵步骤平均相对差异: {result1['high_entropy_mean_rel_diff']:.4f}")
    print(f"  - 低熵步骤平均相对差异: {result1['low_entropy_mean_rel_diff']:.4f}")
    print(f"  - 🎯 相对熵指标: {result1['relative_entropy_metric']:.4f}")
    
    print(f"\n文件2: {file2.split('/')[-1]}")
    print(f"  - Pass@1: {result2['overall_pass1']:.4f}")
    print(f"  - 准确率: {result2['accuracy']:.4f}")
    print(f"  - 全局平均熵: {result2['global_mean_entropy']:.4f}")
    print(f"  - 高熵步骤平均熵: {result2['high_entropy_mean']:.4f}")
    print(f"  - 低熵步骤平均熵: {result2['low_entropy_mean']:.4f}")
    print(f"  - 📊 绝对熵差指标: {result2['absolute_entropy_diff']:.4f}")
    print(f"  - 高熵步骤平均相对差异: {result2['high_entropy_mean_rel_diff']:.4f}")
    print(f"  - 低熵步骤平均相对差异: {result2['low_entropy_mean_rel_diff']:.4f}")
    print(f"  - 🎯 相对熵指标: {result2['relative_entropy_metric']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"最终对比")
    print(f"{'='*70}")
    
    # 对比绝对熵差指标
    abs_diff = result2['absolute_entropy_diff'] - result1['absolute_entropy_diff']
    
    print(f"\n📊 绝对熵差指标对比（高熵步骤平均熵 - 低熵步骤平均熵）：")
    if result2['absolute_entropy_diff'] > result1['absolute_entropy_diff']:
        print(f"✅ 文件2的绝对熵差更大！")
        print(f"   文件2绝对熵差: {result2['absolute_entropy_diff']:.4f}")
        print(f"   文件1绝对熵差: {result1['absolute_entropy_diff']:.4f}")
        print(f"   差值: +{abs_diff:.4f}")
    elif result1['absolute_entropy_diff'] > result2['absolute_entropy_diff']:
        print(f"✅ 文件1的绝对熵差更大！")
        print(f"   文件1绝对熵差: {result1['absolute_entropy_diff']:.4f}")
        print(f"   文件2绝对熵差: {result2['absolute_entropy_diff']:.4f}")
        print(f"   差值: +{-abs_diff:.4f}")
    else:
        print(f"⚖️ 两个文件的绝对熵差相同: {result1['absolute_entropy_diff']:.4f}")
    
    # 对比相对熵指标
    rel_diff = result2['relative_entropy_metric'] - result1['relative_entropy_metric']
    
    print(f"\n🎯 相对熵指标对比（高熵平均相对差异 - 低熵平均相对差异）：")
    if result2['relative_entropy_metric'] > result1['relative_entropy_metric']:
        print(f"✅ 文件2的相对熵指标更大！")
        print(f"   文件2相对熵指标: {result2['relative_entropy_metric']:.4f}")
        print(f"   文件1相对熵指标: {result1['relative_entropy_metric']:.4f}")
        print(f"   差值: +{rel_diff:.4f}")
    elif result1['relative_entropy_metric'] > result2['relative_entropy_metric']:
        print(f"✅ 文件1的相对熵指标更大！")
        print(f"   文件1相对熵指标: {result1['relative_entropy_metric']:.4f}")
        print(f"   文件2相对熵指标: {result2['relative_entropy_metric']:.4f}")
        print(f"   差值: +{-rel_diff:.4f}")
    else:
        print(f"⚖️ 两个文件的相对熵指标相同: {result1['relative_entropy_metric']:.4f}")
    
    # 对比Pass@1
    print(f"\nPass@1对比：")
    if result2['overall_pass1'] > result1['overall_pass1']:
        print(f"  文件2的Pass@1更高: {result2['overall_pass1']:.4f} vs {result1['overall_pass1']:.4f}")
        print(f"  提升: +{(result2['overall_pass1'] - result1['overall_pass1']):.4f}")
    else:
        print(f"  文件1的Pass@1更高: {result1['overall_pass1']:.4f} vs {result2['overall_pass1']:.4f}")
        print(f"  提升: +{(result1['overall_pass1'] - result2['overall_pass1']):.4f}")
    
    # ========== 新增：比值指标对比 ==========
    print(f"\n{'='*70}")
    print(f"📐 比值指标对比")
    print(f"{'='*70}")
    
    print(f"\n🔢 比值1：前7步平均熵 / 后8步平均熵")
    print(f"   文件1: {result1['first_7_mean_entropy']:.4f} / {result1['last_8_mean_entropy']:.4f} = {result1['entropy_ratio_first7_last8']:.4f}")
    print(f"   文件2: {result2['first_7_mean_entropy']:.4f} / {result2['last_8_mean_entropy']:.4f} = {result2['entropy_ratio_first7_last8']:.4f}")
    ratio_diff_1 = result2['entropy_ratio_first7_last8'] - result1['entropy_ratio_first7_last8']
    print(f"   差异: {ratio_diff_1:+.4f}")
    
    print(f"\n🔢 比值2：前7步平均relative difference / 后8步平均relative difference")
    print(f"   文件1: {result1['first_7_mean_rel_diff']:.4f} / {result1['last_8_mean_rel_diff']:.4f} = {result1['rel_diff_ratio_first7_last8']:.4f}")
    print(f"   文件2: {result2['first_7_mean_rel_diff']:.4f} / {result2['last_8_mean_rel_diff']:.4f} = {result2['rel_diff_ratio_first7_last8']:.4f}")
    ratio_diff_2 = result2['rel_diff_ratio_first7_last8'] - result1['rel_diff_ratio_first7_last8']
    print(f"   差异: {ratio_diff_2:+.4f}")
    
    print(f"\n🔢 比值3：高熵步骤平均熵 / 低熵步骤平均熵")
    print(f"   文件1: {result1['high_entropy_mean']:.4f} / {result1['low_entropy_mean']:.4f} = {result1['high_low_entropy_ratio']:.4f}")
    print(f"   文件2: {result2['high_entropy_mean']:.4f} / {result2['low_entropy_mean']:.4f} = {result2['high_low_entropy_ratio']:.4f}")
    ratio_diff_3 = result2['high_low_entropy_ratio'] - result1['high_low_entropy_ratio']
    print(f"   差异: {ratio_diff_3:+.4f}")
    
    print(f"\n🔢 比值4：高熵步骤平均rel_diff / |低熵步骤平均rel_diff|")
    print(f"   文件1: {result1['high_entropy_mean_rel_diff']:.4f} / {abs(result1['low_entropy_mean_rel_diff']):.4f} = {result1['high_low_rel_diff_ratio']:.4f}")
    print(f"   文件2: {result2['high_entropy_mean_rel_diff']:.4f} / {abs(result2['low_entropy_mean_rel_diff']):.4f} = {result2['high_low_rel_diff_ratio']:.4f}")
    ratio_diff_4 = result2['high_low_rel_diff_ratio'] - result1['high_low_rel_diff_ratio']
    print(f"   差异: {ratio_diff_4:+.4f}")
    
    print(f"\n{'='*70}")
    print(f"📊 比值指标汇总表")
    print(f"{'='*70}")
    print(f"{'指标':<45} {'文件1':<12} {'文件2':<12} {'差异':<12}")
    print("-" * 80)
    print(f"{'比值1: 前7步/后8步平均熵':<45} {result1['entropy_ratio_first7_last8']:<12.4f} {result2['entropy_ratio_first7_last8']:<12.4f} {ratio_diff_1:<+12.4f}")
    print(f"{'比值2: 前7步/后8步平均rel_diff':<45} {result1['rel_diff_ratio_first7_last8']:<12.4f} {result2['rel_diff_ratio_first7_last8']:<12.4f} {ratio_diff_2:<+12.4f}")
    print(f"{'比值3: 高熵/低熵步骤平均熵':<45} {result1['high_low_entropy_ratio']:<12.4f} {result2['high_low_entropy_ratio']:<12.4f} {ratio_diff_3:<+12.4f}")
    print(f"{'比值4: 高熵/|低熵|步骤平均rel_diff':<45} {result1['high_low_rel_diff_ratio']:<12.4f} {result2['high_low_rel_diff_ratio']:<12.4f} {ratio_diff_4:<+12.4f}")
    
    print(f"\n结论：")
    print(f"  更高的相对熵指标意味着模型在推理过程中的熵变化更明显，")
    print(f"  即高熵步骤相对基线更高，低熵步骤相对基线更低，")
    print(f"  表现为更强的早期探索和后期收敛特性。")
    print(f"\n  比值越大表示早期/高熵步骤与后期/低熵步骤之间的差异越显著。")


def compare_three_files(file1, file2, file3, labels=None):
    """对比三个文件的相对熵"""
    
    if labels is None:
        labels = ["基础模型", "训练模型v1", "训练模型v2（最新）"]
    
    # 分析三个文件
    result1 = analyze_file_entropy(file1)
    result2 = analyze_file_entropy(file2)
    result3 = analyze_file_entropy(file3)
    
    results = [result1, result2, result3]
    files = [file1, file2, file3]
    
    # 打印对比表格
    print(f"\n{'='*100}")
    print(f"三模型对比总结")
    print(f"{'='*100}")
    
    print(f"\n{'指标':<20} {labels[0]:<25} {labels[1]:<25} {labels[2]:<25}")
    print("-" * 100)
    
    # Pass@1
    print(f"{'Pass@1':<20} {result1['overall_pass1']:<25.4f} {result2['overall_pass1']:<25.4f} {result3['overall_pass1']:<25.4f}")
    
    # 准确率
    print(f"{'准确率':<20} {result1['accuracy']:<25.4f} {result2['accuracy']:<25.4f} {result3['accuracy']:<25.4f}")
    
    # 全局平均熵
    print(f"{'全局平均熵':<20} {result1['global_mean_entropy']:<25.4f} {result2['global_mean_entropy']:<25.4f} {result3['global_mean_entropy']:<25.4f}")
    
    # 高熵步骤平均相对差异
    print(f"{'高熵步骤平均相对差异':<20} {result1['high_entropy_mean_rel_diff']:<25.4f} {result2['high_entropy_mean_rel_diff']:<25.4f} {result3['high_entropy_mean_rel_diff']:<25.4f}")
    
    # 低熵步骤平均相对差异
    print(f"{'低熵步骤平均相对差异':<20} {result1['low_entropy_mean_rel_diff']:<25.4f} {result2['low_entropy_mean_rel_diff']:<25.4f} {result3['low_entropy_mean_rel_diff']:<25.4f}")
    
    # 🎯 相对熵指标
    print(f"{'🎯 相对熵指标':<20} {result1['relative_entropy_metric']:<25.4f} {result2['relative_entropy_metric']:<25.4f} {result3['relative_entropy_metric']:<25.4f}")
    
    # 找出最佳结果
    print(f"\n{'='*100}")
    print(f"关键指标排名")
    print(f"{'='*100}")
    
    # Pass@1排名
    pass1_sorted = sorted(enumerate(results), key=lambda x: x[1]['overall_pass1'], reverse=True)
    print(f"\n✅ Pass@1排名：")
    for rank, (idx, result) in enumerate(pass1_sorted, 1):
        print(f"  {rank}. {labels[idx]}: {result['overall_pass1']:.4f}")
    
    # 相对熵指标排名
    relative_entropy_sorted = sorted(enumerate(results), key=lambda x: x[1]['relative_entropy_metric'], reverse=True)
    print(f"\n🎯 相对熵指标排名：")
    for rank, (idx, result) in enumerate(relative_entropy_sorted, 1):
        print(f"  {rank}. {labels[idx]}: {result['relative_entropy_metric']:.4f}")
    
    # 综合分析
    print(f"\n{'='*100}")
    print(f"综合分析")
    print(f"{'='*100}")
    
    # 相对熵指标变化
    print(f"\n📊 相对熵指标变化趋势：")
    print(f"  {labels[0]} → {labels[1]}: {result1['relative_entropy_metric']:.4f} → {result2['relative_entropy_metric']:.4f} "
          f"(变化: {result2['relative_entropy_metric']-result1['relative_entropy_metric']:+.4f})")
    print(f"  {labels[1]} → {labels[2]}: {result2['relative_entropy_metric']:.4f} → {result3['relative_entropy_metric']:.4f} "
          f"(变化: {result3['relative_entropy_metric']-result2['relative_entropy_metric']:+.4f})")
    print(f"  {labels[0]} → {labels[2]}: {result1['relative_entropy_metric']:.4f} → {result3['relative_entropy_metric']:.4f} "
          f"(变化: {result3['relative_entropy_metric']-result1['relative_entropy_metric']:+.4f})")
    
    # Pass@1变化
    print(f"\n📈 Pass@1变化趋势：")
    print(f"  {labels[0]} → {labels[1]}: {result1['overall_pass1']:.4f} → {result2['overall_pass1']:.4f} "
          f"(变化: {result2['overall_pass1']-result1['overall_pass1']:+.4f}, {(result2['overall_pass1']-result1['overall_pass1'])/result1['overall_pass1']*100:+.2f}%)")
    print(f"  {labels[1]} → {labels[2]}: {result2['overall_pass1']:.4f} → {result3['overall_pass1']:.4f} "
          f"(变化: {result3['overall_pass1']-result2['overall_pass1']:+.4f}, {(result3['overall_pass1']-result2['overall_pass1'])/result2['overall_pass1']*100:+.2f}%)")
    print(f"  {labels[0]} → {labels[2]}: {result1['overall_pass1']:.4f} → {result3['overall_pass1']:.4f} "
          f"(变化: {result3['overall_pass1']-result1['overall_pass1']:+.4f}, {(result3['overall_pass1']-result1['overall_pass1'])/result1['overall_pass1']*100:+.2f}%)")
    
    # 最终结论
    print(f"\n{'='*100}")
    print(f"🎯 结论")
    print(f"{'='*100}")
    
    best_pass1_idx = pass1_sorted[0][0]
    best_relative_entropy_idx = relative_entropy_sorted[0][0]
    
    if best_pass1_idx == best_relative_entropy_idx:
        print(f"\n🏆 {labels[best_pass1_idx]} 在准确率和相对熵指标两个维度上都表现最佳！")
        print(f"   - Pass@1: {results[best_pass1_idx]['overall_pass1']:.4f}")
        print(f"   - 相对熵指标: {results[best_pass1_idx]['relative_entropy_metric']:.4f}")
    else:
        print(f"\n📊 不同维度表现各异：")
        print(f"   准确率最佳: {labels[best_pass1_idx]} (Pass@1: {results[best_pass1_idx]['overall_pass1']:.4f})")
        print(f"   相对熵指标最佳: {labels[best_relative_entropy_idx]} (相对熵指标: {results[best_relative_entropy_idx]['relative_entropy_metric']:.4f})")
        print(f"\n💡 需要根据具体目标选择最适合的模型：")
        print(f"   - 如果优先考虑准确率 → 选择 {labels[best_pass1_idx]}")
        print(f"   - 如果优先考虑推理多样性 → 选择 {labels[best_relative_entropy_idx]}")
    
    return results


if __name__ == "__main__":
    # 对比两个AIME评估结果
    file1 = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260204_041649.json"
    file2 = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260204_043201.json"
    
    # 对比两个文件
    compare_files(file1, file2)
