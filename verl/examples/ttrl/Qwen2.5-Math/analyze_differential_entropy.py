#!/usr/bin/env python3
"""分析两个评估结果JSON文件的差分熵（Differential Entropy）

差分熵计算规则：
- 高熵步骤：
  - 第1个高熵步骤：差分熵 = 0
  - 第2个高熵步骤：差分熵 = |entropy_step2 - entropy_step1|
  - 第3个高熵步骤：差分熵 = |entropy_step3 - avg(entropy_step1, entropy_step2)|
  - 以此类推...

- 低熵步骤：
  - 从高熵切换到低熵时重置，第一个低熵步骤：差分熵 = 0
  - 第2个低熵步骤：差分熵 = |entropy_step2 - entropy_step1|
  - 以此类推...
"""

import json
import numpy as np
from collections import defaultdict


def calculate_differential_entropy_for_response(steps, overall_mean_entropy=None):
    """
    为单个response计算每个步骤的差分熵
    
    参数：
        steps: list of step dicts, each containing 'step_index' and 'mean_entropy'
        overall_mean_entropy: token加权的平均熵 = sum(所有token熵) / 总token数
                              如果提供，则使用这个作为阈值；否则使用步骤级平均
    
    返回：
        dict containing analysis results
    """
    if not steps:
        return None
    
    # 按step_index排序
    sorted_steps = sorted(steps, key=lambda x: x.get('step_index', 0))
    
    # 使用传入的overall_mean_entropy作为阈值（这是token加权的平均熵）
    # 如果没有提供，则退化为步骤级平均
    if overall_mean_entropy is not None:
        threshold = overall_mean_entropy
    else:
        step_entropies = [step.get('mean_entropy', 0) for step in sorted_steps]
        threshold = np.mean(step_entropies)
    
    # 使用 > threshold（严格大于）来划分高熵步骤
    # 这样当 entropy == threshold 时归为低熵，避免全部变成高熵的问题
    # 分类步骤为高熵或低熵
    high_entropy_steps = []  # 存储 (step_index, entropy)
    low_entropy_steps = []   # 存储 (step_index, entropy)
    
    for step in sorted_steps:
        step_index = step.get('step_index')
        entropy = step.get('mean_entropy', 0)
        
        # 使用 > threshold（严格大于）来划分
        if entropy > threshold:
            high_entropy_steps.append((step_index, entropy))
        else:
            low_entropy_steps.append((step_index, entropy))
    
    # 如果所有步骤都在一边（比如全部熵值相等），则需要特殊处理
    # 在这种情况下，全部归为低熵步骤
    if len(high_entropy_steps) == 0 and len(low_entropy_steps) > 0:
        # 所有步骤熵 <= threshold，保持现状（全部为低熵）
        pass
    elif len(low_entropy_steps) == 0 and len(high_entropy_steps) > 0:
        # 这种情况理论上不应该发生（因为用的是 >），但以防万一
        # 如果所有步骤熵 > threshold，说明 threshold 计算有问题
        pass
    
    # 计算高熵步骤的差分熵
    high_diff_entropies = []
    for i, (step_idx, entropy) in enumerate(high_entropy_steps):
        if i == 0:
            diff_entropy = 0
        else:
            # 计算前面所有高熵步骤的平均值
            prev_entropies = [e for _, e in high_entropy_steps[:i]]
            prev_mean = np.mean(prev_entropies)
            diff_entropy = abs(entropy - prev_mean)
        high_diff_entropies.append({
            'step_index': step_idx,
            'entropy': entropy,
            'diff_entropy': diff_entropy,
            'type': 'high'
        })
    
    # 计算低熵步骤的差分熵（需要考虑阻断）
    low_diff_entropies = []
    
    # 找出低熵步骤中的阻断点（连续性检查）
    # 如果前一步是高熵，当前步是低熵，则重置
    prev_was_high = True  # 初始假设前一步是高熵，这样第一个低熵步骤差分熵为0
    low_entropy_sequence = []  # 当前连续的低熵序列
    
    for step in sorted_steps:
        step_index = step.get('step_index')
        entropy = step.get('mean_entropy', 0)
        
        if entropy > threshold:
            # 高熵步骤
            if low_entropy_sequence:
                # 结束当前低熵序列，计算差分熵
                for i, (idx, ent) in enumerate(low_entropy_sequence):
                    if i == 0:
                        diff_entropy = 0
                    else:
                        prev_entropies = [e for _, e in low_entropy_sequence[:i]]
                        prev_mean = np.mean(prev_entropies)
                        diff_entropy = abs(ent - prev_mean)
                    low_diff_entropies.append({
                        'step_index': idx,
                        'entropy': ent,
                        'diff_entropy': diff_entropy,
                        'type': 'low'
                    })
                low_entropy_sequence = []
            prev_was_high = True
        else:
            # 低熵步骤
            if prev_was_high:
                # 从高熵切换到低熵，重置序列
                if low_entropy_sequence:
                    # 先处理之前的序列
                    for i, (idx, ent) in enumerate(low_entropy_sequence):
                        if i == 0:
                            diff_entropy = 0
                        else:
                            prev_entropies = [e for _, e in low_entropy_sequence[:i]]
                            prev_mean = np.mean(prev_entropies)
                            diff_entropy = abs(ent - prev_mean)
                        low_diff_entropies.append({
                            'step_index': idx,
                            'entropy': ent,
                            'diff_entropy': diff_entropy,
                            'type': 'low'
                        })
                low_entropy_sequence = [(step_index, entropy)]
            else:
                # 继续低熵序列
                low_entropy_sequence.append((step_index, entropy))
            prev_was_high = False
    
    # 处理最后的低熵序列
    if low_entropy_sequence:
        for i, (idx, ent) in enumerate(low_entropy_sequence):
            if i == 0:
                diff_entropy = 0
            else:
                prev_entropies = [e for _, e in low_entropy_sequence[:i]]
                prev_mean = np.mean(prev_entropies)
                diff_entropy = abs(ent - prev_mean)
            low_diff_entropies.append({
                'step_index': idx,
                'entropy': ent,
                'diff_entropy': diff_entropy,
                'type': 'low'
            })
    
    # 计算高熵步骤和低熵步骤相对于阈值的平均相对差异
    # 高熵步骤: (entropy - threshold) / threshold
    # 低熵步骤: (threshold - entropy) / threshold
    if high_diff_entropies and threshold > 0:
        high_relative_to_threshold = np.mean([(x['entropy'] - threshold) / threshold for x in high_diff_entropies])
    else:
        high_relative_to_threshold = None
    
    if low_diff_entropies and threshold > 0:
        low_relative_to_threshold = np.mean([(threshold - x['entropy']) / threshold for x in low_diff_entropies])
    else:
        low_relative_to_threshold = None
    
    return {
        'high_entropy_steps': high_diff_entropies,
        'low_entropy_steps': low_diff_entropies,
        'n_high_entropy_steps': len(high_diff_entropies),
        'n_low_entropy_steps': len(low_diff_entropies),
        'avg_high_diff_entropy': np.mean([x['diff_entropy'] for x in high_diff_entropies]) if high_diff_entropies else None,
        'avg_low_diff_entropy': np.mean([x['diff_entropy'] for x in low_diff_entropies]) if low_diff_entropies else None,
        'threshold': threshold,  # 用于划分高低熵步骤的阈值（token加权平均熵）
        'n_steps': len(sorted_steps),
        'high_relative_to_threshold': high_relative_to_threshold,  # 高熵步骤相对于阈值的平均相对差异
        'low_relative_to_threshold': low_relative_to_threshold,    # 低熵步骤相对于阈值的平均相对差异
    }


def analyze_file(json_file):
    """分析单个JSON文件的差分熵
    
    采用回答级别平均：先对每个回答计算平均差分熵，再对所有回答取平均
    这样每个回答的权重相等，不会因为步骤数多而权重更大
    """
    print(f"\n{'='*80}")
    print(f"分析文件: {json_file.split('/')[-1]}")
    print(f"{'='*80}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    overall_pass1 = data.get('overall_pass@1', 0)
    total_correct = data.get('total_correct_responses', 0)
    total_responses = data.get('total_responses', 0)
    
    print(f"Overall Pass@1: {overall_pass1:.4f}")
    print(f"正确率: {total_correct}/{total_responses} = {total_correct/total_responses:.4f}")
    
    # 收集每个response的平均差分熵（回答级别）
    response_high_diff_means = []  # 每个回答的高熵差分熵平均值
    response_low_diff_means = []   # 每个回答的低熵差分熵平均值
    response_high_entropy_means = []  # 每个回答的高熵原始熵平均值
    response_low_entropy_means = []   # 每个回答的低熵原始熵平均值
    
    # 按正确/错误分类（回答级别）
    correct_response_high_diff_means = []
    correct_response_low_diff_means = []
    wrong_response_high_diff_means = []
    wrong_response_low_diff_means = []
    
    # 按正确/错误分类的原始熵（用于计算相对差分熵）
    correct_response_high_entropy_means = []
    correct_response_low_entropy_means = []
    wrong_response_high_entropy_means = []
    wrong_response_low_entropy_means = []
    
    # 同时保留步骤级别统计（用于显示总步骤数）
    total_high_steps = 0
    total_low_steps = 0
    
    # 用于调试：统计边界情况
    responses_with_no_high = 0
    responses_with_no_low = 0
    
    # 用于计算协方差的配对数据（只收集同时有高熵和低熵步骤的回答）
    paired_high_entropy_means = []
    paired_low_entropy_means = []
    paired_high_diff_means = []
    paired_low_diff_means = []
    paired_high_relative_diffs = []  # 高熵步骤的相对差分熵（差分熵/平均熵）
    paired_low_relative_diffs = []   # 低熵步骤的相对差分熵（差分熵/平均熵）
    
    # 高熵/低熵步骤相对于阈值的平均相对差异
    response_high_relative_to_threshold = []
    response_low_relative_to_threshold = []
    
    # 用于计算协方差的配对数据（按正确/错误分类）
    paired_high_rel_to_thresh = []  # 高熵步骤相对于阈值的相对差异
    paired_low_rel_to_thresh = []   # 低熵步骤相对于阈值的相对差异
    correct_high_rel_to_thresh = []
    correct_low_rel_to_thresh = []
    wrong_high_rel_to_thresh = []
    wrong_low_rel_to_thresh = []
    
    results = data.get('results', [])
    
    for result in results:
        responses = result.get('responses', [])
        
        for response in responses:
            is_correct = response.get('is_correct', False)
            entropy_analysis = response.get('entropy_analysis', {})
            
            if not entropy_analysis:
                continue
            
            steps = entropy_analysis.get('steps', [])
            if not steps:
                continue
            
            # 获取overall_mean_entropy（token加权的平均熵 = sum(所有token熵) / 总token数）
            overall_stats = entropy_analysis.get('overall_stats', {})
            overall_mean_entropy = overall_stats.get('overall_mean_entropy', None)
            
            # 计算差分熵（使用token加权的overall_mean_entropy作为阈值）
            diff_analysis = calculate_differential_entropy_for_response(steps, overall_mean_entropy)
            
            if diff_analysis:
                # 获取该回答的高熵和低熵步骤的差分熵
                high_steps = diff_analysis['high_entropy_steps']
                low_steps = diff_analysis['low_entropy_steps']
                
                total_high_steps += len(high_steps)
                total_low_steps += len(low_steps)
                
                # 统计边界情况
                if len(high_steps) == 0:
                    responses_with_no_high += 1
                if len(low_steps) == 0:
                    responses_with_no_low += 1
                
                # 计算该回答的平均差分熵
                # 注意：如果某类步骤为空，avg_xxx_diff_entropy 为 None
                high_diff_mean = diff_analysis['avg_high_diff_entropy']
                low_diff_mean = diff_analysis['avg_low_diff_entropy']
                
                # 只有当有高熵步骤时才记录高熵差分熵
                if high_diff_mean is not None:
                    high_entropy_mean = np.mean([s['entropy'] for s in high_steps])
                    response_high_diff_means.append(high_diff_mean)
                    response_high_entropy_means.append(high_entropy_mean)
                    
                    if is_correct:
                        correct_response_high_diff_means.append(high_diff_mean)
                        correct_response_high_entropy_means.append(high_entropy_mean)
                    else:
                        wrong_response_high_diff_means.append(high_diff_mean)
                        wrong_response_high_entropy_means.append(high_entropy_mean)
                
                # 只有当有低熵步骤时才记录低熵差分熵
                if low_diff_mean is not None:
                    low_entropy_mean = np.mean([s['entropy'] for s in low_steps])
                    response_low_diff_means.append(low_diff_mean)
                    response_low_entropy_means.append(low_entropy_mean)
                    
                    if is_correct:
                        correct_response_low_diff_means.append(low_diff_mean)
                        correct_response_low_entropy_means.append(low_entropy_mean)
                    else:
                        wrong_response_low_diff_means.append(low_diff_mean)
                        wrong_response_low_entropy_means.append(low_entropy_mean)
                
                # 如果该回答同时有高熵和低熵步骤，收集配对数据用于计算协方差
                if high_diff_mean is not None and low_diff_mean is not None:
                    high_entropy_mean = np.mean([s['entropy'] for s in high_steps])
                    low_entropy_mean = np.mean([s['entropy'] for s in low_steps])
                    paired_high_entropy_means.append(high_entropy_mean)
                    paired_low_entropy_means.append(low_entropy_mean)
                    paired_high_diff_means.append(high_diff_mean)
                    paired_low_diff_means.append(low_diff_mean)
                    
                    # 计算该回答的相对差分熵（差分熵/平均熵）
                    high_relative_diff = high_diff_mean / high_entropy_mean if high_entropy_mean > 0 else 0
                    low_relative_diff = low_diff_mean / low_entropy_mean if low_entropy_mean > 0 else 0
                    paired_high_relative_diffs.append(high_relative_diff)
                    paired_low_relative_diffs.append(low_relative_diff)
                
                # 收集高熵/低熵步骤相对于阈值的平均相对差异
                if diff_analysis.get('high_relative_to_threshold') is not None:
                    response_high_relative_to_threshold.append(diff_analysis['high_relative_to_threshold'])
                if diff_analysis.get('low_relative_to_threshold') is not None:
                    response_low_relative_to_threshold.append(diff_analysis['low_relative_to_threshold'])
                
                # 如果同时有高熵和低熵步骤的相对差异，收集配对数据
                if diff_analysis.get('high_relative_to_threshold') is not None and diff_analysis.get('low_relative_to_threshold') is not None:
                    h_rel = diff_analysis['high_relative_to_threshold']
                    l_rel = diff_analysis['low_relative_to_threshold']
                    paired_high_rel_to_thresh.append(h_rel)
                    paired_low_rel_to_thresh.append(l_rel)
                    
                    if is_correct:
                        correct_high_rel_to_thresh.append(h_rel)
                        correct_low_rel_to_thresh.append(l_rel)
                    else:
                        wrong_high_rel_to_thresh.append(h_rel)
                        wrong_low_rel_to_thresh.append(l_rel)
    
    # 打印边界情况调试信息
    print(f"\n📋 边界情况统计：")
    print(f"  - 无高熵步骤的回答数: {responses_with_no_high}")
    print(f"  - 无低熵步骤的回答数: {responses_with_no_low}")
    
    # 计算回答级别的统计指标
    avg_high_diff_entropy = np.mean(response_high_diff_means) if response_high_diff_means else 0
    avg_low_diff_entropy = np.mean(response_low_diff_means) if response_low_diff_means else 0
    std_high_diff_entropy = np.std(response_high_diff_means) if response_high_diff_means else 0
    std_low_diff_entropy = np.std(response_low_diff_means) if response_low_diff_means else 0
    
    # 原始熵值统计（回答级别）
    avg_high_entropy = np.mean(response_high_entropy_means) if response_high_entropy_means else 0
    avg_low_entropy = np.mean(response_low_entropy_means) if response_low_entropy_means else 0
    
    # 正确/错误分类统计（回答级别）
    avg_correct_high_diff = np.mean(correct_response_high_diff_means) if correct_response_high_diff_means else 0
    avg_correct_low_diff = np.mean(correct_response_low_diff_means) if correct_response_low_diff_means else 0
    avg_wrong_high_diff = np.mean(wrong_response_high_diff_means) if wrong_response_high_diff_means else 0
    avg_wrong_low_diff = np.mean(wrong_response_low_diff_means) if wrong_response_low_diff_means else 0
    
    # 正确/错误分类的原始熵平均值
    avg_correct_high_entropy = np.mean(correct_response_high_entropy_means) if correct_response_high_entropy_means else 0
    avg_correct_low_entropy = np.mean(correct_response_low_entropy_means) if correct_response_low_entropy_means else 0
    avg_wrong_high_entropy = np.mean(wrong_response_high_entropy_means) if wrong_response_high_entropy_means else 0
    avg_wrong_low_entropy = np.mean(wrong_response_low_entropy_means) if wrong_response_low_entropy_means else 0
    
    # 正确/错误分类的相对差分熵
    correct_high_relative_diff = avg_correct_high_diff / avg_correct_high_entropy if avg_correct_high_entropy > 0 else 0
    correct_low_relative_diff = avg_correct_low_diff / avg_correct_low_entropy if avg_correct_low_entropy > 0 else 0
    wrong_high_relative_diff = avg_wrong_high_diff / avg_wrong_high_entropy if avg_wrong_high_entropy > 0 else 0
    wrong_low_relative_diff = avg_wrong_low_diff / avg_wrong_low_entropy if avg_wrong_low_entropy > 0 else 0
    
    # 计算协方差
    # 高熵步骤平均熵 与 低熵步骤平均熵 的协方差
    if len(paired_high_entropy_means) > 1:
        cov_entropy = np.cov(paired_high_entropy_means, paired_low_entropy_means)[0, 1]
        corr_entropy = np.corrcoef(paired_high_entropy_means, paired_low_entropy_means)[0, 1]
    else:
        cov_entropy = 0
        corr_entropy = 0
    
    # 高熵步骤差分熵 与 低熵步骤差分熵 的协方差
    if len(paired_high_diff_means) > 1:
        cov_diff = np.cov(paired_high_diff_means, paired_low_diff_means)[0, 1]
        corr_diff = np.corrcoef(paired_high_diff_means, paired_low_diff_means)[0, 1]
    else:
        cov_diff = 0
        corr_diff = 0
    
    # 高熵步骤相对差分熵 与 低熵步骤相对差分熵 的协方差
    if len(paired_high_relative_diffs) > 1:
        cov_relative_diff = np.cov(paired_high_relative_diffs, paired_low_relative_diffs)[0, 1]
        corr_relative_diff = np.corrcoef(paired_high_relative_diffs, paired_low_relative_diffs)[0, 1]
    else:
        cov_relative_diff = 0
        corr_relative_diff = 0
    
    # 计算高熵/低熵步骤相对于阈值的平均相对差异
    avg_high_relative_to_threshold = np.mean(response_high_relative_to_threshold) if response_high_relative_to_threshold else 0
    avg_low_relative_to_threshold = np.mean(response_low_relative_to_threshold) if response_low_relative_to_threshold else 0
    
    # 计算高熵相对差异 vs 低熵相对差异的协方差和相关系数（全部样本）
    if len(paired_high_rel_to_thresh) > 1:
        cov_rel_to_thresh = np.cov(paired_high_rel_to_thresh, paired_low_rel_to_thresh)[0, 1]
        corr_rel_to_thresh = np.corrcoef(paired_high_rel_to_thresh, paired_low_rel_to_thresh)[0, 1]
    else:
        cov_rel_to_thresh = 0
        corr_rel_to_thresh = 0
    
    # 正确回答中的协方差和相关系数
    if len(correct_high_rel_to_thresh) > 1:
        cov_rel_to_thresh_correct = np.cov(correct_high_rel_to_thresh, correct_low_rel_to_thresh)[0, 1]
        corr_rel_to_thresh_correct = np.corrcoef(correct_high_rel_to_thresh, correct_low_rel_to_thresh)[0, 1]
    else:
        cov_rel_to_thresh_correct = 0
        corr_rel_to_thresh_correct = 0
    
    # 错误回答中的协方差和相关系数
    if len(wrong_high_rel_to_thresh) > 1:
        cov_rel_to_thresh_wrong = np.cov(wrong_high_rel_to_thresh, wrong_low_rel_to_thresh)[0, 1]
        corr_rel_to_thresh_wrong = np.corrcoef(wrong_high_rel_to_thresh, wrong_low_rel_to_thresh)[0, 1]
    else:
        cov_rel_to_thresh_wrong = 0
        corr_rel_to_thresh_wrong = 0
    
    print(f"\n{'='*60}")
    print(f"差分熵统计（回答级别平均）")
    print(f"阈值计算方式: overall_mean_entropy = sum(所有token熵) / 总token数")
    print(f"步骤熵计算: step.mean_entropy = sum(该步骤token熵) / 该步骤token数")
    print(f"划分方式: step.mean_entropy > threshold 为高熵，<= threshold 为低熵")
    print(f"{'='*60}")
    print(f"\n高熵步骤（entropy > threshold）：")
    print(f"  - 总步骤数: {total_high_steps}")
    print(f"  - 有高熵步骤的回答数: {len(response_high_diff_means)}")
    print(f"  - 原始熵平均值: {avg_high_entropy:.6f}")
    print(f"  - 差分熵平均值: {avg_high_diff_entropy:.6f}")
    print(f"  - 差分熵标准差: {std_high_diff_entropy:.6f}")
    
    print(f"\n低熵步骤（entropy <= threshold）：")
    print(f"  - 总步骤数: {total_low_steps}")
    print(f"  - 有低熵步骤的回答数: {len(response_low_diff_means)}")
    print(f"  - 原始熵平均值: {avg_low_entropy:.6f}")
    print(f"  - 差分熵平均值: {avg_low_diff_entropy:.6f}")
    print(f"  - 差分熵标准差: {std_low_diff_entropy:.6f}")
    
    print(f"\n差分熵比值:")
    if avg_low_diff_entropy > 0:
        ratio = avg_high_diff_entropy / avg_low_diff_entropy
        print(f"  - 高熵差分熵 / 低熵差分熵 = {ratio:.4f}")
    else:
        ratio = float('inf')
        print(f"  - 高熵差分熵 / 低熵差分熵 = ∞ (低熵差分熵为0)")
    
    # 计算相对差分熵（差分熵/平均熵）
    print(f"\n相对差分熵（差分熵/平均熵）:")
    if avg_high_entropy > 0:
        high_relative_diff = avg_high_diff_entropy / avg_high_entropy
        print(f"  - 高熵步骤: 差分熵/平均熵 = {avg_high_diff_entropy:.6f} / {avg_high_entropy:.6f} = {high_relative_diff:.4f}")
    else:
        high_relative_diff = 0
        print(f"  - 高熵步骤: 差分熵/平均熵 = N/A (平均熵为0)")
    
    if avg_low_entropy > 0:
        low_relative_diff = avg_low_diff_entropy / avg_low_entropy
        print(f"  - 低熵步骤: 差分熵/平均熵 = {avg_low_diff_entropy:.6f} / {avg_low_entropy:.6f} = {low_relative_diff:.4f}")
    else:
        low_relative_diff = 0
        print(f"  - 低熵步骤: 差分熵/平均熵 = N/A (平均熵为0)")
    
    print(f"\n协方差分析（同时有高熵和低熵步骤的回答，n={len(paired_high_entropy_means)}）：")
    print(f"  - 高熵平均熵 vs 低熵平均熵:")
    print(f"    协方差: {cov_entropy:.6f}")
    print(f"    相关系数: {corr_entropy:.4f}")
    print(f"  - 高熵差分熵 vs 低熵差分熵:")
    print(f"    协方差: {cov_diff:.6f}")
    print(f"    相关系数: {corr_diff:.4f}")
    print(f"  - 高熵相对差分熵 vs 低熵相对差分熵:")
    print(f"    协方差: {cov_relative_diff:.6f}")
    print(f"    相关系数: {corr_relative_diff:.4f}")
    
    print(f"\n相对于阈值的平均相对差异：")
    print(f"  - 高熵步骤: mean((entropy - threshold) / threshold) = {avg_high_relative_to_threshold:.4f}")
    print(f"  - 低熵步骤: mean((threshold - entropy) / threshold) = {avg_low_relative_to_threshold:.4f}")
    print(f"  - 高熵/低熵 比值 = {avg_high_relative_to_threshold / avg_low_relative_to_threshold:.4f}" if avg_low_relative_to_threshold > 0 else "  - 高熵/低熵 比值 = N/A")
    
    print(f"\n高熵相对差异 vs 低熵相对差异 的协方差和相关系数：")
    print(f"  全部样本 (n={len(paired_high_rel_to_thresh)}):")
    print(f"    协方差: {cov_rel_to_thresh:.6f}")
    print(f"    相关系数: {corr_rel_to_thresh:.4f}")
    print(f"  ✅ 正确回答 (n={len(correct_high_rel_to_thresh)}):")
    print(f"    协方差: {cov_rel_to_thresh_correct:.6f}")
    print(f"    相关系数: {corr_rel_to_thresh_correct:.4f}")
    print(f"  ❌ 错误回答 (n={len(wrong_high_rel_to_thresh)}):")
    print(f"    协方差: {cov_rel_to_thresh_wrong:.6f}")
    print(f"    相关系数: {corr_rel_to_thresh_wrong:.4f}")
    
    print(f"\n{'='*60}")
    print(f"按正确/错误分类的差分熵（回答级别）")
    print(f"{'='*60}")
    print(f"\n✅ 正确回答：")
    print(f"  - 高熵步骤差分熵: {avg_correct_high_diff:.6f} (n_responses={len(correct_response_high_diff_means)})")
    print(f"  - 高熵步骤平均熵: {avg_correct_high_entropy:.6f}")
    print(f"  - 高熵步骤相对差分熵: {correct_high_relative_diff:.4f}")
    print(f"  - 低熵步骤差分熵: {avg_correct_low_diff:.6f} (n_responses={len(correct_response_low_diff_means)})")
    print(f"  - 低熵步骤平均熵: {avg_correct_low_entropy:.6f}")
    print(f"  - 低熵步骤相对差分熵: {correct_low_relative_diff:.4f}")
    
    print(f"\n❌ 错误回答：")
    print(f"  - 高熵步骤差分熵: {avg_wrong_high_diff:.6f} (n_responses={len(wrong_response_high_diff_means)})")
    print(f"  - 高熵步骤平均熵: {avg_wrong_high_entropy:.6f}")
    print(f"  - 高熵步骤相对差分熵: {wrong_high_relative_diff:.4f}")
    print(f"  - 低熵步骤差分熵: {avg_wrong_low_diff:.6f} (n_responses={len(wrong_response_low_diff_means)})")
    print(f"  - 低熵步骤平均熵: {avg_wrong_low_entropy:.6f}")
    print(f"  - 低熵步骤相对差分熵: {wrong_low_relative_diff:.4f}")
    
    return {
        'file': json_file,
        'overall_pass1': overall_pass1,
        'total_correct': total_correct,
        'total_responses': total_responses,
        'accuracy': total_correct / total_responses if total_responses > 0 else 0,
        # 全局统计（回答级别）
        'avg_high_diff_entropy': avg_high_diff_entropy,
        'avg_low_diff_entropy': avg_low_diff_entropy,
        'std_high_diff_entropy': std_high_diff_entropy,
        'std_low_diff_entropy': std_low_diff_entropy,
        'avg_high_entropy': avg_high_entropy,
        'avg_low_entropy': avg_low_entropy,
        'n_high_steps': total_high_steps,
        'n_low_steps': total_low_steps,
        'n_responses_with_high': len(response_high_diff_means),
        'n_responses_with_low': len(response_low_diff_means),
        'diff_entropy_ratio': ratio,
        # 相对差分熵（差分熵/平均熵）
        'high_relative_diff': high_relative_diff,
        'low_relative_diff': low_relative_diff,
        # 按正确/错误分类（回答级别）
        'avg_correct_high_diff': avg_correct_high_diff,
        'avg_correct_low_diff': avg_correct_low_diff,
        'avg_wrong_high_diff': avg_wrong_high_diff,
        'avg_wrong_low_diff': avg_wrong_low_diff,
        'avg_correct_high_entropy': avg_correct_high_entropy,
        'avg_correct_low_entropy': avg_correct_low_entropy,
        'avg_wrong_high_entropy': avg_wrong_high_entropy,
        'avg_wrong_low_entropy': avg_wrong_low_entropy,
        'correct_high_relative_diff': correct_high_relative_diff,
        'correct_low_relative_diff': correct_low_relative_diff,
        'wrong_high_relative_diff': wrong_high_relative_diff,
        'wrong_low_relative_diff': wrong_low_relative_diff,
        'n_correct_responses_high': len(correct_response_high_diff_means),
        'n_correct_responses_low': len(correct_response_low_diff_means),
        'n_wrong_responses_high': len(wrong_response_high_diff_means),
        'n_wrong_responses_low': len(wrong_response_low_diff_means),
        # 边界情况
        'responses_with_no_high': responses_with_no_high,
        'responses_with_no_low': responses_with_no_low,
        # 协方差分析
        'cov_entropy': cov_entropy,
        'corr_entropy': corr_entropy,
        'cov_diff': cov_diff,
        'corr_diff': corr_diff,
        'cov_relative_diff': cov_relative_diff,
        'corr_relative_diff': corr_relative_diff,
        'n_paired_responses': len(paired_high_entropy_means),
        # 相对于阈值的平均相对差异
        'avg_high_relative_to_threshold': avg_high_relative_to_threshold,
        'avg_low_relative_to_threshold': avg_low_relative_to_threshold,
        # 高熵相对差异 vs 低熵相对差异的协方差和相关系数
        'cov_rel_to_thresh': cov_rel_to_thresh,
        'corr_rel_to_thresh': corr_rel_to_thresh,
        'cov_rel_to_thresh_correct': cov_rel_to_thresh_correct,
        'corr_rel_to_thresh_correct': corr_rel_to_thresh_correct,
        'cov_rel_to_thresh_wrong': cov_rel_to_thresh_wrong,
        'corr_rel_to_thresh_wrong': corr_rel_to_thresh_wrong,
        'n_paired_rel_to_thresh': len(paired_high_rel_to_thresh),
        'n_correct_rel_to_thresh': len(correct_high_rel_to_thresh),
        'n_wrong_rel_to_thresh': len(wrong_high_rel_to_thresh),
    }


def compare_two_files(file1, file2, labels=None):
    """对比两个文件的差分熵"""
    
    if labels is None:
        labels = ["文件1", "文件2"]
    
    result1 = analyze_file(file1)
    result2 = analyze_file(file2)
    
    print(f"\n{'='*80}")
    print(f"对比结果")
    print(f"{'='*80}")
    
    # 表格对比
    print(f"\n{'指标':<35} {labels[0]:<20} {labels[1]:<20} {'差值':<15}")
    print("-" * 90)
    
    # 基本指标
    print(f"{'Pass@1':<35} {result1['overall_pass1']:<20.4f} {result2['overall_pass1']:<20.4f} {result2['overall_pass1']-result1['overall_pass1']:+.4f}")
    print(f"{'准确率':<35} {result1['accuracy']:<20.4f} {result2['accuracy']:<20.4f} {result2['accuracy']-result1['accuracy']:+.4f}")
    
    print("-" * 90)
    print("差分熵指标：")
    
    # 差分熵指标
    print(f"{'高熵步骤差分熵平均值':<35} {result1['avg_high_diff_entropy']:<20.6f} {result2['avg_high_diff_entropy']:<20.6f} {result2['avg_high_diff_entropy']-result1['avg_high_diff_entropy']:+.6f}")
    print(f"{'低熵步骤差分熵平均值':<35} {result1['avg_low_diff_entropy']:<20.6f} {result2['avg_low_diff_entropy']:<20.6f} {result2['avg_low_diff_entropy']-result1['avg_low_diff_entropy']:+.6f}")
    print(f"{'差分熵比值(高/低)':<35} {result1['diff_entropy_ratio']:<20.4f} {result2['diff_entropy_ratio']:<20.4f} {result2['diff_entropy_ratio']-result1['diff_entropy_ratio']:+.4f}")
    
    print("-" * 90)
    print("原始熵值：")
    print(f"{'高熵步骤原始熵平均值':<35} {result1['avg_high_entropy']:<20.6f} {result2['avg_high_entropy']:<20.6f} {result2['avg_high_entropy']-result1['avg_high_entropy']:+.6f}")
    print(f"{'低熵步骤原始熵平均值':<35} {result1['avg_low_entropy']:<20.6f} {result2['avg_low_entropy']:<20.6f} {result2['avg_low_entropy']-result1['avg_low_entropy']:+.6f}")
    
    print("-" * 90)
    print("相对差分熵（差分熵/平均熵）：")
    print(f"{'高熵步骤相对差分熵':<35} {result1['high_relative_diff']:<20.4f} {result2['high_relative_diff']:<20.4f} {result2['high_relative_diff']-result1['high_relative_diff']:+.4f}")
    print(f"{'低熵步骤相对差分熵':<35} {result1['low_relative_diff']:<20.4f} {result2['low_relative_diff']:<20.4f} {result2['low_relative_diff']-result1['low_relative_diff']:+.4f}")
    
    print("-" * 90)
    print("正确回答的差分熵：")
    print(f"{'正确-高熵步骤差分熵':<35} {result1['avg_correct_high_diff']:<20.6f} {result2['avg_correct_high_diff']:<20.6f} {result2['avg_correct_high_diff']-result1['avg_correct_high_diff']:+.6f}")
    print(f"{'正确-高熵步骤平均熵':<35} {result1['avg_correct_high_entropy']:<20.6f} {result2['avg_correct_high_entropy']:<20.6f} {result2['avg_correct_high_entropy']-result1['avg_correct_high_entropy']:+.6f}")
    print(f"{'正确-高熵步骤相对差分熵':<35} {result1['correct_high_relative_diff']:<20.4f} {result2['correct_high_relative_diff']:<20.4f} {result2['correct_high_relative_diff']-result1['correct_high_relative_diff']:+.4f}")
    print(f"{'正确-低熵步骤差分熵':<35} {result1['avg_correct_low_diff']:<20.6f} {result2['avg_correct_low_diff']:<20.6f} {result2['avg_correct_low_diff']-result1['avg_correct_low_diff']:+.6f}")
    print(f"{'正确-低熵步骤平均熵':<35} {result1['avg_correct_low_entropy']:<20.6f} {result2['avg_correct_low_entropy']:<20.6f} {result2['avg_correct_low_entropy']-result1['avg_correct_low_entropy']:+.6f}")
    print(f"{'正确-低熵步骤相对差分熵':<35} {result1['correct_low_relative_diff']:<20.4f} {result2['correct_low_relative_diff']:<20.4f} {result2['correct_low_relative_diff']-result1['correct_low_relative_diff']:+.4f}")
    
    print("-" * 90)
    print("错误回答的差分熵：")
    print(f"{'错误-高熵步骤差分熵':<35} {result1['avg_wrong_high_diff']:<20.6f} {result2['avg_wrong_high_diff']:<20.6f} {result2['avg_wrong_high_diff']-result1['avg_wrong_high_diff']:+.6f}")
    print(f"{'错误-高熵步骤平均熵':<35} {result1['avg_wrong_high_entropy']:<20.6f} {result2['avg_wrong_high_entropy']:<20.6f} {result2['avg_wrong_high_entropy']-result1['avg_wrong_high_entropy']:+.6f}")
    print(f"{'错误-高熵步骤相对差分熵':<35} {result1['wrong_high_relative_diff']:<20.4f} {result2['wrong_high_relative_diff']:<20.4f} {result2['wrong_high_relative_diff']-result1['wrong_high_relative_diff']:+.4f}")
    print(f"{'错误-低熵步骤差分熵':<35} {result1['avg_wrong_low_diff']:<20.6f} {result2['avg_wrong_low_diff']:<20.6f} {result2['avg_wrong_low_diff']-result1['avg_wrong_low_diff']:+.6f}")
    print(f"{'错误-低熵步骤平均熵':<35} {result1['avg_wrong_low_entropy']:<20.6f} {result2['avg_wrong_low_entropy']:<20.6f} {result2['avg_wrong_low_entropy']-result1['avg_wrong_low_entropy']:+.6f}")
    print(f"{'错误-低熵步骤相对差分熵':<35} {result1['wrong_low_relative_diff']:<20.4f} {result2['wrong_low_relative_diff']:<20.4f} {result2['wrong_low_relative_diff']-result1['wrong_low_relative_diff']:+.4f}")
    
    print("-" * 90)
    print(f"协方差分析（高熵 vs 低熵）：")
    print(f"{'高熵平均熵vs低熵平均熵-协方差':<35} {result1['cov_entropy']:<20.6f} {result2['cov_entropy']:<20.6f} {result2['cov_entropy']-result1['cov_entropy']:+.6f}")
    print(f"{'高熵平均熵vs低熵平均熵-相关系数':<35} {result1['corr_entropy']:<20.4f} {result2['corr_entropy']:<20.4f} {result2['corr_entropy']-result1['corr_entropy']:+.4f}")
    print(f"{'高熵差分熵vs低熵差分熵-协方差':<35} {result1['cov_diff']:<20.6f} {result2['cov_diff']:<20.6f} {result2['cov_diff']-result1['cov_diff']:+.6f}")
    print(f"{'高熵差分熵vs低熵差分熵-相关系数':<35} {result1['corr_diff']:<20.4f} {result2['corr_diff']:<20.4f} {result2['corr_diff']-result1['corr_diff']:+.4f}")
    print(f"{'高熵相对差分熵vs低熵相对差分熵-协方差':<35} {result1['cov_relative_diff']:<20.6f} {result2['cov_relative_diff']:<20.6f} {result2['cov_relative_diff']-result1['cov_relative_diff']:+.6f}")
    print(f"{'高熵相对差分熵vs低熵相对差分熵-相关系数':<35} {result1['corr_relative_diff']:<20.4f} {result2['corr_relative_diff']:<20.4f} {result2['corr_relative_diff']-result1['corr_relative_diff']:+.4f}")
    
    print("-" * 90)
    print("相对于阈值的平均相对差异：")
    print(f"{'高熵步骤(entropy-threshold)/threshold':<35} {result1['avg_high_relative_to_threshold']:<20.4f} {result2['avg_high_relative_to_threshold']:<20.4f} {result2['avg_high_relative_to_threshold']-result1['avg_high_relative_to_threshold']:+.4f}")
    print(f"{'低熵步骤(threshold-entropy)/threshold':<35} {result1['avg_low_relative_to_threshold']:<20.4f} {result2['avg_low_relative_to_threshold']:<20.4f} {result2['avg_low_relative_to_threshold']-result1['avg_low_relative_to_threshold']:+.4f}")
    ratio1 = result1['avg_high_relative_to_threshold'] / result1['avg_low_relative_to_threshold'] if result1['avg_low_relative_to_threshold'] > 0 else 0
    ratio2 = result2['avg_high_relative_to_threshold'] / result2['avg_low_relative_to_threshold'] if result2['avg_low_relative_to_threshold'] > 0 else 0
    print(f"{'高熵/低熵 比值':<35} {ratio1:<20.4f} {ratio2:<20.4f} {ratio2-ratio1:+.4f}")
    
    print("-" * 90)
    print("高熵相对差异 vs 低熵相对差异 协方差和相关系数：")
    print(f"{'全部样本-协方差':<35} {result1['cov_rel_to_thresh']:<20.6f} {result2['cov_rel_to_thresh']:<20.6f} {result2['cov_rel_to_thresh']-result1['cov_rel_to_thresh']:+.6f}")
    print(f"{'全部样本-相关系数':<35} {result1['corr_rel_to_thresh']:<20.4f} {result2['corr_rel_to_thresh']:<20.4f} {result2['corr_rel_to_thresh']-result1['corr_rel_to_thresh']:+.4f}")
    print(f"{'正确回答-协方差':<35} {result1['cov_rel_to_thresh_correct']:<20.6f} {result2['cov_rel_to_thresh_correct']:<20.6f} {result2['cov_rel_to_thresh_correct']-result1['cov_rel_to_thresh_correct']:+.6f}")
    print(f"{'正确回答-相关系数':<35} {result1['corr_rel_to_thresh_correct']:<20.4f} {result2['corr_rel_to_thresh_correct']:<20.4f} {result2['corr_rel_to_thresh_correct']-result1['corr_rel_to_thresh_correct']:+.4f}")
    print(f"{'错误回答-协方差':<35} {result1['cov_rel_to_thresh_wrong']:<20.6f} {result2['cov_rel_to_thresh_wrong']:<20.6f} {result2['cov_rel_to_thresh_wrong']-result1['cov_rel_to_thresh_wrong']:+.6f}")
    print(f"{'错误回答-相关系数':<35} {result1['corr_rel_to_thresh_wrong']:<20.4f} {result2['corr_rel_to_thresh_wrong']:<20.4f} {result2['corr_rel_to_thresh_wrong']-result1['corr_rel_to_thresh_wrong']:+.4f}")
    
    # 结论
    print(f"\n{'='*80}")
    print(f"🎯 结论分析")
    print(f"{'='*80}")
    
    print(f"\n📊 差分熵变化趋势：")
    
    # 分析高熵步骤差分熵变化
    high_diff_change = result2['avg_high_diff_entropy'] - result1['avg_high_diff_entropy']
    if result1['avg_high_diff_entropy'] > 0:
        if high_diff_change > 0:
            print(f"  • 高熵步骤差分熵增加了 {high_diff_change:.6f} ({high_diff_change/result1['avg_high_diff_entropy']*100:.2f}%)")
            print(f"    → 说明模型在高熵步骤之间的熵变化更剧烈，探索性更强")
        else:
            print(f"  • 高熵步骤差分熵减少了 {-high_diff_change:.6f} ({-high_diff_change/result1['avg_high_diff_entropy']*100:.2f}%)")
            print(f"    → 说明模型在高熵步骤之间的熵变化更平稳")
    
    # 分析低熵步骤差分熵变化
    low_diff_change = result2['avg_low_diff_entropy'] - result1['avg_low_diff_entropy']
    if result1['avg_low_diff_entropy'] > 0:
        if low_diff_change > 0:
            print(f"  • 低熵步骤差分熵增加了 {low_diff_change:.6f} ({low_diff_change/result1['avg_low_diff_entropy']*100:.2f}%)")
            print(f"    → 说明模型在低熵步骤之间仍有较大波动")
        else:
            print(f"  • 低熵步骤差分熵减少了 {-low_diff_change:.6f} ({-low_diff_change/result1['avg_low_diff_entropy']*100:.2f}%)")
            print(f"    → 说明模型在低熵步骤之间更加稳定，收敛性更好")
    
    # 分析比值变化
    ratio_change = result2['diff_entropy_ratio'] - result1['diff_entropy_ratio']
    print(f"\n📈 差分熵比值变化（高/低）：")
    print(f"  • 从 {result1['diff_entropy_ratio']:.4f} 变为 {result2['diff_entropy_ratio']:.4f}，变化 {ratio_change:+.4f}")
    
    if ratio_change > 0:
        print(f"    → 比值增大，说明高熵步骤的变化相对更剧烈，而低熵步骤更稳定")
        print(f"    → 这可能意味着模型学会了'该探索时大胆探索，该收敛时稳定收敛'")
    else:
        print(f"    → 比值减小，说明高熵和低熵步骤的差分熵差距在缩小")
    
    # 准确率与差分熵的关系
    print(f"\n🔍 准确率与差分熵的关系：")
    pass1_change = result2['overall_pass1'] - result1['overall_pass1']
    if pass1_change > 0:
        print(f"  • Pass@1 从 {result1['overall_pass1']:.4f} 提升到 {result2['overall_pass1']:.4f} (+{pass1_change:.4f})")
    else:
        print(f"  • Pass@1 从 {result1['overall_pass1']:.4f} 下降到 {result2['overall_pass1']:.4f} ({pass1_change:.4f})")
    
    # 分析正确回答与错误回答的差异
    print(f"\n📋 正确回答 vs 错误回答的差分熵特征（回答级别平均）：")
    print(f"  {labels[0]}:")
    print(f"    正确回答 - 高熵差分熵: {result1['avg_correct_high_diff']:.6f} (n={result1['n_correct_responses_high']}), 低熵差分熵: {result1['avg_correct_low_diff']:.6f} (n={result1['n_correct_responses_low']})")
    print(f"    错误回答 - 高熵差分熵: {result1['avg_wrong_high_diff']:.6f} (n={result1['n_wrong_responses_high']}), 低熵差分熵: {result1['avg_wrong_low_diff']:.6f} (n={result1['n_wrong_responses_low']})")
    
    print(f"  {labels[1]}:")
    print(f"    正确回答 - 高熵差分熵: {result2['avg_correct_high_diff']:.6f} (n={result2['n_correct_responses_high']}), 低熵差分熵: {result2['avg_correct_low_diff']:.6f} (n={result2['n_correct_responses_low']})")
    print(f"    错误回答 - 高熵差分熵: {result2['avg_wrong_high_diff']:.6f} (n={result2['n_wrong_responses_high']}), 低熵差分熵: {result2['avg_wrong_low_diff']:.6f} (n={result2['n_wrong_responses_low']})")
    
    return result1, result2


if __name__ == "__main__":
    # 两个AIME评估结果
    file1 = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260204_041649.json"
    file2 = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260204_043201.json"
    
    # 对比分析
    compare_two_files(
        file1, file2,
        labels=["基础模型", "训练模型"]
    )
