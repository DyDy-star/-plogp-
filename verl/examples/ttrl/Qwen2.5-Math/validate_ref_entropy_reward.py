#!/usr/bin/env python3
"""
验证 参考模型熵效率奖励 R = mean(H_ref - H_θ) 的可行性

由于我们没有参考模型在当前模型生成文本上的熵数据，
使用以下代理分析：

1. -H_θ 作为奖励的 GRPO Gap（核心代理指标）
   如果组内 H_θ 越低 → 越正确，那么 H_ref - H_θ 会进一步放大信号
   
2. 不同熵聚合方式的比较（overall, last-3, min, std 等）

3. 与之前所有度量的对比

4. 理论估算：Checkpoint vs Base 模型的熵水平差异

5. 组内方差分析：-H_θ 的组内区分度是否足够
"""

import json
import numpy as np
from collections import defaultdict

N_SAMPLES_PER_PROMPT = 16  # 评估数据每个prompt 16个样本


def load_samples(filepath):
    """加载评估数据"""
    with open(filepath) as f:
        data = json.load(f)
    
    samples = []
    for result in data['results']:
        problem_id = result.get('problem_id', result.get('id', ''))
        
        for resp in result['responses']:
            ea = resp['entropy_analysis']
            steps = ea.get('steps', [])
            transitions = ea.get('step_transitions', [])
            overall = ea.get('overall_stats', {})
            
            if not steps:
                continue
            
            # === 基础熵指标 ===
            step_means = [s['mean_entropy'] for s in steps if 'mean_entropy' in s]
            
            # 整体平均熵
            overall_mean = overall.get('overall_mean_entropy', np.mean(step_means) if step_means else 0)
            overall_std = overall.get('overall_std_entropy', 0)
            
            # Token 级熵 (从步骤内收集)
            all_token_entropies = []
            for s in steps:
                if 'token_entropies' in s:
                    all_token_entropies.extend(s['token_entropies'])
            
            if not step_means or len(step_means) < 3:
                continue
            
            # === 计算各种代理指标 ===
            metrics = {}
            
            # ===== 核心代理: -H_θ (越低越好 → 越正确) =====
            metrics['A_neg_mean_H'] = -overall_mean  # 整体平均
            metrics['B_neg_last3_H'] = -np.mean(step_means[-3:])  # 最后3步
            metrics['C_neg_first3_H'] = -np.mean(step_means[:3])  # 前3步
            metrics['D_neg_std_H'] = -overall_std  # 标准差
            metrics['E_neg_max_H'] = -max(step_means)  # 最大值
            metrics['F_neg_min_H'] = -min(step_means)  # 最小值
            
            # ===== 熵变动力学指标 =====
            velocities = [step_means[i+1] - step_means[i] for i in range(len(step_means)-1)]
            
            # FM 平滑度
            metrics['G_fm_smooth'] = -np.std(velocities)
            
            # Peak ratio
            abs_v = [abs(v) for v in velocities]
            mean_abs = np.mean(abs_v)
            max_abs = max(abs_v)
            metrics['H_peak_ratio'] = max_abs / mean_abs if mean_abs > 1e-10 else 0.0
            
            # 平均 JS
            if transitions:
                js_vals = [t['js_divergence'] for t in transitions if 'js_divergence' in t]
                metrics['I_mean_js'] = np.mean(js_vals) if js_vals else 0.0
            else:
                metrics['I_mean_js'] = 0.0
            
            # ===== 复合指标: -H_θ + 动力学 =====
            # 如果 -H_θ 有信号，与动力学指标组合可能更强
            metrics['J_negH_plus_smooth'] = metrics['A_neg_mean_H'] + metrics['G_fm_smooth']
            metrics['K_negH_plus_peak'] = metrics['A_neg_mean_H'] + metrics['H_peak_ratio']
            
            # ===== Token 级统计 =====
            if all_token_entropies:
                metrics['L_neg_token_mean'] = -np.mean(all_token_entropies)
                metrics['M_neg_token_median'] = -np.median(all_token_entropies)
                # 高熵token的比例 (>1.0)
                high_entropy_ratio = sum(1 for e in all_token_entropies if e > 1.0) / len(all_token_entropies)
                metrics['N_neg_high_ent_ratio'] = -high_entropy_ratio
                # 零熵token的比例 (确定性token)
                zero_entropy_ratio = sum(1 for e in all_token_entropies if e < 0.01) / len(all_token_entropies)
                metrics['O_certainty_ratio'] = zero_entropy_ratio
            else:
                metrics['L_neg_token_mean'] = metrics['A_neg_mean_H']
                metrics['M_neg_token_median'] = 0.0
                metrics['N_neg_high_ent_ratio'] = 0.0
                metrics['O_certainty_ratio'] = 0.0
            
            # ===== 其他分布级指标 =====
            # top1 概率均值
            top1_probs = [s.get('top1_prob', 0) for s in steps if 'top1_prob' in s]
            if top1_probs:
                metrics['P_mean_top1'] = np.mean(top1_probs)
            else:
                metrics['P_mean_top1'] = 0.0
            
            # 熵下降趋势 (线性回归斜率)
            if len(step_means) >= 3:
                x = np.arange(len(step_means))
                slope = np.polyfit(x, step_means, 1)[0]
                metrics['Q_entropy_slope'] = -slope  # 负斜率 = 熵下降 = 好
            else:
                metrics['Q_entropy_slope'] = 0.0
            
            # 最后一步熵
            metrics['R_neg_final_H'] = -step_means[-1]
            
            samples.append({
                'problem_id': problem_id,
                'is_correct': resp.get('is_correct', False),
                'metrics': metrics,
                'overall_mean_entropy': overall_mean,
                'n_tokens': overall.get('total_tokens', len(all_token_entropies)),
            })
    
    return samples


def compute_grpo_gap(samples, metric_name):
    """计算 GRPO Gap"""
    groups = defaultdict(list)
    for s in samples:
        groups[s['problem_id']].append(s)
    
    correct_advantages = []
    incorrect_advantages = []
    
    for pid, group in groups.items():
        if len(group) < 4:
            continue
        
        values = [s['metrics'][metric_name] for s in group]
        mean_v = np.mean(values)
        std_v = np.std(values)
        
        if std_v < 1e-10:
            continue
        
        for s, v in zip(group, values):
            adv = (v - mean_v) / std_v
            if s['is_correct']:
                correct_advantages.append(adv)
            else:
                incorrect_advantages.append(adv)
    
    if not correct_advantages or not incorrect_advantages:
        return 0.0, 0, 0
    
    gap = np.mean(correct_advantages) - np.mean(incorrect_advantages)
    return gap, len(correct_advantages), len(incorrect_advantages)


def compute_within_group_stats(samples, metric_name):
    """计算组内统计"""
    groups = defaultdict(list)
    for s in samples:
        groups[s['problem_id']].append(s)
    
    stds = []
    ranges = []
    for pid, group in groups.items():
        if len(group) < 4:
            continue
        values = [s['metrics'][metric_name] for s in group]
        stds.append(np.std(values))
        ranges.append(max(values) - min(values))
    
    return np.mean(stds), np.mean(ranges)


def main():
    files = {
        'Checkpoint (24.8%)': 'eval_results_aime_full_entropy/aime_eval_full_entropy_20260206_131034.json',
        'Base (2.7%)':        'eval_results_aime_full_entropy/aime_eval_full_entropy_20260206_131750.json',
    }
    
    all_metric_names = [
        'A_neg_mean_H', 'B_neg_last3_H', 'C_neg_first3_H', 'D_neg_std_H',
        'E_neg_max_H', 'F_neg_min_H',
        'G_fm_smooth', 'H_peak_ratio', 'I_mean_js',
        'J_negH_plus_smooth', 'K_negH_plus_peak',
        'L_neg_token_mean', 'M_neg_token_median', 'N_neg_high_ent_ratio',
        'O_certainty_ratio', 'P_mean_top1',
        'Q_entropy_slope', 'R_neg_final_H',
    ]
    
    descriptions = {
        'A_neg_mean_H':       '-H_θ 整体平均 (核心代理)',
        'B_neg_last3_H':      '-H_θ 最后3步',
        'C_neg_first3_H':     '-H_θ 前3步',
        'D_neg_std_H':        '-std(H_θ) 熵波动',
        'E_neg_max_H':        '-max(H_θ) 最大熵',
        'F_neg_min_H':        '-min(H_θ) 最小熵',
        'G_fm_smooth':        'FM平滑度 -std(v)',
        'H_peak_ratio':       '尖峰比 max|v|/mean|v|',
        'I_mean_js':          '平均JS散度',
        'J_negH_plus_smooth': '-H_θ + FM平滑度',
        'K_negH_plus_peak':   '-H_θ + 尖峰比',
        'L_neg_token_mean':   '-H_θ token级均值',
        'M_neg_token_median': '-H_θ token级中位数',
        'N_neg_high_ent_ratio': '-(高熵token比例)',
        'O_certainty_ratio':  '确定性token比例',
        'P_mean_top1':        'Top-1概率均值',
        'Q_entropy_slope':    '熵下降斜率',
        'R_neg_final_H':      '-H_θ 最后一步',
    }
    
    for model_name, filepath in files.items():
        print(f"\n{'='*90}")
        print(f"  模型: {model_name}")
        print(f"{'='*90}")
        
        samples = load_samples(filepath)
        n_correct = sum(1 for s in samples if s['is_correct'])
        n_total = len(samples)
        print(f"  样本数: {n_total}, 正确: {n_correct}, 准确率: {n_correct/n_total:.3f}")
        
        # === 基础统计 ===
        correct_entropy = [s['overall_mean_entropy'] for s in samples if s['is_correct']]
        incorrect_entropy = [s['overall_mean_entropy'] for s in samples if not s['is_correct']]
        
        print(f"\n  === 基础统计 ===")
        print(f"  正确样本 平均H_θ: {np.mean(correct_entropy):.6f} ± {np.std(correct_entropy):.6f}")
        print(f"  错误样本 平均H_θ: {np.mean(incorrect_entropy):.6f} ± {np.std(incorrect_entropy):.6f}")
        print(f"  差异 (正确-错误): {np.mean(correct_entropy) - np.mean(incorrect_entropy):+.6f}")
        print(f"  → 正确样本熵{'更低' if np.mean(correct_entropy) < np.mean(incorrect_entropy) else '更高'} ← {'✅ 符合假设' if np.mean(correct_entropy) < np.mean(incorrect_entropy) else '❌ 不符合假设'}")
        
        # === Token数量与正确性 ===
        correct_tokens = [s['n_tokens'] for s in samples if s['is_correct']]
        incorrect_tokens = [s['n_tokens'] for s in samples if not s['is_correct']]
        print(f"\n  正确样本 平均token数: {np.mean(correct_tokens):.0f}")
        print(f"  错误样本 平均token数: {np.mean(incorrect_tokens):.0f}")
        
        # === GRPO Gap 对比 ===
        print(f"\n  {'度量':<35} {'GRPO Gap':>10} {'组内Std':>10} {'组内Range':>10}")
        print(f"  {'-'*75}")
        
        results = []
        for mn in all_metric_names:
            gap, nc, ni = compute_grpo_gap(samples, mn)
            within_std, within_range = compute_within_group_stats(samples, mn)
            desc = descriptions.get(mn, mn)
            
            # 标记
            if mn == 'A_neg_mean_H':
                marker = '★'
            elif gap > 0.1:
                marker = '↑'
            elif gap < -0.1:
                marker = '↓'
            else:
                marker = ' '
            
            print(f"  {marker}{desc:<34} {gap:>+10.4f} {within_std:>10.6f} {within_range:>10.4f}")
            results.append((mn, desc, gap, within_std, within_range))
        
        # === 排名 ===
        print(f"\n  === GRPO Gap 排名 (Top 10) ===")
        ranked = sorted(results, key=lambda x: x[2], reverse=True)
        for i, (mn, desc, gap, std, rng) in enumerate(ranked[:10]):
            arrow = '★' if mn == 'A_neg_mean_H' else ' '
            print(f"    {i+1:>2}. {arrow}{desc}: gap={gap:+.4f}, std={std:.6f}, range={rng:.4f}")
        
        # === 组内分析: -H_θ 的差异能力 ===
        print(f"\n  === 组内分析: -H_θ 在每个题目上的表现 ===")
        groups = defaultdict(list)
        for s in samples:
            groups[s['problem_id']].append(s)
        
        n_correct_signal = 0
        n_wrong_signal = 0
        n_mixed = 0
        n_no_signal = 0  # 组内全对或全错
        
        for pid, group in groups.items():
            correct_vals = [s['metrics']['A_neg_mean_H'] for s in group if s['is_correct']]
            incorrect_vals = [s['metrics']['A_neg_mean_H'] for s in group if not s['is_correct']]
            
            if not correct_vals or not incorrect_vals:
                n_no_signal += 1
                continue
            
            if np.mean(correct_vals) > np.mean(incorrect_vals):
                n_correct_signal += 1
            else:
                n_wrong_signal += 1
        
        total_mixed = n_correct_signal + n_wrong_signal
        print(f"  有混合组（含正确+错误）: {total_mixed}/{len(groups)}")
        print(f"  正确信号（正确样本-H_θ更高）: {n_correct_signal}/{total_mixed} = {n_correct_signal/max(total_mixed,1):.1%}")
        print(f"  错误信号（正确样本-H_θ更低）: {n_wrong_signal}/{total_mixed} = {n_wrong_signal/max(total_mixed,1):.1%}")
    
    # === 跨模型对比 ===
    print(f"\n{'='*90}")
    print(f"  跨模型对比: Checkpoint vs Base 整体熵水平")
    print(f"{'='*90}")
    
    for model_name, filepath in files.items():
        samples = load_samples(filepath)
        means = [s['overall_mean_entropy'] for s in samples]
        print(f"  {model_name}: mean(H_θ) = {np.mean(means):.6f} ± {np.std(means):.6f}")
    
    print(f"\n  理论分析:")
    print(f"  如果 Base 模型作为参考, 在 Checkpoint 生成的文本上:")
    print(f"  - H_ref(correct_text) 可能较高 (Base 不理解正确推理的精确逻辑)")
    print(f"  - H_ref(incorrect_text) 可能较低 (错误文本更像 Base 的分布)")
    print(f"  - H_θ(correct_text) 较低 (Checkpoint 对正确推理很自信)")
    print(f"  - H_θ(incorrect_text) 较高 (Checkpoint 对错误推理不确定)")
    print(f"  → H_ref - H_θ 对正确样本: 高(H_ref) - 低(H_θ) = 大正值 ✅")
    print(f"  → H_ref - H_θ 对错误样本: 低(H_ref) - 高(H_θ) = 小或负值 ✅")
    print(f"  → GRPO gap 应该比纯 -H_θ 更大 (H_ref 提供额外区分)")
    
    # === 关键结论 ===
    print(f"\n{'='*90}")
    print(f"  关键结论")
    print(f"{'='*90}")
    
    # 重新加载 Checkpoint 数据
    ckpt_samples = load_samples(files['Checkpoint (24.8%)'])
    gap_A, _, _ = compute_grpo_gap(ckpt_samples, 'A_neg_mean_H')
    gap_G, _, _ = compute_grpo_gap(ckpt_samples, 'G_fm_smooth')
    gap_H, _, _ = compute_grpo_gap(ckpt_samples, 'H_peak_ratio')
    gap_I, _, _ = compute_grpo_gap(ckpt_samples, 'I_mean_js')
    gap_P, _, _ = compute_grpo_gap(ckpt_samples, 'P_mean_top1')
    
    print(f"\n  Checkpoint 模型上的 GRPO Gap 对比:")
    print(f"  -H_θ (整体平均):   {gap_A:+.4f}  ← R = mean(H_ref - H_θ) 的下界")
    print(f"  Top-1 概率均值:     {gap_P:+.4f}")
    print(f"  FM 平滑度:          {gap_G:+.4f}  (已知会导致熵坍缩)")
    print(f"  尖峰比:             {gap_H:+.4f}  (当前实现)")
    print(f"  平均 JS:            {gap_I:+.4f}  (之前失败)")
    
    if gap_A > 0:
        print(f"\n  ✅ -H_θ 的 GRPO Gap 为正 ({gap_A:+.4f})")
        print(f"  → 组内正确样本确实熵更低")
        print(f"  → 参考模型熵效率 R = mean(H_ref - H_θ) 有望进一步放大信号")
        print(f"  → H_ref 提供额外的难度参照，使信号更强")
    else:
        print(f"\n  ❌ -H_θ 的 GRPO Gap 为负 ({gap_A:+.4f})")
        print(f"  → 组内正确样本熵反而更高")
        print(f"  → 参考模型熵效率 R = mean(H_ref - H_θ) 可能不可行")


if __name__ == '__main__':
    main()
