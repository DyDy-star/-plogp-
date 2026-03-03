#!/usr/bin/env python3
"""
验证 非对称KL熵效率 奖励函数

核心公式:
  R_inc = -mean(KL_forward(i→i+1) | ΔH > 0)   无损熵增
  R_dec = -mean(KL_reverse(i+1→i) | ΔH < 0)   无损熵减
  R = R_inc + R_dec

KL方向的信息论含义:
  - KL(P_i || P_{i+1}): 步骤i对步骤i+1的"惊讶度"
    → 熵增时越低 = 增加越"可预期" = 不是随机噪声
  - KL(P_{i+1} || P_i): 步骤i+1对步骤i的"遗忘度"  
    → 熵减时越低 = 压缩越"可逆" = 没丢重要信息

对比度量:
  A: R_inc + R_dec (非对称KL,主提案)
  B: 仅R_inc
  C: 仅R_dec  
  D: 交换方向 (KL_reverse用于熵增, KL_forward用于熵减)
  E: -mean(KL_forward) 全步骤不分方向
  F: -mean(KL_reverse) 全步骤不分方向
  G: fm_smoothness (对比基线)
  H: peak_ratio (对比基线)
  I: mean(JS) (对比基线)
"""

import json
import numpy as np
from collections import defaultdict

N_SAMPLES_PER_PROMPT = 32


def load_samples(filepath):
    """加载评估数据, 返回样本列表"""
    with open(filepath) as f:
        data = json.load(f)
    
    samples = []
    for result in data['results']:
        problem_id = result.get('problem_id', result.get('id', ''))
        ground_truth = result.get('ground_truth', '')
        
        for resp in result['responses']:
            ea = resp['entropy_analysis']
            transitions = ea.get('step_transitions', [])
            steps = ea.get('steps', [])
            
            if not transitions or len(transitions) < 2:
                continue
            
            # 提取各转换的 KL_forward, KL_reverse, entropy_delta
            kl_forwards = []
            kl_reverses = []
            js_divs = []
            entropy_deltas = []
            
            for t in transitions:
                kl_f = t.get('kl_forward', 0)
                kl_r = t.get('kl_reverse', 0)
                js = t.get('js_divergence', 0)
                ed = t.get('entropy_delta', 0)
                
                # 过滤异常值
                if kl_f > 100 or kl_r > 100:
                    continue
                    
                kl_forwards.append(kl_f)
                kl_reverses.append(kl_r)
                js_divs.append(js)
                entropy_deltas.append(ed)
            
            if len(kl_forwards) < 2:
                continue
            
            # 分方向提取
            inc_kl_fwd = [kl_forwards[i] for i in range(len(entropy_deltas)) if entropy_deltas[i] > 0]
            inc_kl_rev = [kl_reverses[i] for i in range(len(entropy_deltas)) if entropy_deltas[i] > 0]
            dec_kl_fwd = [kl_forwards[i] for i in range(len(entropy_deltas)) if entropy_deltas[i] < 0]
            dec_kl_rev = [kl_reverses[i] for i in range(len(entropy_deltas)) if entropy_deltas[i] < 0]
            
            # 步骤熵用于计算fm_smoothness和peak_ratio
            step_entropies = [s['mean_entropy'] for s in steps if 'mean_entropy' in s]
            velocities = [step_entropies[i+1] - step_entropies[i] for i in range(len(step_entropies)-1)]
            
            # ===== 计算各度量 =====
            metrics = {}
            
            # A: R_inc + R_dec (核心提案)
            r_inc = -np.mean(inc_kl_fwd) if inc_kl_fwd else 0.0
            r_dec = -np.mean(dec_kl_rev) if dec_kl_rev else 0.0
            metrics['A_asym_kl'] = r_inc + r_dec
            
            # B: 仅R_inc
            metrics['B_only_inc'] = r_inc
            
            # C: 仅R_dec
            metrics['C_only_dec'] = r_dec
            
            # D: 交换方向
            r_inc_swap = -np.mean(inc_kl_rev) if inc_kl_rev else 0.0
            r_dec_swap = -np.mean(dec_kl_fwd) if dec_kl_fwd else 0.0
            metrics['D_swapped'] = r_inc_swap + r_dec_swap
            
            # E: -mean(KL_forward) 全步骤
            metrics['E_all_kl_fwd'] = -np.mean(kl_forwards)
            
            # F: -mean(KL_reverse) 全步骤
            metrics['F_all_kl_rev'] = -np.mean(kl_reverses)
            
            # G: fm_smoothness
            if len(velocities) >= 2:
                metrics['G_fm_smooth'] = -np.std(velocities)
            else:
                metrics['G_fm_smooth'] = 0.0
            
            # H: peak_ratio (S2)
            if len(velocities) >= 2:
                abs_v = [abs(v) for v in velocities]
                mean_abs = np.mean(abs_v)
                max_abs = np.max(abs_v)
                metrics['H_peak_ratio'] = max_abs / mean_abs if mean_abs > 1e-10 else 0.0
            else:
                metrics['H_peak_ratio'] = 0.0
            
            # I: mean(JS) 
            metrics['I_mean_js'] = np.mean(js_divs) if js_divs else 0.0
            
            # J: 比值形式 R_inc / R_dec (效率比)
            if dec_kl_rev and inc_kl_fwd:
                mean_inc_fwd = np.mean(inc_kl_fwd) if inc_kl_fwd else 1e-10
                mean_dec_rev = np.mean(dec_kl_rev) if dec_kl_rev else 1e-10
                # 熵增时的前向KL / 熵减时的逆向KL
                metrics['J_ratio'] = -mean_inc_fwd / (mean_dec_rev + 1e-10)
            else:
                metrics['J_ratio'] = 0.0
            
            # K: 非对称差异 = KL_fwd - KL_rev 在各方向上
            if inc_kl_fwd and inc_kl_rev:
                inc_diff = np.mean(inc_kl_fwd) - np.mean(inc_kl_rev)
                metrics['K_inc_asym'] = -abs(inc_diff)  # 越对称越好?
            else:
                metrics['K_inc_asym'] = 0.0
                
            if dec_kl_fwd and dec_kl_rev:
                dec_diff = np.mean(dec_kl_fwd) - np.mean(dec_kl_rev)
                metrics['K_dec_asym'] = -abs(dec_diff)
            else:
                metrics['K_dec_asym'] = 0.0
            
            # L: 加权版 (步骤位置加权)
            if inc_kl_fwd:
                inc_indices = [i for i in range(len(entropy_deltas)) if entropy_deltas[i] > 0]
                # 后面步骤权重更高 (位置归一化)
                weights = [(idx + 1) / len(entropy_deltas) for idx in inc_indices]
                w_sum = sum(weights)
                weighted_inc = -sum(w * kl_forwards[i] for w, i in zip(weights, inc_indices)) / w_sum if w_sum > 0 else 0
            else:
                weighted_inc = 0
            if dec_kl_rev:
                dec_indices = [i for i in range(len(entropy_deltas)) if entropy_deltas[i] < 0]
                weights = [(idx + 1) / len(entropy_deltas) for idx in dec_indices]
                w_sum = sum(weights)
                weighted_dec = -sum(w * kl_reverses[i] for w, i in zip(weights, dec_indices)) / w_sum if w_sum > 0 else 0
            else:
                weighted_dec = 0
            metrics['L_weighted'] = weighted_inc + weighted_dec
            
            samples.append({
                'problem_id': problem_id,
                'is_correct': resp.get('is_correct', False),
                'metrics': metrics,
                'n_inc': len(inc_kl_fwd),
                'n_dec': len(dec_kl_rev),
                'r_inc': r_inc,
                'r_dec': r_dec,
            })
    
    return samples


def compute_grpo_gap(samples, metric_name, n_per_group=N_SAMPLES_PER_PROMPT):
    """
    计算GRPO Gap:
      对每个prompt组(32个样本), 计算组内advantage后,
      看正确样本的平均advantage是否>不正确样本的平均advantage
    """
    # 按problem_id分组
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
        return 0.0, 0, 0, 0.0, 0.0
    
    gap = np.mean(correct_advantages) - np.mean(incorrect_advantages)
    return gap, len(correct_advantages), len(incorrect_advantages), np.mean(correct_advantages), np.mean(incorrect_advantages)


def compute_within_group_stats(samples, metric_name):
    """计算组内标准差"""
    groups = defaultdict(list)
    for s in samples:
        groups[s['problem_id']].append(s)
    
    stds = []
    for pid, group in groups.items():
        if len(group) < 4:
            continue
        values = [s['metrics'][metric_name] for s in group]
        stds.append(np.std(values))
    
    return np.mean(stds) if stds else 0.0


def main():
    files = {
        'Checkpoint': 'eval_results_aime_full_entropy/aime_eval_full_entropy_20260206_131034.json',
        'Base': 'eval_results_aime_full_entropy/aime_eval_full_entropy_20260206_131750.json',
    }
    
    metric_names = [
        'A_asym_kl', 'B_only_inc', 'C_only_dec', 'D_swapped',
        'E_all_kl_fwd', 'F_all_kl_rev', 
        'G_fm_smooth', 'H_peak_ratio', 'I_mean_js',
        'J_ratio', 'L_weighted',
    ]
    
    metric_descriptions = {
        'A_asym_kl':    '非对称KL熵效率 (-KLfwd|ΔH>0 + -KLrev|ΔH<0)',
        'B_only_inc':   '仅无损熵增 (-KLfwd|ΔH>0)',
        'C_only_dec':   '仅无损熵减 (-KLrev|ΔH<0)',
        'D_swapped':    '交换方向 (-KLrev|ΔH>0 + -KLfwd|ΔH<0)',
        'E_all_kl_fwd': '全步骤KL_fwd (-mean KLfwd)',
        'F_all_kl_rev': '全步骤KL_rev (-mean KLrev)',
        'G_fm_smooth':  'FM平滑度 (-std(v))',
        'H_peak_ratio': '尖峰比 (max|v|/mean|v|)',
        'I_mean_js':    '平均JS散度',
        'J_ratio':      'KL比值 (-KLfwd_inc / KLrev_dec)',
        'L_weighted':   '位置加权非对称KL',
    }
    
    for model_name, filepath in files.items():
        print(f"\n{'='*80}")
        print(f" 模型: {model_name}")
        print(f"{'='*80}")
        
        samples = load_samples(filepath)
        n_correct = sum(1 for s in samples if s['is_correct'])
        n_total = len(samples)
        print(f"  样本数: {n_total}, 正确: {n_correct}, 准确率: {n_correct/n_total:.3f}")
        
        # 统计方向分布
        avg_inc = np.mean([s['n_inc'] for s in samples])
        avg_dec = np.mean([s['n_dec'] for s in samples])
        print(f"  平均熵增步骤数: {avg_inc:.1f}, 平均熵减步骤数: {avg_dec:.1f}")
        
        # 打印R_inc和R_dec的分布
        r_inc_correct = [s['r_inc'] for s in samples if s['is_correct']]
        r_inc_incorrect = [s['r_inc'] for s in samples if not s['is_correct']]
        r_dec_correct = [s['r_dec'] for s in samples if s['is_correct']]
        r_dec_incorrect = [s['r_dec'] for s in samples if not s['is_correct']]
        
        print(f"\n  分部件统计:")
        print(f"  R_inc (无损熵增): 正确={np.mean(r_inc_correct):.6f}, 不正确={np.mean(r_inc_incorrect):.6f}")
        print(f"  R_dec (无损熵减): 正确={np.mean(r_dec_correct):.6f}, 不正确={np.mean(r_dec_incorrect):.6f}")
        
        print(f"\n  {'度量':<45} {'GRPO Gap':>10} {'组内Std':>10} {'正确Adv':>10} {'错误Adv':>10}")
        print(f"  {'-'*85}")
        
        results = []
        for mn in metric_names:
            gap, nc, ni, ca, ia = compute_grpo_gap(samples, mn)
            within_std = compute_within_group_stats(samples, mn)
            
            desc = metric_descriptions.get(mn, mn)
            marker = '★' if mn == 'A_asym_kl' else ' '
            
            print(f"  {marker}{desc:<44} {gap:>+10.4f} {within_std:>10.6f} {ca:>+10.4f} {ia:>+10.4f}")
            results.append((mn, gap, within_std))
        
        # 排名
        print(f"\n  按 GRPO Gap 排名:")
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        for i, (mn, gap, std) in enumerate(ranked):
            desc = metric_descriptions.get(mn, mn)
            marker = ' ← 核心提案' if mn == 'A_asym_kl' else ''
            print(f"    {i+1}. {desc}: gap={gap:+.4f}, std={std:.6f}{marker}")
    
    # 额外分析: 正确vs错误样本的KL_forward和KL_reverse在不同方向上的分布
    print(f"\n{'='*80}")
    print(f" 深度分析: KL_forward vs KL_reverse 在熵增/熵减时的行为差异")
    print(f"{'='*80}")
    
    for model_name, filepath in files.items():
        samples = load_samples(filepath)
        print(f"\n  [{model_name}]")
        
        # 重新遍历获取原始数据
        with open(filepath) as f:
            data = json.load(f)
        
        inc_fwd_correct, inc_fwd_incorrect = [], []
        inc_rev_correct, inc_rev_incorrect = [], []
        dec_fwd_correct, dec_fwd_incorrect = [], []
        dec_rev_correct, dec_rev_incorrect = [], []
        
        for result in data['results']:
            for resp in result['responses']:
                ea = resp['entropy_analysis']
                transitions = ea.get('step_transitions', [])
                is_correct = resp.get('is_correct', False)
                
                for t in transitions:
                    kl_f = t.get('kl_forward', 0)
                    kl_r = t.get('kl_reverse', 0)
                    ed = t.get('entropy_delta', 0)
                    
                    if kl_f > 100 or kl_r > 100:
                        continue
                    
                    if ed > 0:  # 熵增
                        if is_correct:
                            inc_fwd_correct.append(kl_f)
                            inc_rev_correct.append(kl_r)
                        else:
                            inc_fwd_incorrect.append(kl_f)
                            inc_rev_incorrect.append(kl_r)
                    elif ed < 0:  # 熵减
                        if is_correct:
                            dec_fwd_correct.append(kl_f)
                            dec_rev_correct.append(kl_r)
                        else:
                            dec_fwd_incorrect.append(kl_f)
                            dec_rev_incorrect.append(kl_r)
        
        print(f"  熵增时 KL_forward:  正确={np.mean(inc_fwd_correct):.6f} (n={len(inc_fwd_correct)}), "
              f"不正确={np.mean(inc_fwd_incorrect):.6f} (n={len(inc_fwd_incorrect)})")
        print(f"  熵增时 KL_reverse:  正确={np.mean(inc_rev_correct):.6f}, 不正确={np.mean(inc_rev_incorrect):.6f}")
        print(f"  熵减时 KL_forward:  正确={np.mean(dec_fwd_correct):.6f} (n={len(dec_fwd_correct)}), "
              f"不正确={np.mean(dec_fwd_incorrect):.6f} (n={len(dec_fwd_incorrect)})")
        print(f"  熵减时 KL_reverse:  正确={np.mean(dec_rev_correct):.6f}, 不正确={np.mean(dec_rev_incorrect):.6f}")
        
        # 非对称比
        print(f"\n  非对称比 (KL_fwd / KL_rev):")
        print(f"  熵增时: 正确={np.mean(inc_fwd_correct)/np.mean(inc_rev_correct):.3f}, "
              f"不正确={np.mean(inc_fwd_incorrect)/np.mean(inc_rev_incorrect):.3f}")
        print(f"  熵减时: 正确={np.mean(dec_fwd_correct)/np.mean(dec_rev_correct):.3f}, "
              f"不正确={np.mean(dec_fwd_incorrect)/np.mean(dec_rev_incorrect):.3f}")


if __name__ == '__main__':
    main()
