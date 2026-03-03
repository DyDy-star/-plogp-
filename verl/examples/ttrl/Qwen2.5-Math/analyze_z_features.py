"""
深度分析: 正确/错误样本的 z-score 特征对比
z_t = ε_t / σ_pos (仅 ε>0 有非零 z, ε≤0 的 z=0)
"""
import json
import numpy as np
from collections import defaultdict

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_z_and_epsilon(H_arr):
    n = len(H_arr)
    if n < 10:
        return None, None, None, None
    
    rho = np.corrcoef(H_arr[:-1], H_arr[1:])[0, 1]
    if np.isnan(rho): rho = 0.0
    rho = max(0.05, min(0.95, rho))
    alpha = 1.0 - rho
    
    ema = np.zeros(n)
    epsilon = np.zeros(n)
    ema[0] = H_arr[0]
    for t in range(1, n):
        epsilon[t] = H_arr[t] - ema[t-1]
        ema[t] = alpha * H_arr[t] + (1-alpha) * ema[t-1]
    
    pos_eps = epsilon[epsilon > 0]
    if len(pos_eps) < 3:
        return None, None, None, None
    
    sigma_pos = np.std(pos_eps)
    if sigma_pos < 1e-6:
        return None, None, None, None
    
    z = np.where(epsilon > 0, epsilon / sigma_pos, 0.0)
    return z, epsilon, rho, sigma_pos

def analyze(data, label):
    print(f"\n{'='*90}")
    print(f"  {label}  (pass@1={data['overall_pass@1']:.4f})")
    print(f"{'='*90}")
    
    correct_stats = []
    wrong_stats = []
    
    for problem in data['results']:
        for resp in problem['responses']:
            is_correct = resp['is_correct']
            ea = resp.get('entropy_analysis')
            if not ea or 'steps' not in ea:
                continue
            
            all_tokens = []
            all_entropies = []
            for step in ea['steps']:
                toks = step.get('tokens', [])
                ents = step.get('token_entropies', [])
                if len(toks) == len(ents):
                    all_tokens.extend(toks)
                    all_entropies.extend(ents)
            
            if len(all_tokens) < 20:
                continue
            
            H = np.array(all_entropies)
            result = compute_z_and_epsilon(H)
            if result[0] is None:
                continue
            
            z, epsilon, rho, sigma_pos = result
            
            n_total = len(z)
            n_pos_eps = int(np.sum(epsilon > 0))
            n_neg_eps = int(np.sum(epsilon <= 0))
            
            z_at_pos = z[epsilon > 0]
            
            entry = {
                'n_tokens': n_total,
                'rho': rho,
                'sigma_pos': sigma_pos,
                'mean_H': float(H.mean()),
                'std_H': float(H.std()),
                # epsilon stats
                'n_pos_eps': n_pos_eps,
                'n_neg_eps': n_neg_eps,
                'ratio_pos_eps': n_pos_eps / n_total,
                'mean_pos_eps': float(epsilon[epsilon > 0].mean()) if n_pos_eps > 0 else 0,
                'mean_neg_eps': float(epsilon[epsilon <= 0].mean()) if n_neg_eps > 0 else 0,
                # z stats (only at ε>0 positions)
                'z_mean': float(z_at_pos.mean()),
                'z_std': float(z_at_pos.std()),
                'z_median': float(np.median(z_at_pos)),
                'z_max': float(z_at_pos.max()),
                # z threshold counts
                'n_z_gt1': int(np.sum(z_at_pos > 1)),
                'n_z_gt2': int(np.sum(z_at_pos > 2)),
                'n_z_gt3': int(np.sum(z_at_pos > 3)),
                'n_z_gt5': int(np.sum(z_at_pos > 5)),
                # ratios
                'ratio_z_gt1': float(np.sum(z_at_pos > 1)) / max(n_pos_eps, 1),
                'ratio_z_gt2': float(np.sum(z_at_pos > 2)) / max(n_pos_eps, 1),
                'ratio_z_gt3': float(np.sum(z_at_pos > 3)) / max(n_pos_eps, 1),
                # tanh stats
                'tanh_mean': float(np.tanh(z_at_pos).mean()),
                'tanh_sum': float(np.tanh(z_at_pos).sum()),
                'tanh_sum_per_token': float(np.tanh(z_at_pos).sum()) / n_total,
                # overall z (including zeros)
                'z_all_mean': float(z.mean()),
                'z_all_nonzero_ratio': float(np.sum(z > 0)) / n_total,
            }
            
            if is_correct:
                correct_stats.append(entry)
            else:
                wrong_stats.append(entry)
    
    nc = len(correct_stats)
    nw = len(wrong_stats)
    print(f"\n  正确样本: {nc}, 错误样本: {nw}")
    
    def agg(stats, key):
        vals = [s[key] for s in stats]
        return np.mean(vals), np.std(vals), np.median(vals)
    
    def compare(key, label_str, fmt=".4f"):
        cm, cs, cmed = agg(correct_stats, key)
        wm, ws, wmed = agg(wrong_stats, key)
        delta = cm - wm
        effect = delta / max(((cs + ws) / 2), 1e-8)
        print(f"  {label_str:<35} 正确: {cm:{fmt}}±{cs:{fmt}}  错误: {wm:{fmt}}±{ws:{fmt}}  "
              f"Δ={delta:+{fmt}}  d={effect:+.3f}")
    
    # ============================================================
    print(f"\n--- 1. 基础统计 ---")
    compare('n_tokens', 'token 数量', '.1f')
    compare('mean_H', '平均熵 H', '.4f')
    compare('std_H', '熵标准差', '.4f')
    compare('rho', '自相关 ρ', '.4f')
    compare('sigma_pos', 'σ_pos (正 ε 标准差)', '.4f')
    
    # ============================================================
    print(f"\n--- 2. ε (创新) 分布 ---")
    compare('ratio_pos_eps', 'ε>0 占比', '.4f')
    compare('mean_pos_eps', 'ε>0 均值', '.4f')
    compare('mean_neg_eps', 'ε≤0 均值', '.4f')
    
    # ============================================================
    print(f"\n--- 3. z-score 分布 (仅 ε>0 位置) ---")
    compare('z_mean', 'z 均值', '.4f')
    compare('z_std', 'z 标准差', '.4f')
    compare('z_median', 'z 中位数', '.4f')
    compare('z_max', 'z 最大值', '.2f')
    
    # ============================================================
    print(f"\n--- 4. z 阈值比例 (占 ε>0 token) ---")
    compare('ratio_z_gt1', 'z > 1 比例', '.4f')
    compare('ratio_z_gt2', 'z > 2 比例', '.4f')
    compare('ratio_z_gt3', 'z > 3 比例', '.4f')
    
    # ============================================================
    print(f"\n--- 5. tanh(z) 统计 ---")
    compare('tanh_mean', 'tanh(z) 均值 (at ε>0)', '.4f')
    compare('tanh_sum', 'tanh(z) 总和 (per response)', '.2f')
    compare('tanh_sum_per_token', 'tanh(z) 总和/总token', '.4f')
    
    # ============================================================
    print(f"\n--- 6. 整体 z 比例 (含零值) ---")
    compare('z_all_nonzero_ratio', 'z>0 占总 token 比例', '.4f')
    compare('z_all_mean', 'z 全局均值 (含 0)', '.4f')
    
    # ============================================================
    # Per-problem analysis
    print(f"\n--- 7. 按问题分组: 正确 vs 错误的特征差异 ---")
    problem_diffs = []
    for problem in data['results']:
        c_entries = []
        w_entries = []
        for resp in problem['responses']:
            is_correct = resp['is_correct']
            ea = resp.get('entropy_analysis')
            if not ea or 'steps' not in ea:
                continue
            all_ents = []
            for step in ea['steps']:
                ents = step.get('token_entropies', [])
                all_ents.extend(ents)
            if len(all_ents) < 20:
                continue
            H = np.array(all_ents)
            result = compute_z_and_epsilon(H)
            if result[0] is None:
                continue
            z, epsilon, _, sigma_pos = result
            z_at_pos = z[epsilon > 0]
            if len(z_at_pos) == 0:
                continue
            entry = {
                'ratio_pos': float(np.sum(epsilon > 0)) / len(z),
                'z_mean': float(z_at_pos.mean()),
                'tanh_mean': float(np.tanh(z_at_pos).mean()),
                'ratio_z_gt2': float(np.sum(z_at_pos > 2)) / max(len(z_at_pos), 1),
                'mean_H': float(H.mean()),
                'sigma_pos': sigma_pos,
            }
            if is_correct:
                c_entries.append(entry)
            else:
                w_entries.append(entry)
        
        if c_entries and w_entries:
            for key in ['ratio_pos', 'z_mean', 'tanh_mean', 'ratio_z_gt2', 'mean_H', 'sigma_pos']:
                c_val = np.mean([e[key] for e in c_entries])
                w_val = np.mean([e[key] for e in w_entries])
                problem_diffs.append({'key': key, 'c': c_val, 'w': w_val, 'diff': c_val - w_val})
    
    if problem_diffs:
        from collections import Counter
        keys = set(d['key'] for d in problem_diffs)
        for key in sorted(keys):
            diffs = [d['diff'] for d in problem_diffs if d['key'] == key]
            c_vals = [d['c'] for d in problem_diffs if d['key'] == key]
            w_vals = [d['w'] for d in problem_diffs if d['key'] == key]
            n_c_wins = sum(1 for d in diffs if d > 0)
            n_w_wins = sum(1 for d in diffs if d < 0)
            print(f"  {key:<20}  Δ mean={np.mean(diffs):+.4f}  "
                  f"正确>错误: {n_c_wins}/{len(diffs)} ({n_c_wins/len(diffs):.0%})  "
                  f"正确 mean={np.mean(c_vals):.4f}  错误 mean={np.mean(w_vals):.4f}")
    
    # ============================================================
    print(f"\n--- 8. AEP reward-conditioned 效果预估 (√n=√32≈5.66) ---")
    sqrt_n = 32 ** 0.5
    
    c_delta_per_token = [s['tanh_sum_per_token'] / sqrt_n for s in correct_stats]
    w_delta_per_token = [-s['tanh_sum_per_token'] / sqrt_n for s in wrong_stats]
    
    print(f"  正确样本 δ/token (A 的平均增量): +{np.mean(c_delta_per_token):.5f}")
    print(f"  错误样本 δ/token (A 的平均减量): {np.mean(w_delta_per_token):.5f}")
    print(f"  净区分力 (每 token): {np.mean(c_delta_per_token) - np.mean(w_delta_per_token):.5f}")
    
    # Simulate effective advantage
    A_c = 1.0  # typical correct advantage
    A_w = -1.0  # typical wrong advantage
    c_boost = np.mean([s['tanh_sum_per_token'] for s in correct_stats]) / sqrt_n
    w_boost = np.mean([s['tanh_sum_per_token'] for s in wrong_stats]) / sqrt_n
    
    print(f"\n  模拟 (A_correct=+1, A_wrong=-1):")
    print(f"    正确: A + δ = {A_c} + {c_boost:.4f} = {A_c + c_boost:.4f}")
    print(f"    错误: A - δ = {A_w} - {w_boost:.4f} = {A_w - w_boost:.4f}")
    print(f"    净 gap: {(A_c + c_boost) - (A_w - w_boost):.4f} (纯 GRPO: 2.0)")

# Load and analyze
print("加载数据...")
data_later = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json")
data_early = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json")

analyze(data_later, "后期数据")
analyze(data_early, "早期数据")
