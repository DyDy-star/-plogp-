"""
分析用户提出的改进方案:
- 仅对 ε>0 的 token 赋权 A*(1+tanh(z))
- z<=0 的 token 权重为 1
- z_t = ε_t / σ_positive (σ 仅从 ε>0 的 token 计算)
"""
import json
import numpy as np
from collections import Counter, defaultdict

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_original_z(H_arr):
    """原始 AEP z-score"""
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
    
    sigma_all = np.std(epsilon[1:])
    if sigma_all < 1e-6:
        return None, None, None, None
    
    z_original = epsilon / sigma_all
    return epsilon, z_original, rho, sigma_all

def compute_proposed_z(epsilon):
    """用户提出的方案: σ 仅从 ε>0 计算"""
    pos_eps = epsilon[epsilon > 0]
    if len(pos_eps) < 3:
        return None, None
    
    sigma_pos = np.std(pos_eps)
    if sigma_pos < 1e-6:
        return None, None
    
    z_proposed = np.where(epsilon > 0, epsilon / sigma_pos, 0.0)
    return z_proposed, sigma_pos

def analyze(data, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    
    correct_weights_orig = []
    correct_weights_prop = []
    wrong_weights_orig = []
    wrong_weights_prop = []
    
    correct_boost_tokens = []  # ε>0 且获得加权的 token (正确样本)
    wrong_boost_tokens = []    # ε>0 且获得加权的 token (错误样本)
    
    correct_tanh_at_boost = []
    wrong_tanh_at_boost = []
    
    garbage_boost_correct = 0
    garbage_boost_wrong = 0
    garbage_total_correct = 0
    garbage_total_wrong = 0
    
    n_correct = 0
    n_wrong = 0
    
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
            result = compute_original_z(H)
            if result[0] is None:
                continue
            
            epsilon, z_orig, rho, sigma_all = result
            result2 = compute_proposed_z(epsilon)
            if result2[0] is None:
                continue
            
            z_prop, sigma_pos = result2
            
            if is_correct:
                n_correct += 1
            else:
                n_wrong += 1
            
            for i, (tok, h_val) in enumerate(zip(all_tokens, all_entropies)):
                eps_val = epsilon[i]
                
                # 原始 AEP: tanh(z) 加到 advantage
                w_orig = np.tanh(z_orig[i])
                
                # 提议方案: ε>0 → A*(1+tanh(z)), else → A*1
                if eps_val > 0:
                    w_prop = 1.0 + np.tanh(z_prop[i])  # ∈ (1, 2)
                else:
                    w_prop = 1.0
                
                if is_correct:
                    correct_weights_orig.append(w_orig)
                    correct_weights_prop.append(w_prop)
                else:
                    wrong_weights_orig.append(w_orig)
                    wrong_weights_prop.append(w_prop)
                
                is_garbage = h_val > 4.0
                
                if eps_val > 0:
                    entry = {
                        'token': tok, 'entropy': h_val,
                        'eps': eps_val, 'z': z_prop[i],
                        'weight': w_prop, 'is_garbage': is_garbage,
                    }
                    if is_correct:
                        correct_boost_tokens.append(entry)
                        correct_tanh_at_boost.append(np.tanh(z_prop[i]))
                        if is_garbage:
                            garbage_boost_correct += 1
                        garbage_total_correct += (1 if is_garbage else 0)
                    else:
                        wrong_boost_tokens.append(entry)
                        wrong_tanh_at_boost.append(np.tanh(z_prop[i]))
                        if is_garbage:
                            garbage_boost_wrong += 1
                        garbage_total_wrong += (1 if is_garbage else 0)
    
    co = np.array(correct_weights_orig)
    cp = np.array(correct_weights_prop)
    wo = np.array(wrong_weights_orig)
    wp = np.array(wrong_weights_prop)
    ct = np.array(correct_tanh_at_boost)
    wt = np.array(wrong_tanh_at_boost)
    
    print(f"\n正确样本: {n_correct}, 错误样本: {n_wrong}")
    
    # 1. 权重分布对比
    print(f"\n--- 权重分布对比 ---")
    print(f"  原始 AEP (additive tanh(z)):")
    print(f"    正确: mean={co.mean():.6f}, std={co.std():.4f}")
    print(f"    错误: mean={wo.mean():.6f}, std={wo.std():.4f}")
    print(f"    Δ(正确-错误) = {co.mean()-wo.mean():.6f}")
    
    print(f"\n  提议方案 (multiplicative 1+tanh for ε>0):")
    print(f"    正确: mean={cp.mean():.6f}, std={cp.std():.4f}")
    print(f"    错误: mean={wp.mean():.6f}, std={wp.std():.4f}")
    print(f"    Δ(正确-错误) = {cp.mean()-wp.mean():.6f}")
    
    # 2. Boost 区域 (ε>0) 的 tanh 分布
    print(f"\n--- ε>0 区域的 tanh(z) 分布 ---")
    print(f"  正确 (N={len(ct)}): mean={ct.mean():.6f}, std={ct.std():.4f}")
    print(f"  错误 (N={len(wt)}): mean={wt.mean():.6f}, std={wt.std():.4f}")
    print(f"  Δ = {ct.mean()-wt.mean():.6f}")
    
    # 3. ε>0 token 中 content 分布
    print(f"\n--- ε>0 获得加权的 token 占比 ---")
    n_boost_c = len(correct_boost_tokens)
    n_total_c = len(correct_weights_prop)
    n_boost_w = len(wrong_boost_tokens)
    n_total_w = len(wrong_weights_prop)
    print(f"  正确: {n_boost_c}/{n_total_c} = {n_boost_c/n_total_c:.2%}")
    print(f"  错误: {n_boost_w}/{n_total_w} = {n_boost_w/n_total_w:.2%}")
    
    # 4. 垃圾 token (H>4) 在 boost 区域的比例
    print(f"\n--- 垃圾 token (H>4) 在 ε>0 boost 区域的情况 ---")
    gb_c = sum(1 for e in correct_boost_tokens if e['is_garbage'])
    gb_w = sum(1 for e in wrong_boost_tokens if e['is_garbage'])
    print(f"  正确样本: 垃圾={gb_c}, 占boost区={gb_c/max(n_boost_c,1):.4%}")
    print(f"  错误样本: 垃圾={gb_w}, 占boost区={gb_w/max(n_boost_w,1):.4%}")
    
    # 垃圾 token 的平均 weight
    gb_weights_c = [e['weight'] for e in correct_boost_tokens if e['is_garbage']]
    gb_weights_w = [e['weight'] for e in wrong_boost_tokens if e['is_garbage']]
    if gb_weights_c:
        print(f"  正确垃圾 token 平均 weight: {np.mean(gb_weights_c):.4f}")
    if gb_weights_w:
        print(f"  错误垃圾 token 平均 weight: {np.mean(gb_weights_w):.4f}")
    
    # 5. Top boosted tokens in wrong samples
    print(f"\n--- 错误样本中 weight 最高的 20 个 token ---")
    wrong_sorted = sorted(wrong_boost_tokens, key=lambda x: x['weight'], reverse=True)[:20]
    for e in wrong_sorted:
        tag = " ⚠️GARBAGE" if e['is_garbage'] else ""
        print(f"    w={e['weight']:.4f}  H={e['entropy']:.3f}  z={e['z']:.2f}  tok='{e['token']}'{tag}")
    
    print(f"\n--- 正确样本中 weight 最高的 20 个 token ---")
    correct_sorted = sorted(correct_boost_tokens, key=lambda x: x['weight'], reverse=True)[:20]
    for e in correct_sorted:
        tag = " ⚠️GARBAGE" if e['is_garbage'] else ""
        print(f"    w={e['weight']:.4f}  H={e['entropy']:.3f}  z={e['z']:.2f}  tok='{e['token']}'{tag}")
    
    # 6. 核心问题: 模拟 GRPO 梯度效果
    print(f"\n--- 模拟 GRPO 梯度效果 ---")
    print(f"  假设 GRPO advantage: 正确 A=+1, 错误 A=-1")
    print(f"\n  原始 AEP (A + tanh(z)):")
    eff_correct_orig = 1.0 + co.mean()
    eff_wrong_orig = -1.0 + wo.mean()
    print(f"    正确有效 advantage: 1 + {co.mean():.6f} = {eff_correct_orig:.6f}")
    print(f"    错误有效 advantage: -1 + {wo.mean():.6f} = {eff_wrong_orig:.6f}")
    print(f"    净梯度方向强度: {eff_correct_orig - eff_wrong_orig:.6f}")
    
    print(f"\n  提议方案 (A * w_t):")
    eff_correct_prop = 1.0 * cp.mean()
    eff_wrong_prop = -1.0 * wp.mean()
    print(f"    正确有效 advantage: +1 × {cp.mean():.6f} = {eff_correct_prop:.6f}")
    print(f"    错误有效 advantage: -1 × {wp.mean():.6f} = {eff_wrong_prop:.6f}")
    print(f"    净梯度方向强度: {eff_correct_prop - eff_wrong_prop:.6f}")
    
    print(f"\n  无修改 (纯 GRPO):")
    print(f"    正确有效 advantage: +1")
    print(f"    错误有效 advantage: -1")
    print(f"    净梯度方向强度: 2.0")
    
    # 7. 按熵区间分析 boost 效果
    print(f"\n--- 按熵区间分析 boost token 的 weight ---")
    entropy_bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 100.0)]
    for lo, hi in entropy_bins:
        c_in_bin = [e['weight'] for e in correct_boost_tokens if lo <= e['entropy'] < hi]
        w_in_bin = [e['weight'] for e in wrong_boost_tokens if lo <= e['entropy'] < hi]
        c_mean = np.mean(c_in_bin) if c_in_bin else 0
        w_mean = np.mean(w_in_bin) if w_in_bin else 0
        label = f"H∈[{lo},{hi})"
        print(f"  {label:<12}  正确: N={len(c_in_bin):>6} w={c_mean:.4f}  "
              f"错误: N={len(w_in_bin):>6} w={w_mean:.4f}  Δ={c_mean-w_mean:.4f}")

# Load and analyze
data_later = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json")
data_early = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json")

analyze(data_later, "后期数据 (pass@1=0.26)")
analyze(data_early, "早期数据 (pass@1=0.04)")
