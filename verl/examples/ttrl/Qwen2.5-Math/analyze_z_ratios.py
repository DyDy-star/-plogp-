"""
分析: 正确/错误样本中 z>0 vs z<0 的比值特征
这里使用原始 AEP 的 z (全 ε 归一化), 不是只看 ε>0
"""
import json
import numpy as np
from collections import defaultdict

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_full_z(H_arr):
    """原始 z: z_t = ε_t / σ_all"""
    n = len(H_arr)
    if n < 10:
        return None, None
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
    sigma = np.std(epsilon[1:])
    if sigma < 1e-6:
        return None, None
    z = epsilon / sigma
    return z, epsilon

def analyze(data, label):
    print(f"\n{'='*95}")
    print(f"  {label}  (pass@1={data['overall_pass@1']:.4f})")
    print(f"{'='*95}")
    
    correct = []
    wrong = []
    
    for problem in data['results']:
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
            result = compute_full_z(H)
            if result[0] is None:
                continue
            z, eps = result
            z_valid = z[1:]  # skip t=0 (ε=0)
            
            n = len(z_valid)
            n_pos = int(np.sum(z_valid > 0))
            n_neg = int(np.sum(z_valid < 0))
            n_zero = n - n_pos - n_neg
            
            # 各阈值
            thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
            counts = {}
            for th in thresholds:
                n_above = int(np.sum(z_valid > th))
                n_below = int(np.sum(z_valid < -th))
                counts[f'n_gt{th}'] = n_above
                counts[f'n_lt{th}'] = n_below
                counts[f'ratio_{th}'] = n_above / max(n_below, 1)
            
            entry = {
                'n_tokens': n,
                'n_pos': n_pos,
                'n_neg': n_neg,
                'ratio_pos_neg': n_pos / max(n_neg, 1),
                'frac_pos': n_pos / n,
                'frac_neg': n_neg / n,
                'z_pos_mean': float(z_valid[z_valid > 0].mean()) if n_pos > 0 else 0,
                'z_neg_mean': float(z_valid[z_valid < 0].mean()) if n_neg > 0 else 0,
                'z_pos_sum': float(z_valid[z_valid > 0].sum()),
                'z_neg_sum': float(z_valid[z_valid < 0].sum()),
                'sum_ratio': float(z_valid[z_valid > 0].sum()) / max(abs(float(z_valid[z_valid < 0].sum())), 1e-8),
                **counts,
            }
            
            if is_correct:
                correct.append(entry)
            else:
                wrong.append(entry)
    
    nc, nw = len(correct), len(wrong)
    print(f"  正确: {nc}, 错误: {nw}")
    
    def compare(key, label_str, fmt=".4f"):
        cv = [s[key] for s in correct]
        wv = [s[key] for s in wrong]
        cm, cs = np.mean(cv), np.std(cv)
        wm, ws = np.mean(wv), np.std(wv)
        d = (cm - wm) / max((cs + ws) / 2, 1e-8)
        print(f"  {label_str:<40} 正确: {cm:{fmt}}±{cs:{fmt}}  错误: {wm:{fmt}}±{ws:{fmt}}  Δ={cm-wm:+{fmt}}  d={d:+.3f}")
    
    # =============================================
    print(f"\n--- 1. z>0 vs z<0 的数量与比值 ---")
    compare('frac_pos', 'z>0 占比')
    compare('frac_neg', 'z<0 占比')
    compare('ratio_pos_neg', 'z>0/z<0 数量比 ★')
    
    # =============================================
    print(f"\n--- 2. z>0 vs z<0 的幅度 ---")
    compare('z_pos_mean', 'z>0 的均值')
    compare('z_neg_mean', 'z<0 的均值 (负数)')
    compare('z_pos_sum', 'z>0 的总和')
    compare('z_neg_sum', 'z<0 的总和 (负数)')
    compare('sum_ratio', 'Σ(z>0) / |Σ(z<0)| 总和比 ★')
    
    # =============================================
    print(f"\n--- 3. 各阈值的 |z|>θ 数量比 ---")
    for th in [0.5, 1.0, 1.5, 2.0, 3.0]:
        compare(f'n_gt{th}', f'z>{th} 个数', '.1f')
        compare(f'n_lt{th}', f'z<-{th} 个数', '.1f')
        compare(f'ratio_{th}', f'z>{th} / z<-{th} 比值 ★')
        print()
    
    # =============================================
    print(f"\n--- 4. 按问题分组: 比值特征 ---")
    problem_data = defaultdict(lambda: {'c': [], 'w': []})
    for problem in data['results']:
        pid = problem['id']
        for resp in problem['responses']:
            is_correct = resp['is_correct']
            ea = resp.get('entropy_analysis')
            if not ea or 'steps' not in ea:
                continue
            all_ents = []
            for step in ea['steps']:
                all_ents.extend(step.get('token_entropies', []))
            if len(all_ents) < 20:
                continue
            H = np.array(all_ents)
            result = compute_full_z(H)
            if result[0] is None:
                continue
            z = result[0][1:]
            n_pos = int(np.sum(z > 0))
            n_neg = int(np.sum(z < 0))
            ratio = n_pos / max(n_neg, 1)
            
            n_gt2 = int(np.sum(z > 2))
            n_lt2 = int(np.sum(z < -2))
            ratio_2 = n_gt2 / max(n_lt2, 1)
            
            entry = {'ratio_pos_neg': ratio, 'ratio_2': ratio_2}
            if is_correct:
                problem_data[pid]['c'].append(entry)
            else:
                problem_data[pid]['w'].append(entry)
    
    for key, key_label in [('ratio_pos_neg', 'z>0/z<0'), ('ratio_2', 'z>2/z<-2')]:
        diffs = []
        c_wins = 0
        total = 0
        for pid, d in problem_data.items():
            if d['c'] and d['w']:
                c_val = np.mean([e[key] for e in d['c']])
                w_val = np.mean([e[key] for e in d['w']])
                diffs.append(c_val - w_val)
                if c_val > w_val:
                    c_wins += 1
                total += 1
        if diffs:
            print(f"  {key_label:<20}  Δ mean={np.mean(diffs):+.4f}  "
                  f"正确>错误: {c_wins}/{total} ({c_wins/total:.0%})  "
                  f"Δ median={np.median(diffs):+.4f}")
    
    # =============================================
    print(f"\n--- 5. 分位数详细对比 ---")
    for key, key_label in [('ratio_pos_neg', 'z>0/z<0 比值'),
                            ('ratio_2.0', 'z>2/z<-2 比值'),
                            ('ratio_1.0', 'z>1/z<-1 比值')]:
        cv = [s[key] for s in correct]
        wv = [s[key] for s in wrong]
        pcts = [10, 25, 50, 75, 90]
        c_pcts = np.percentile(cv, pcts)
        w_pcts = np.percentile(wv, pcts)
        print(f"\n  {key_label}:")
        print(f"    {'分位':>6}  {'正确':>8}  {'错误':>8}  {'Δ':>8}")
        for i, p in enumerate(pcts):
            print(f"    {p:>5}%  {c_pcts[i]:>8.3f}  {w_pcts[i]:>8.3f}  {c_pcts[i]-w_pcts[i]:>+8.3f}")

# Load
data_later = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json")
data_early = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json")

analyze(data_later, "后期数据")
analyze(data_early, "早期数据")
