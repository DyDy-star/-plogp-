"""
深度分析: 正确/错误样本中 z-score 极端 token 的类型分布
z_t = epsilon_t / sigma, epsilon_t = H_t - EMA[t-1]  (AEP innovation)
"""
import json
import numpy as np
from collections import Counter, defaultdict
import re

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_z_scores(token_entropies):
    """Replicate AEP Step 1 z-score computation"""
    H = np.array(token_entropies, dtype=np.float64)
    n = len(H)
    if n < 10:
        return np.zeros(n), 0.0, 0.0
    
    rho_matrix = np.corrcoef(H[:-1], H[1:])
    rho = rho_matrix[0, 1]
    if np.isnan(rho):
        rho = 0.0
    rho = max(0.05, min(0.95, rho))
    alpha = 1.0 - rho
    
    ema = np.zeros(n)
    epsilon = np.zeros(n)
    ema[0] = H[0]
    for t in range(1, n):
        epsilon[t] = H[t] - ema[t - 1]
        ema[t] = alpha * H[t] + (1 - alpha) * ema[t - 1]
    
    sigma = np.std(epsilon[1:])
    if sigma < 1e-6:
        return np.zeros(n), rho, 0.0
    
    z = epsilon / sigma
    return z, rho, sigma

def classify_token(tok):
    """将 token 分类为语义类别"""
    tok_stripped = tok.strip()
    if not tok_stripped:
        return "whitespace/newline"
    if tok_stripped in [',', '.', ';', ':', '!', '?', ')', '(', '{', '}', '[', ']']:
        return "punctuation"
    if tok_stripped in ['\\(', '\\)', '\\[', '\\]', '\\\\', '\\{', '\\}']:
        return "latex_delim"
    if re.match(r'^\\[a-zA-Z]+$', tok_stripped):
        return "latex_cmd"
    if tok_stripped.startswith('**') or tok_stripped.endswith('**'):
        return "markdown_bold"
    if tok_stripped.startswith('```'):
        return "code_fence"
    if re.match(r'^-?\d+\.?\d*$', tok_stripped):
        return "number"
    if re.match(r'^[+\-*/=<>^_]+$', tok_stripped):
        return "math_op"
    if tok_stripped.lower() in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                                 'to', 'of', 'in', 'for', 'on', 'at', 'by', 'with', 'from',
                                 'and', 'or', 'but', 'so', 'if', 'then', 'that', 'this',
                                 'we', 'i', 'you', 'it', 'he', 'she', 'they', 'them',
                                 'can', 'will', 'would', 'could', 'should', 'do', 'does',
                                 'not', 'no', 'yes', 'have', 'has', 'had']:
        return "function_word"
    if tok_stripped.lower() in ['let', "'s", 'break', 'step', 'solve', 'find', 'need',
                                 'first', 'next', 'now', 'therefore', 'thus', 'hence',
                                 'since', 'because', 'given', 'note', 'recall', 'consider',
                                 'suppose', 'assume', 'know', 'verify', 'check', 'calculate',
                                 'compute', 'determine', 'substitute', 'simplify', 'expand',
                                 'equation', 'expression', 'value', 'result', 'answer',
                                 'solution', 'problem']:
        return "reasoning_word"
    if tok_stripped.lower() in ['\\boxed', 'boxed']:
        return "boxed_answer"
    if tok_stripped.startswith('\n') or tok_stripped == '\n':
        return "newline"
    return "content_word"

def analyze_responses(data, label=""):
    correct_z_by_cat = defaultdict(list)
    wrong_z_by_cat = defaultdict(list)
    
    correct_extreme_high = []
    correct_extreme_low = []
    wrong_extreme_high = []
    wrong_extreme_low = []
    
    correct_z_all = []
    wrong_z_all = []
    
    correct_rhos = []
    wrong_rhos = []
    correct_sigmas = []
    wrong_sigmas = []
    
    correct_entropy_profiles = []
    wrong_entropy_profiles = []

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
            
            z_scores, rho, sigma = compute_z_scores(all_entropies)
            
            if sigma < 1e-6:
                continue
            
            if is_correct:
                n_correct += 1
                correct_rhos.append(rho)
                correct_sigmas.append(sigma)
                correct_entropy_profiles.append(np.array(all_entropies))
            else:
                n_wrong += 1
                wrong_rhos.append(rho)
                wrong_sigmas.append(sigma)
                wrong_entropy_profiles.append(np.array(all_entropies))
            
            for i, (tok, z_val) in enumerate(zip(all_tokens, z_scores)):
                cat = classify_token(tok)
                if is_correct:
                    correct_z_by_cat[cat].append(z_val)
                    correct_z_all.append(z_val)
                else:
                    wrong_z_by_cat[cat].append(z_val)
                    wrong_z_all.append(z_val)
                
                if abs(z_val) > 2.0:
                    entry = {
                        'token': tok,
                        'z': z_val,
                        'entropy': all_entropies[i],
                        'category': cat,
                        'position_ratio': i / len(all_tokens),
                        'problem_id': problem['id'],
                    }
                    if is_correct:
                        if z_val > 2.0:
                            correct_extreme_high.append(entry)
                        else:
                            correct_extreme_low.append(entry)
                    else:
                        if z_val > 2.0:
                            wrong_extreme_high.append(entry)
                        else:
                            wrong_extreme_low.append(entry)
    
    print(f"\n{'='*80}")
    print(f"  数据集: {label}")
    print(f"  正确样本数: {n_correct}, 错误样本数: {n_wrong}")
    print(f"{'='*80}")
    
    # 1. rho and sigma comparison
    print(f"\n--- 自相关 rho & 创新标准差 sigma ---")
    print(f"  正确: rho={np.mean(correct_rhos):.4f}±{np.std(correct_rhos):.4f}, "
          f"sigma={np.mean(correct_sigmas):.4f}±{np.std(correct_sigmas):.4f}")
    print(f"  错误: rho={np.mean(wrong_rhos):.4f}±{np.std(wrong_rhos):.4f}, "
          f"sigma={np.mean(wrong_sigmas):.4f}±{np.std(wrong_sigmas):.4f}")
    
    # 2. z-score distribution comparison
    print(f"\n--- z-score 整体分布 ---")
    cz = np.array(correct_z_all)
    wz = np.array(wrong_z_all)
    print(f"  正确: mean={cz.mean():.4f}, std={cz.std():.4f}, "
          f"|z|>2 比例={np.mean(np.abs(cz)>2):.4f}, |z|>3 比例={np.mean(np.abs(cz)>3):.4f}")
    print(f"  错误: mean={wz.mean():.4f}, std={wz.std():.4f}, "
          f"|z|>2 比例={np.mean(np.abs(wz)>2):.4f}, |z|>3 比例={np.mean(np.abs(wz)>3):.4f}")
    
    # 3. z-score by token category
    print(f"\n--- 各 token 类别的 z-score 分布 (正确 vs 错误) ---")
    all_cats = sorted(set(list(correct_z_by_cat.keys()) + list(wrong_z_by_cat.keys())))
    print(f"  {'类别':<20} {'正确 mean':>10} {'正确 std':>10} {'正确 N':>8} "
          f"{'错误 mean':>10} {'错误 std':>10} {'错误 N':>8} {'Δmean':>8}")
    for cat in all_cats:
        cvals = np.array(correct_z_by_cat.get(cat, [0]))
        wvals = np.array(wrong_z_by_cat.get(cat, [0]))
        delta = cvals.mean() - wvals.mean()
        print(f"  {cat:<20} {cvals.mean():>10.4f} {cvals.std():>10.4f} {len(correct_z_by_cat.get(cat,[]))}:>8 "
              f"{wvals.mean():>10.4f} {wvals.std():>10.4f} {len(wrong_z_by_cat.get(cat,[]))}:>8 {delta:>8.4f}")
    
    # 4. Extreme z tokens analysis
    print(f"\n--- 极端 z>2 的 token 类别分布 ---")
    print(f"  正确样本 z>2 (共{len(correct_extreme_high)}个):")
    ch_cats = Counter(e['category'] for e in correct_extreme_high)
    for cat, cnt in ch_cats.most_common(10):
        avg_z = np.mean([e['z'] for e in correct_extreme_high if e['category'] == cat])
        avg_pos = np.mean([e['position_ratio'] for e in correct_extreme_high if e['category'] == cat])
        print(f"    {cat:<20} 次数={cnt:>5}  平均z={avg_z:.2f}  平均位置={avg_pos:.2f}")
    
    print(f"\n  错误样本 z>2 (共{len(wrong_extreme_high)}个):")
    wh_cats = Counter(e['category'] for e in wrong_extreme_high)
    for cat, cnt in wh_cats.most_common(10):
        avg_z = np.mean([e['z'] for e in wrong_extreme_high if e['category'] == cat])
        avg_pos = np.mean([e['position_ratio'] for e in wrong_extreme_high if e['category'] == cat])
        print(f"    {cat:<20} 次数={cnt:>5}  平均z={avg_z:.2f}  平均位置={avg_pos:.2f}")
    
    print(f"\n--- 极端 z<-2 的 token 类别分布 ---")
    print(f"  正确样本 z<-2 (共{len(correct_extreme_low)}个):")
    cl_cats = Counter(e['category'] for e in correct_extreme_low)
    for cat, cnt in cl_cats.most_common(10):
        avg_z = np.mean([e['z'] for e in correct_extreme_low if e['category'] == cat])
        avg_pos = np.mean([e['position_ratio'] for e in correct_extreme_low if e['category'] == cat])
        print(f"    {cat:<20} 次数={cnt:>5}  平均z={avg_z:.2f}  平均位置={avg_pos:.2f}")
    
    print(f"\n  错误样本 z<-2 (共{len(wrong_extreme_low)}个):")
    wl_cats = Counter(e['category'] for e in wrong_extreme_low)
    for cat, cnt in wl_cats.most_common(10):
        avg_z = np.mean([e['z'] for e in wrong_extreme_low if e['category'] == cat])
        avg_pos = np.mean([e['position_ratio'] for e in wrong_extreme_low if e['category'] == cat])
        print(f"    {cat:<20} 次数={cnt:>5}  平均z={avg_z:.2f}  平均位置={avg_pos:.2f}")
    
    # 5. Top tokens with extreme z
    print(f"\n--- 正确样本: z 最高的 20 个具体 token ---")
    sorted_high_c = sorted(correct_extreme_high, key=lambda x: x['z'], reverse=True)[:20]
    for e in sorted_high_c:
        print(f"    z={e['z']:>7.2f}  H={e['entropy']:.3f}  pos={e['position_ratio']:.2f}  "
              f"cat={e['category']:<18}  tok='{e['token']}'")
    
    print(f"\n--- 正确样本: z 最低的 20 个具体 token ---")
    sorted_low_c = sorted(correct_extreme_low, key=lambda x: x['z'])[:20]
    for e in sorted_low_c:
        print(f"    z={e['z']:>7.2f}  H={e['entropy']:.3f}  pos={e['position_ratio']:.2f}  "
              f"cat={e['category']:<18}  tok='{e['token']}'")
    
    print(f"\n--- 错误样本: z 最高的 20 个具体 token ---")
    sorted_high_w = sorted(wrong_extreme_high, key=lambda x: x['z'], reverse=True)[:20]
    for e in sorted_high_w:
        print(f"    z={e['z']:>7.2f}  H={e['entropy']:.3f}  pos={e['position_ratio']:.2f}  "
              f"cat={e['category']:<18}  tok='{e['token']}'")
    
    print(f"\n--- 错误样本: z 最低的 20 个具体 token ---")
    sorted_low_w = sorted(wrong_extreme_low, key=lambda x: x['z'])[:20]
    for e in sorted_low_w:
        print(f"    z={e['z']:>7.2f}  H={e['entropy']:.3f}  pos={e['position_ratio']:.2f}  "
              f"cat={e['category']:<18}  tok='{e['token']}'")
    
    # 6. Position analysis: where do extreme z tokens occur?
    print(f"\n--- 极端 z token 的位置分布 (0=开头, 1=结尾) ---")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    def pos_hist(entries, label):
        positions = [e['position_ratio'] for e in entries]
        if not positions:
            return
        hist, _ = np.histogram(positions, bins=bins)
        total = len(positions)
        print(f"  {label}:")
        for i in range(len(bins)-1):
            bar = '█' * int(hist[i] / total * 50) if total > 0 else ''
            print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:>5} ({hist[i]/total*100:>5.1f}%) {bar}")
    
    pos_hist(correct_extreme_high, "正确 z>2")
    pos_hist(wrong_extreme_high, "错误 z>2")
    pos_hist(correct_extreme_low, "正确 z<-2")
    pos_hist(wrong_extreme_low, "错误 z<-2")
    
    # 7. Entropy profile analysis
    print(f"\n--- 熵时间轮廓 (按位置分段) ---")
    def profile_stats(profiles, label):
        if not profiles:
            return
        early_ents = []
        mid_ents = []
        late_ents = []
        for prof in profiles:
            n = len(prof)
            q1 = n // 3
            q2 = 2 * n // 3
            early_ents.extend(prof[:q1].tolist())
            mid_ents.extend(prof[q1:q2].tolist())
            late_ents.extend(prof[q2:].tolist())
        print(f"  {label}: early={np.mean(early_ents):.4f}, mid={np.mean(mid_ents):.4f}, late={np.mean(late_ents):.4f}")
    
    profile_stats(correct_entropy_profiles, "正确")
    profile_stats(wrong_entropy_profiles, "错误")
    
    # 8. tanh(z) impact analysis
    print(f"\n--- tanh(z) 加权对 advantage 的影响 ---")
    c_tanh = np.tanh(cz)
    w_tanh = np.tanh(wz)
    print(f"  正确: mean(tanh(z))={c_tanh.mean():.4f}, std={c_tanh.std():.4f}")
    print(f"  错误: mean(tanh(z))={w_tanh.mean():.4f}, std={w_tanh.std():.4f}")
    print(f"  差异: {c_tanh.mean() - w_tanh.mean():.4f}")
    
    # Tanh(z) by position
    print(f"\n  正确/错误样本 tanh(z) 在不同位置的均值:")
    for entries, label in [(correct_extreme_high + correct_extreme_low, "正确极端"),
                           (wrong_extreme_high + wrong_extreme_low, "错误极端")]:
        early = [np.tanh(e['z']) for e in entries if e['position_ratio'] < 0.33]
        mid = [np.tanh(e['z']) for e in entries if 0.33 <= e['position_ratio'] < 0.66]
        late = [np.tanh(e['z']) for e in entries if e['position_ratio'] >= 0.66]
        print(f"    {label}: early={np.mean(early):.4f} mid={np.mean(mid):.4f} late={np.mean(late):.4f}" 
              if early and mid and late else f"    {label}: 数据不足")
    
    return {
        'correct_z': cz, 'wrong_z': wz,
        'correct_rhos': correct_rhos, 'wrong_rhos': wrong_rhos,
        'correct_sigmas': correct_sigmas, 'wrong_sigmas': wrong_sigmas,
    }

# Load data
print("加载数据文件...")
data_later = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json")
data_early = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json")

print(f"\n后期 (pass@1={data_later['overall_pass@1']:.4f}):")
stats_later = analyze_responses(data_later, "后期训练 (pass@1=0.26)")

print(f"\n\n早期 (pass@1={data_early['overall_pass@1']:.4f}):")
stats_early = analyze_responses(data_early, "早期训练 (pass@1=0.04)")

# Cross-comparison
print(f"\n\n{'='*80}")
print(f"  早期 vs 后期 对比")
print(f"{'='*80}")
print(f"\n  早期正确: z mean={stats_early['correct_z'].mean():.4f}, std={stats_early['correct_z'].std():.4f}")
print(f"  后期正确: z mean={stats_later['correct_z'].mean():.4f}, std={stats_later['correct_z'].std():.4f}")
print(f"  早期错误: z mean={stats_early['wrong_z'].mean():.4f}, std={stats_early['wrong_z'].std():.4f}")
print(f"  后期错误: z mean={stats_later['wrong_z'].mean():.4f}, std={stats_later['wrong_z'].std():.4f}")
