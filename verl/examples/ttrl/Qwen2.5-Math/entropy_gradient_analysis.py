"""
熵梯度阻断：假说验证
5 个实验，验证"低熵 token 梯度污染高熵 token"假说
"""
import json
import numpy as np
from collections import Counter, defaultdict

LATE_PATH = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json"
EARLY_PATH = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json"

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_sequences(data, min_len=30):
    """Extract (tokens, entropies, is_correct) for each response."""
    seqs = []
    for problem in data['results']:
        for resp in problem['responses']:
            ea = resp.get('entropy_analysis')
            if not ea or 'steps' not in ea:
                continue
            tokens, ents = [], []
            for step in ea['steps']:
                toks = step.get('tokens', [])
                es = step.get('token_entropies', [])
                for tok, ent in zip(toks, es):
                    tokens.append(tok)
                    ents.append(ent)
            if len(tokens) >= min_len:
                seqs.append((tokens, np.array(ents), resp['is_correct']))
    return seqs

# =====================================================================
# EXPERIMENT 1: Low-entropy context suppression of high-entropy tokens
# =====================================================================
def experiment_1(seqs, label):
    print(f"\n{'='*90}")
    print(f"  实验 1: 低熵上下文对高熵 token 的压制效应 [{label}]")
    print(f"{'='*90}")

    all_H = np.concatenate([s[1] for s in seqs])
    P25, P50, P75 = np.percentile(all_H, [25, 50, 75])
    print(f"  全局分位数: P25={P25:.4f}, P50={P50:.4f}, P75={P75:.4f}")

    for k in [3, 5, 10]:
        results = {'low_ctx': [], 'mid_ctx': [], 'high_ctx': []}
        for tokens, H, is_correct in seqs:
            n = len(H)
            for t in range(k, n):
                if H[t] <= P50:
                    continue
                ctx_mean = H[t-k:t].mean()
                if ctx_mean < P25:
                    results['low_ctx'].append(H[t])
                elif ctx_mean > P75:
                    results['high_ctx'].append(H[t])
                else:
                    results['mid_ctx'].append(H[t])

        for key in ['low_ctx', 'mid_ctx', 'high_ctx']:
            results[key] = np.array(results[key]) if results[key] else np.array([0])

        diff = results['high_ctx'].mean() - results['low_ctx'].mean()
        pooled_std = (results['high_ctx'].std() + results['low_ctx'].std()) / 2
        d = diff / max(pooled_std, 1e-8)

        print(f"\n  k={k} (前{k}个token的均熵作为上下文):")
        print(f"    H>P50 token 在 low_ctx 中: mean={results['low_ctx'].mean():.4f} "
              f"(N={len(results['low_ctx'])})")
        print(f"    H>P50 token 在 mid_ctx 中: mean={results['mid_ctx'].mean():.4f} "
              f"(N={len(results['mid_ctx'])})")
        print(f"    H>P50 token 在 high_ctx 中: mean={results['high_ctx'].mean():.4f} "
              f"(N={len(results['high_ctx'])})")
        print(f"    Δ(high_ctx - low_ctx) = {diff:+.4f}, Cohen's d = {d:+.3f}")

    # correct vs wrong breakdown
    print(f"\n  --- 正确 vs 错误 (k=5) ---")
    for cond_label, cond_fn in [("正确", lambda c: c), ("错误", lambda c: not c)]:
        results = {'low_ctx': [], 'high_ctx': []}
        for tokens, H, is_correct in seqs:
            if not cond_fn(is_correct):
                continue
            n = len(H)
            for t in range(5, n):
                if H[t] <= P50:
                    continue
                ctx_mean = H[t-5:t].mean()
                if ctx_mean < P25:
                    results['low_ctx'].append(H[t])
                elif ctx_mean > P75:
                    results['high_ctx'].append(H[t])
        for key in results:
            results[key] = np.array(results[key]) if results[key] else np.array([0])
        diff = results['high_ctx'].mean() - results['low_ctx'].mean()
        print(f"    {cond_label}: low_ctx={results['low_ctx'].mean():.4f} (N={len(results['low_ctx'])}), "
              f"high_ctx={results['high_ctx'].mean():.4f} (N={len(results['high_ctx'])}), "
              f"Δ={diff:+.4f}")


# =====================================================================
# EXPERIMENT 2: Entropy transition matrix (1-step vs 3-step)
# =====================================================================
def experiment_2(seqs, label):
    print(f"\n{'='*90}")
    print(f"  实验 2: 熵转移矩阵 (多步依赖) [{label}]")
    print(f"{'='*90}")

    all_H = np.concatenate([s[1] for s in seqs])
    quartiles = np.percentile(all_H, [25, 50, 75])

    def to_q(h):
        if h <= quartiles[0]: return 0
        elif h <= quartiles[1]: return 1
        elif h <= quartiles[2]: return 2
        else: return 3

    q_labels = ['Q1(0-25%)', 'Q2(25-50%)', 'Q3(50-75%)', 'Q4(75-100%)']

    # 1-step transition
    trans_1 = np.zeros((4, 4))
    for tokens, H, _ in seqs:
        for t in range(1, len(H)):
            q_prev = to_q(H[t-1])
            q_curr = to_q(H[t])
            trans_1[q_prev, q_curr] += 1

    trans_1_prob = trans_1 / trans_1.sum(axis=1, keepdims=True).clip(1)

    print(f"\n  1步转移概率 P(H_t | H_{{t-1}}):")
    print(f"    {'':>12}", end='')
    for j in range(4):
        print(f"  {q_labels[j]:>12}", end='')
    print()
    for i in range(4):
        print(f"    {q_labels[i]:>12}", end='')
        for j in range(4):
            print(f"  {trans_1_prob[i,j]:>11.3f}", end='')
        print()

    # 3-step transition: P(H_t | H_{t-1}, H_{t-2}, H_{t-3})
    key_patterns = [
        (0, 0, 0),  # Q1,Q1,Q1
        (0, 0, 3),  # Q1,Q1,Q4
        (3, 3, 3),  # Q4,Q4,Q4
        (3, 3, 0),  # Q4,Q4,Q1
        (0, 3, 0),  # Q1,Q4,Q1
        (3, 0, 3),  # Q4,Q1,Q4
    ]

    print(f"\n  3步条件概率 P(H_t=Q4 | 前3步):")
    trans_3 = defaultdict(lambda: np.zeros(4))
    for tokens, H, _ in seqs:
        for t in range(3, len(H)):
            pattern = (to_q(H[t-3]), to_q(H[t-2]), to_q(H[t-1]))
            trans_3[pattern][to_q(H[t])] += 1

    for pattern in key_patterns:
        counts = trans_3[pattern]
        total = counts.sum()
        if total < 10:
            continue
        probs = counts / total
        p_name = ','.join([f'Q{p+1}' for p in pattern])
        print(f"    P(·|{p_name:>11}): Q1={probs[0]:.3f} Q2={probs[1]:.3f} "
              f"Q3={probs[2]:.3f} Q4={probs[3]:.3f}  (N={int(total)})")

    # Key comparison: P(Q4|Q1,Q1,Q1) vs P(Q4|Q4,Q4,Q4)
    c_low = trans_3[(0,0,0)]
    c_high = trans_3[(3,3,3)]
    if c_low.sum() > 0 and c_high.sum() > 0:
        p_q4_after_low = c_low[3] / c_low.sum()
        p_q4_after_high = c_high[3] / c_high.sum()
        print(f"\n  ★ P(Q4|Q1,Q1,Q1) = {p_q4_after_low:.4f}")
        print(f"  ★ P(Q4|Q4,Q4,Q4) = {p_q4_after_high:.4f}")
        print(f"  ★ 压制比 = {p_q4_after_low / max(p_q4_after_high, 1e-8):.4f} "
              f"(如果<1，说明低熵上下文确实压制高熵出现)")


# =====================================================================
# EXPERIMENT 3: Training-induced sharpening contamination
# =====================================================================
def experiment_3(seqs_early, seqs_late):
    print(f"\n{'='*90}")
    print(f"  实验 3: 训练前后的锐化污染证据")
    print(f"{'='*90}")

    def compute_profile(seqs):
        """Per-token-type entropy profile."""
        type_ents = defaultdict(list)
        for tokens, H, is_correct in seqs:
            n = len(H)
            for t in range(1, n):
                low_nbr_frac = np.mean(H[max(0,t-5):t] < np.percentile(H, 25))
                tok = tokens[t].strip()
                h = H[t]
                type_ents[tok].append((h, low_nbr_frac, is_correct))
        return type_ents

    # Compute global stats for both periods
    all_H_early = np.concatenate([s[1] for s in seqs_early])
    all_H_late = np.concatenate([s[1] for s in seqs_late])
    P75_early = np.percentile(all_H_early, 75)
    P75_late = np.percentile(all_H_late, 75)

    # For each sequence, measure how much high-H tokens sharpen based on low-H neighbor density
    bins_nbr = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]

    for period_label, seqs, P75 in [("早期", seqs_early, P75_early), ("后期", seqs_late, P75_late)]:
        bin_data = defaultdict(list)
        for tokens, H, is_correct in seqs:
            P25_local = np.percentile(H, 25)
            n = len(H)
            for t in range(5, n):
                if H[t] < np.percentile(H, 50):
                    continue
                low_nbr_frac = np.mean(H[t-5:t] < P25_local)
                bin_data[low_nbr_frac >= 0.6].append(H[t])

        high_nbr = np.array(bin_data[False]) if bin_data[False] else np.array([0])
        low_nbr = np.array(bin_data[True]) if bin_data[True] else np.array([0])
        print(f"\n  [{period_label}] 高熵token(H>P50)的熵值，按低熵邻居密度分组:")
        print(f"    低熵邻居多(≥60%): mean H={low_nbr.mean():.4f} (N={len(low_nbr)})")
        print(f"    低熵邻居少(<60%): mean H={high_nbr.mean():.4f} (N={len(high_nbr)})")
        print(f"    Δ = {low_nbr.mean() - high_nbr.mean():+.4f}")

    # Direct comparison: same entropy percentile across periods
    print(f"\n  全局熵分位数变化:")
    for p in [50, 75, 90, 95]:
        e_val = np.percentile(all_H_early, p)
        l_val = np.percentile(all_H_late, p)
        print(f"    P{p}: 早期={e_val:.4f} → 后期={l_val:.4f} (Δ={l_val-e_val:+.4f})")


# =====================================================================
# EXPERIMENT 4: Critical point identification
# =====================================================================
def experiment_4(seqs, label):
    print(f"\n{'='*90}")
    print(f"  实验 4: 关键探索点/决策峰/收敛点的精细识别 [{label}]")
    print(f"{'='*90}")

    all_H = np.concatenate([s[1] for s in seqs])
    P25, P50, P75 = np.percentile(all_H, [25, 50, 75])

    cp_data = {
        'correct': {'onset': [], 'peak': [], 'converge': [], 'execute': []},
        'wrong':   {'onset': [], 'peak': [], 'converge': [], 'execute': []},
    }
    cp_tokens = {
        'correct': {'onset': [], 'peak': [], 'converge': [], 'execute': []},
        'wrong':   {'onset': [], 'peak': [], 'converge': [], 'execute': []},
    }

    for tokens, H, is_correct in seqs:
        key = 'correct' if is_correct else 'wrong'
        n = len(H)
        dH = np.diff(H)

        for t in range(2, n - 2):
            relpos = t / n
            # Exploration onset: H crosses median upward
            if H[t] > P50 and H[t-1] <= P50 and dH[t-1] > 0:
                cp_data[key]['onset'].append((H[t], relpos))
                cp_tokens[key]['onset'].append(tokens[t])

            # Decision peak: local max with H > P75
            if H[t] > P75 and H[t] >= H[t-1] and H[t] >= H[t+1] and (H[t] > H[t-2] or H[t] > H[t+2]):
                cp_data[key]['peak'].append((H[t], relpos))
                cp_tokens[key]['peak'].append(tokens[t])

            # Convergence: H crosses median downward
            if H[t] < P50 and H[t-1] >= P50 and dH[t-1] < 0:
                cp_data[key]['converge'].append((H[t], relpos))
                cp_tokens[key]['converge'].append(tokens[t])

            # Execution: in a low-H run of 5+
            if t >= 4 and all(H[t-j] < P25 for j in range(5)):
                cp_data[key]['execute'].append((H[t], relpos))
                cp_tokens[key]['execute'].append(tokens[t])

    cp_names = {
        'onset': '探索起始点', 'peak': '决策峰',
        'converge': '收敛点', 'execute': '执行态'
    }

    for cp_type in ['onset', 'peak', 'converge', 'execute']:
        print(f"\n  --- {cp_names[cp_type]} ---")
        for key, key_label in [('correct', '正确'), ('wrong', '错误')]:
            data = cp_data[key][cp_type]
            toks = cp_tokens[key][cp_type]
            if not data:
                print(f"    {key_label}: 无数据")
                continue
            H_arr = np.array([d[0] for d in data])
            pos_arr = np.array([d[1] for d in data])
            n_per_seq = len(data) / max(sum(1 for s in seqs if s[2] == (key == 'correct')), 1)

            tok_counter = Counter(t.strip() for t in toks)
            top_5 = tok_counter.most_common(5)
            top_str = ', '.join(f'"{t}"({c})' for t, c in top_5)

            print(f"    {key_label}: N={len(data)}, 每序列≈{n_per_seq:.1f}, "
                  f"mean_H={H_arr.mean():.3f}, mean_pos={pos_arr.mean():.2%}")
            print(f"      top tokens: {top_str}")

    # Key comparison: correct vs wrong ratios
    print(f"\n  ★ 正确 vs 错误 关键点密度 (每序列):")
    n_correct = max(sum(1 for s in seqs if s[2]), 1)
    n_wrong = max(sum(1 for s in seqs if not s[2]), 1)
    for cp_type in ['onset', 'peak', 'converge', 'execute']:
        c_rate = len(cp_data['correct'][cp_type]) / n_correct
        w_rate = len(cp_data['wrong'][cp_type]) / n_wrong
        ratio = c_rate / max(w_rate, 1e-8)
        print(f"    {cp_names[cp_type]:<8}: 正确={c_rate:.1f}/seq, 错误={w_rate:.1f}/seq, "
              f"比值={ratio:.3f}")


# =====================================================================
# EXPERIMENT 5: Exploration-convergence rhythm
# =====================================================================
def experiment_5(seqs, label):
    print(f"\n{'='*90}")
    print(f"  实验 5: 探索-收敛节奏分析 [{label}]")
    print(f"{'='*90}")

    all_H = np.concatenate([s[1] for s in seqs])
    median_H = np.median(all_H)

    rhythm_data = {'correct': [], 'wrong': []}

    for tokens, H, is_correct in seqs:
        key = 'correct' if is_correct else 'wrong'
        above = H > median_H

        # Segment into runs
        explore_lens = []
        converge_lens = []
        run_len = 1
        for t in range(1, len(above)):
            if above[t] == above[t-1]:
                run_len += 1
            else:
                if above[t-1]:
                    explore_lens.append(run_len)
                else:
                    converge_lens.append(run_len)
                run_len = 1
        if above[-1]:
            explore_lens.append(run_len)
        else:
            converge_lens.append(run_len)

        n_segments = len(explore_lens) + len(converge_lens)
        avg_explore = np.mean(explore_lens) if explore_lens else 0
        avg_converge = np.mean(converge_lens) if converge_lens else 0
        explore_ratio = sum(explore_lens) / max(len(H), 1)

        # Rhythm regularity: CV of explore segment lengths
        if len(explore_lens) > 2:
            cv_explore = np.std(explore_lens) / max(np.mean(explore_lens), 1e-8)
        else:
            cv_explore = float('nan')

        if len(converge_lens) > 2:
            cv_converge = np.std(converge_lens) / max(np.mean(converge_lens), 1e-8)
        else:
            cv_converge = float('nan')

        rhythm_data[key].append({
            'n_segments': n_segments,
            'n_explore': len(explore_lens),
            'n_converge': len(converge_lens),
            'avg_explore_len': avg_explore,
            'avg_converge_len': avg_converge,
            'explore_ratio': explore_ratio,
            'cv_explore': cv_explore,
            'cv_converge': cv_converge,
            'seq_len': len(H),
        })

    for key, key_label in [('correct', '正确'), ('wrong', '错误')]:
        data = rhythm_data[key]
        if not data:
            continue
        print(f"\n  [{key_label}] (N={len(data)}):")
        for metric, metric_label in [
            ('n_segments', '总段数'),
            ('avg_explore_len', '平均探索段长'),
            ('avg_converge_len', '平均收敛段长'),
            ('explore_ratio', '探索时间占比'),
            ('cv_explore', '探索节奏规律性(CV,越低越规律)'),
            ('cv_converge', '收敛节奏规律性(CV)'),
        ]:
            vals = [d[metric] for d in data if not np.isnan(d[metric])]
            if vals:
                print(f"    {metric_label:<35}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Cohen's d for key metrics
    print(f"\n  ★ 正确 vs 错误 Cohen's d:")
    for metric, metric_label in [
        ('n_segments', '总段数'),
        ('avg_explore_len', '平均探索段长'),
        ('avg_converge_len', '平均收敛段长'),
        ('explore_ratio', '探索时间占比'),
        ('cv_explore', '探索节奏CV'),
    ]:
        c_vals = [d[metric] for d in rhythm_data['correct'] if not np.isnan(d[metric])]
        w_vals = [d[metric] for d in rhythm_data['wrong'] if not np.isnan(d[metric])]
        if c_vals and w_vals:
            cm, cs = np.mean(c_vals), np.std(c_vals)
            wm, ws = np.mean(w_vals), np.std(w_vals)
            d = (cm - wm) / max((cs + ws) / 2, 1e-8)
            print(f"    {metric_label:<25}: 正确={cm:.3f}, 错误={wm:.3f}, d={d:+.3f}")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("Loading data...")
    data_late = load_data(LATE_PATH)
    data_early = load_data(EARLY_PATH)
    seqs_late = extract_sequences(data_late)
    seqs_early = extract_sequences(data_early)
    print(f"  后期: {len(seqs_late)} sequences, 早期: {len(seqs_early)} sequences")

    experiment_1(seqs_late, "后期数据")
    experiment_1(seqs_early, "早期数据")

    experiment_2(seqs_late, "后期数据")
    experiment_2(seqs_early, "早期数据")

    experiment_3(seqs_early, seqs_late)

    experiment_4(seqs_late, "后期数据")

    experiment_5(seqs_late, "后期数据")
    experiment_5(seqs_early, "早期数据")
