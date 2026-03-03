"""
分析 ε>0 均值以上 和 ε<0 均值以下 的 token 具体是什么内容
"""
import json
import numpy as np
from collections import Counter, defaultdict

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_epsilon(H_arr):
    n = len(H_arr)
    if n < 10:
        return None, None, None
    rho = np.corrcoef(H_arr[:-1], H_arr[1:])[0, 1]
    if np.isnan(rho): rho = 0.0
    rho = max(0.05, min(0.95, rho))
    alpha = 1.0 - rho
    ema = np.zeros(n)
    epsilon = np.zeros(n)
    ema[0] = H_arr[0]
    for t in range(1, n):
        epsilon[t] = H_arr[t] - ema[t - 1]
        ema[t] = alpha * H_arr[t] + (1 - alpha) * ema[t - 1]
    pos_eps = epsilon[epsilon > 0]
    if len(pos_eps) < 3:
        return None, None, None
    sigma_pos = np.std(pos_eps)
    if sigma_pos < 1e-6:
        return None, None, None
    return epsilon, H_arr, sigma_pos

def classify_token(tok):
    tok_stripped = tok.strip()
    if not tok_stripped:
        return "空白/换行"
    if tok_stripped in ['\\n', '\n', '\\n\\n']:
        return "换行符"
    if tok_stripped in ['+', '-', '*', '/', '=', '<', '>', '\\leq', '\\geq', '\\neq',
                        '\\times', '\\div', '\\pm', '\\cdot', '\\equiv']:
        return "数学运算符"
    if tok_stripped in ['(', ')', '[', ']', '{', '}', '\\{', '\\}', '\\left', '\\right']:
        return "括号/分隔"
    if tok_stripped in [',', '.', ':', ';', '!', '?', '，', '。', '：', '；']:
        return "标点"
    if tok_stripped.replace('.','',1).replace('-','',1).isdigit():
        return "数字"
    if tok_stripped in ['\\frac', '\\sqrt', '\\sum', '\\prod', '\\int', '\\lim',
                        '\\log', '\\ln', '\\sin', '\\cos', '\\tan', '\\binom',
                        '\\pmod', '\\mod', '\\gcd']:
        return "数学函数"
    if tok_stripped.startswith('\\') and len(tok_stripped) > 1:
        return "LaTeX命令"
    if tok_stripped in ['$', '$$', '\\[', '\\]', '\\(', '\\)']:
        return "数学环境"
    keywords = ['Step', 'step', 'Therefore', 'therefore', 'Thus', 'thus',
                'Since', 'since', 'Because', 'because', 'Hence', 'hence',
                'So', 'so', 'Let', 'let', 'We', 'we', 'Now', 'now',
                'First', 'Next', 'Then', 'then', 'Finally', 'finally',
                'Note', 'note', 'Given', 'given', 'Consider', 'consider',
                'If', 'if', 'where', 'Where', 'which', 'that', 'with']
    if tok_stripped in keywords:
        return "推理关键词"
    if tok_stripped in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                        'to', 'of', 'in', 'for', 'on', 'at', 'by', 'and',
                        'or', 'not', 'this', 'it', 'can', 'has', 'have',
                        'The', 'A', 'An', 'Is']:
        return "英语虚词"
    if tok_stripped in ['answer', 'Answer', 'boxed', 'final', 'Final',
                        'solution', 'Solution', 'result', 'Result']:
        return "答案关键词"
    if all(c.isalpha() for c in tok_stripped):
        return "英语实词"
    if any(ord(c) > 0x4E00 for c in tok_stripped):
        return "中文"
    return "其他"

def analyze(data, label):
    print(f"\n{'='*100}")
    print(f"  {label}  (pass@1={data['overall_pass@1']:.4f})")
    print(f"{'='*100}")

    # collect tokens for correct/incorrect, high-ε-pos / high-ε-neg
    groups = {
        'correct': {'eps_high_pos': [], 'eps_high_neg': [], 'all_tokens': []},
        'wrong':   {'eps_high_pos': [], 'eps_high_neg': [], 'all_tokens': []},
    }
    # also track position info
    pos_info = {
        'correct': {'eps_high_pos_relpos': [], 'eps_high_neg_relpos': []},
        'wrong':   {'eps_high_pos_relpos': [], 'eps_high_neg_relpos': []},
    }

    for problem in data['results']:
        for resp in problem['responses']:
            ea = resp.get('entropy_analysis')
            if not ea or 'steps' not in ea:
                continue
            all_tokens = []
            all_ents = []
            for step in ea['steps']:
                toks = step.get('tokens', [])
                ents = step.get('token_entropies', [])
                for tok, ent in zip(toks, ents):
                    all_tokens.append(tok)
                    all_ents.append(ent)
            if len(all_tokens) < 20:
                continue

            H = np.array(all_ents)
            result = compute_epsilon(H)
            if result[0] is None:
                continue
            epsilon, _, sigma_pos = result

            key = 'correct' if resp['is_correct'] else 'wrong'

            pos_eps_vals = epsilon[epsilon > 0]
            neg_eps_vals = epsilon[epsilon < 0]
            pos_mean = pos_eps_vals.mean() if len(pos_eps_vals) > 0 else 0
            neg_mean = neg_eps_vals.mean() if len(neg_eps_vals) > 0 else 0

            n_total = len(all_tokens)
            for t in range(1, len(all_tokens)):
                if t < len(epsilon):
                    e = epsilon[t]
                    tok = all_tokens[t]
                    h = H[t]
                    relpos = t / n_total

                    if e > 0 and e > pos_mean:
                        groups[key]['eps_high_pos'].append((tok, e, h, relpos))
                        pos_info[key]['eps_high_pos_relpos'].append(relpos)
                    elif e < 0 and e < neg_mean:
                        groups[key]['eps_high_neg'].append((tok, e, h, relpos))
                        pos_info[key]['eps_high_neg_relpos'].append(relpos)
                    groups[key]['all_tokens'].append((tok, e, h))

    for key in ['correct', 'wrong']:
        g = groups[key]
        label_str = "正确" if key == 'correct' else "错误"
        print(f"\n{'─'*50}")
        print(f"  [{label_str}样本] ε>0 均值以上的 token: {len(g['eps_high_pos'])} 个")
        print(f"  [{label_str}样本] ε<0 均值以下的 token: {len(g['eps_high_neg'])} 个")
        print(f"{'─'*50}")

        for subset_key, subset_label in [('eps_high_pos', 'ε > pos_mean (熵大幅上升)'),
                                          ('eps_high_neg', 'ε < neg_mean (熵大幅下降)')]:
            items = g[subset_key]
            if not items:
                continue
            print(f"\n  --- {subset_label} ---")

            # token category distribution
            cat_counter = Counter()
            for tok, e, h, rp in items:
                cat = classify_token(tok)
                cat_counter[cat] += 1
            total = len(items)
            print(f"  Token 类别分布 (共 {total}):")
            for cat, cnt in cat_counter.most_common(15):
                print(f"    {cat:<20} {cnt:>6} ({cnt/total:>5.1%})")

            # top tokens by frequency
            tok_counter = Counter()
            for tok, e, h, rp in items:
                tok_counter[tok.strip()] += 1
            print(f"\n  最常见 token (top 30):")
            for tok, cnt in tok_counter.most_common(30):
                display = repr(tok) if len(tok) > 15 or not tok.strip() else tok
                print(f"    {display:<25} {cnt:>5} ({cnt/total:>5.1%})")

            # top tokens by ε magnitude
            if subset_key == 'eps_high_pos':
                sorted_items = sorted(items, key=lambda x: -x[1])
            else:
                sorted_items = sorted(items, key=lambda x: x[1])
            print(f"\n  ε 最极端的 token (top 20):")
            seen = set()
            count = 0
            for tok, e, h, rp in sorted_items:
                tok_s = tok.strip()
                if tok_s in seen:
                    continue
                seen.add(tok_s)
                display = repr(tok_s) if len(tok_s) > 15 else tok_s
                print(f"    {display:<25} ε={e:>+.4f}  H={h:.4f}  pos={rp:.2%}")
                count += 1
                if count >= 20:
                    break

            # position distribution
            relpos_arr = np.array(pos_info[key][f'{subset_key}_relpos'])
            if len(relpos_arr) > 0:
                print(f"\n  在序列中的相对位置分布:")
                bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
                for lo, hi in bins:
                    n = np.sum((relpos_arr >= lo) & (relpos_arr < hi))
                    bar = '█' * int(n / len(relpos_arr) * 50)
                    print(f"    [{lo:.1f}-{hi:.1f})  {n:>6} ({n/len(relpos_arr):>5.1%})  {bar}")

    # cross comparison
    print(f"\n{'─'*50}")
    print(f"  正确 vs 错误 对比")
    print(f"{'─'*50}")
    for subset_key, subset_label in [('eps_high_pos', 'ε>pos_mean'),
                                      ('eps_high_neg', 'ε<neg_mean')]:
        c_cats = Counter()
        w_cats = Counter()
        for tok, e, h, rp in groups['correct'][subset_key]:
            c_cats[classify_token(tok)] += 1
        for tok, e, h, rp in groups['wrong'][subset_key]:
            w_cats[classify_token(tok)] += 1
        c_total = max(sum(c_cats.values()), 1)
        w_total = max(sum(w_cats.values()), 1)
        all_cats = set(c_cats.keys()) | set(w_cats.keys())
        print(f"\n  [{subset_label}] 类别占比对比:")
        print(f"    {'类别':<20} {'正确':>8} {'错误':>8} {'Δ':>8}")
        rows = []
        for cat in all_cats:
            c_pct = c_cats[cat] / c_total
            w_pct = w_cats[cat] / w_total
            rows.append((cat, c_pct, w_pct, c_pct - w_pct))
        rows.sort(key=lambda x: -abs(x[3]))
        for cat, cp, wp, d in rows[:12]:
            print(f"    {cat:<20} {cp:>7.1%} {wp:>7.1%} {d:>+7.1%}")


data_later = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json")

analyze(data_later, "后期数据")
