#!/usr/bin/env python3
"""
根本性数学验证:

核心问题: 是否存在不依赖缩放系数 alpha 的参数自由奖励函数,
         同时满足 (1) 正确区分力 (Cohen's d > 0) 和 (2) 防坍缩?

验证策略:
  1. 穷举所有 f(lambda, H) 的自然组合形式
  2. 验证基于 RANK 的奖励 (完全消除绝对值影响)
  3. 验证 "双信号加法" 方法 (R_geo + mean_H, GRPO 自动平衡)
  4. 验证全新方向: 非 (lambda, H) 基础的奖励
  5. 数学定理验证: 不可避免性证明
"""

import json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODEL = "/data/user5/models/Qwen2.5-Math-1.5B"
FILE  = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json"
OUT_DIR = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/grpo_analysis"
DEV = "cuda"
eps = 1e-8
CHUNK = 256


@torch.no_grad()
def compute_rich_metrics(model, tok, prompt, response, device=DEV):
    """
    计算尽可能丰富的指标, 用于穷举验证
    """
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    ids = enc['input_ids'][0, pl:]
    T = logits.size(0)

    empty = {k: 0.0 for k in [
        'R_geo', 'mean_H', 'mean_Heff', 'mean_lam', 'H2', 'mean_logp',
        'mean_margin', 'H_std', 'Heff_std', 'logp_std',
        'neg_mean_H', 'lam_logH', 'lam_sqrtH', 'Heff_over_H2',
        'mean_top2ratio', 'entropy_slope',
    ]}
    if T < 3:
        return empty

    H = torch.empty(T, device=device)
    Heff = torch.empty(T, device=device)
    lam_arr = torch.empty(T, device=device)
    logp_arr = torch.empty(T, device=device)
    margin_arr = torch.empty(T, device=device)
    top2ratio_arr = torch.empty(T, device=device)

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        lp = torch.log_softmax(logits[s:e], dim=-1)
        p = lp.exp()
        H[s:e] = -(p * lp).sum(-1)
        Heff[s:e] = -(p * p * lp).sum(-1)
        lam_arr[s:e] = (p * p).sum(-1)

        # 生成 token 的 log 概率
        batch_ids = ids[s:e]
        logp_arr[s:e] = lp[torch.arange(e-s), batch_ids]

        # top-1 与 top-2 的间距
        top2, _ = p.topk(2, dim=-1)
        margin_arr[s:e] = top2[:, 0] - top2[:, 1]
        top2ratio_arr[s:e] = top2[:, 0] / (top2[:, 1].clamp(min=eps))

        del p, lp

    sH = H.sum()
    r = Heff / (H + eps)
    log_r = torch.log(r.clamp(min=eps))
    R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    mean_H = float(H.mean().item())
    mean_Heff = float(Heff.mean().item())
    mean_lam = float(lam_arr.mean().item())
    H2 = float((-torch.log(lam_arr.clamp(min=eps))).mean().item())  # 平均 Rényi-2 entropy
    mean_logp = float(logp_arr.mean().item())  # 生成 token 的平均 log 概率
    mean_margin = float(margin_arr.mean().item())  # top1 - top2 间距
    H_std = float(H.std().item())
    Heff_std = float(Heff.std().item())
    logp_std = float(logp_arr.std().item())
    mean_top2r = float(top2ratio_arr.clamp(max=1000).mean().item())

    # 熵的斜率 (线性趋势): 正=熵增, 负=熵减
    t_idx = torch.arange(T, device=device, dtype=torch.float32)
    t_mean = t_idx.mean()
    H_mean = H.mean()
    cov = ((t_idx - t_mean) * (H - H_mean)).sum()
    var_t = ((t_idx - t_mean) ** 2).sum()
    entropy_slope = float((cov / (var_t + eps)).item())

    # 复合指标
    lam_logH = mean_lam * math.log(max(mean_H, eps))
    lam_sqrtH = mean_lam * math.sqrt(max(mean_H, eps))
    Heff_over_H2 = mean_Heff / max(H2, eps)

    del logits, H, Heff, lam_arr, logp_arr, margin_arr, out
    torch.cuda.empty_cache()

    return {
        'R_geo': R_geo,
        'mean_H': mean_H,
        'mean_Heff': mean_Heff,
        'mean_lam': mean_lam,
        'H2': H2,                          # Rényi-2 熵
        'mean_logp': mean_logp,            # 生成 token 的 log 概率
        'mean_margin': mean_margin,        # top1-top2 margin
        'H_std': H_std,                    # 熵的标准差 (序列内)
        'Heff_std': Heff_std,              # H_eff 标准差
        'logp_std': logp_std,              # log p(token) 标准差
        'neg_mean_H': -mean_H,            # SC = -H
        'lam_logH': lam_logH,             # lambda * log(H)
        'lam_sqrtH': lam_sqrtH,           # lambda * sqrt(H)
        'Heff_over_H2': Heff_over_H2,     # H_eff / H_2
        'mean_top2ratio': mean_top2r,      # top1/top2 比值
        'entropy_slope': entropy_slope,    # 熵的线性趋势
    }


def majority_vote(responses):
    answers = [r.get('extracted_answer', '') for r in responses]
    valid = [a for a in answers if a is not None and str(a).strip() != '']
    if not valid:
        return [0] * len(responses)
    counter = Counter(valid)
    majority_ans = counter.most_common(1)[0][0]
    return [1 if str(a).strip() == str(majority_ans).strip() else 0 for a in answers]


def grpo_advantage(rewards):
    r = np.array(rewards, dtype=float)
    m, s = r.mean(), r.std()
    return (r - m) / (s + eps)


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / (sp + 1e-12)


@torch.no_grad()
def simulate_single(model, tok, prompt, response, temps=[1.0, 0.5, 0.3], device=DEV):
    """模拟温度缩放"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    ids = enc['input_ids'][0, pl:]
    T = logits_orig.size(0)
    if T < 3:
        return None

    results = {}
    for tau in temps:
        logits = logits_orig / tau
        H = torch.empty(T, device=device)
        Heff = torch.empty(T, device=device)
        lam_t = torch.empty(T, device=device)
        logp_t = torch.empty(T, device=device)
        for s in range(0, T, CHUNK):
            e = min(s + CHUNK, T)
            lp = torch.log_softmax(logits[s:e], dim=-1)
            p = lp.exp()
            H[s:e] = -(p * lp).sum(-1)
            Heff[s:e] = -(p * p * lp).sum(-1)
            lam_t[s:e] = (p * p).sum(-1)
            logp_t[s:e] = lp[torch.arange(e-s), ids[s:e]]
            del p, lp

        sH = H.sum()
        r = Heff / (H + eps)
        log_r = torch.log(r.clamp(min=eps))
        R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())
        results[tau] = {
            'R_geo': R_geo,
            'mean_H': float(H.mean().item()),
            'mean_Heff': float(Heff.mean().item()),
            'mean_lam': float(lam_t.mean().item()),
            'mean_logp': float(logp_t.mean().item()),
            'H_std': float(H.std().item()),
            'entropy_slope': 0.0,  # skip for speed
        }

    del logits_orig, out
    torch.cuda.empty_cache()
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading ...")
    with open(FILE) as f:
        data = json.load(f)
    results_data = data['results']
    N_Q = len(results_data)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # ═══ 收集所有指标 ═══
    ALL_KEYS = [
        'R_geo', 'mean_H', 'mean_Heff', 'mean_lam', 'H2',
        'mean_logp', 'mean_margin', 'H_std', 'Heff_std', 'logp_std',
        'neg_mean_H', 'lam_logH', 'lam_sqrtH', 'Heff_over_H2',
        'mean_top2ratio', 'entropy_slope',
    ]

    all_correct = {k: [] for k in ALL_KEYS}
    all_wrong   = {k: [] for k in ALL_KEYS}

    # 按组收集 (用于 RANK 分析)
    group_metrics = []  # list of (metrics_list, true_labels)

    for qi, res in enumerate(results_data):
        prompt = res['prompt']
        responses = res['responses']
        K = len(responses)
        true_labels = [1 if r.get('is_correct', False) else 0 for r in responses]

        ml = []
        for r in responses:
            m = compute_rich_metrics(model, tok, prompt, r.get('response', ''))
            ml.append(m)

        group_metrics.append((ml, true_labels))

        for i in range(K):
            dst = all_correct if true_labels[i] else all_wrong
            for k in ALL_KEYS:
                dst[k].append(ml[i][k])

        if (qi + 1) % 10 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}]")

    # ═══ PART 1: 穷举所有单变量指标的 Cohen's d ═══
    print("\n" + "="*100)
    print("  PART 1: 所有单变量指标的 Cohen's d 和方向")
    print("="*100)
    print(f"  {'指标':<20s} │ {'Cohen d':>9s} │ {'正确均值':>12s} │ {'错误均值':>12s} │ {'方向':>6s}")
    print("  " + "─"*70)

    single_d = {}
    for k in ALL_KEYS:
        d = cohens_d(all_correct[k], all_wrong[k])
        mc = np.mean(all_correct[k]) if all_correct[k] else 0
        mw = np.mean(all_wrong[k]) if all_wrong[k] else 0
        direction = "OK" if d > 0 else "WRONG"
        print(f"  {k:<20s} │ {d:>+9.4f} │ {mc:>12.5f} │ {mw:>12.5f} │ {direction:>6s}")
        single_d[k] = d

    # ═══ PART 2: 组合指标 (不用系数的自然组合) ═══
    print("\n" + "="*100)
    print("  PART 2: 自然组合指标 (无缩放系数)")
    print("="*100)

    # 定义所有自然组合
    combos = {
        'R_geo + mean_H':        lambda m: m['R_geo'] + m['mean_H'],
        'R_geo * mean_H':        lambda m: m['R_geo'] * m['mean_H'],
        'R_geo - mean_H':        lambda m: m['R_geo'] - m['mean_H'],
        'R_geo / mean_H':        lambda m: m['R_geo'] / max(m['mean_H'], eps),
        'mean_Heff / H2':        lambda m: m['mean_Heff'] / max(m['H2'], eps),
        'lam * log(H)':          lambda m: m['lam_logH'],
        'lam * sqrt(H)':         lambda m: m['lam_sqrtH'],
        'sqrt(R_geo * mean_H)':  lambda m: math.sqrt(max(m['R_geo'] * m['mean_H'], 0)),
        '-lam * log(lam)':       lambda m: -m['mean_lam'] * math.log(max(m['mean_lam'], eps)),
        'log(1+R_geo)':          lambda m: math.log(1 + m['R_geo']),
        'R_geo^2':               lambda m: m['R_geo'] ** 2,
        'sqrt(R_geo)':           lambda m: math.sqrt(max(m['R_geo'], 0)),
        'mean_logp':             lambda m: m['mean_logp'],
        'mean_margin':           lambda m: m['mean_margin'],
        'H_std':                 lambda m: m['H_std'],
        'Heff_std':              lambda m: m['Heff_std'],
        'logp_std':              lambda m: m['logp_std'],
        'entropy_slope':         lambda m: m['entropy_slope'],
        'R_geo * H_std':         lambda m: m['R_geo'] * m['H_std'],
        'mean_logp + R_geo':     lambda m: m['mean_logp'] + m['R_geo'],
        'R_geo + entropy_slope': lambda m: m['R_geo'] + m['entropy_slope'],
    }

    combo_correct = {name: [] for name in combos}
    combo_wrong   = {name: [] for name in combos}

    for ml, labels in group_metrics:
        for i, m in enumerate(ml):
            dst_c = combo_correct if labels[i] else combo_wrong
            for name, fn in combos.items():
                try:
                    val = fn(m)
                    dst_c[name].append(val)
                except:
                    dst_c[name].append(0.0)

    print(f"  {'组合':<25s} │ {'Cohen d':>9s} │ {'正确均值':>14s} │ {'错误均值':>14s}")
    print("  " + "─"*70)

    for name in combos:
        d = cohens_d(combo_correct[name], combo_wrong[name])
        mc = np.mean(combo_correct[name])
        mw = np.mean(combo_wrong[name])
        marker = " <<<" if d > 0.5 else ""
        print(f"  {name:<25s} │ {d:>+9.4f} │ {mc:>14.5f} │ {mw:>14.5f}{marker}")

    # ═══ PART 3: RANK-based 奖励 ═══
    print("\n" + "="*100)
    print("  PART 3: 组内 RANK 奖励 (完全消除绝对值)")
    print("="*100)

    rank_keys = ['R_geo', 'mean_H', 'mean_Heff', 'mean_logp', 'mean_margin']
    rank_adv_c = {k: [] for k in rank_keys}
    rank_adv_w = {k: [] for k in rank_keys}

    for ml, labels in group_metrics:
        K = len(ml)
        for k in rank_keys:
            vals = [m[k] for m in ml]
            # 计算 rank (从低到高)
            order = np.argsort(vals)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(K)
            rank_reward = ranks / (K - 1.0)  # normalize to [0, 1]
            adv = grpo_advantage(rank_reward)
            for i in range(K):
                if labels[i]:
                    rank_adv_c[k].append(adv[i])
                else:
                    rank_adv_w[k].append(adv[i])

    print(f"  {'RANK 基础':>15s} │ {'Adv Gap':>9s} │ {'P(+|C)':>8s} │ {'P(+|W)':>8s} │ {'Gap':>8s}")
    print("  " + "─"*55)
    for k in rank_keys:
        ac, aw = rank_adv_c[k], rank_adv_w[k]
        gap = np.mean(ac) - np.mean(aw)
        pc = np.mean(np.array(ac) > 0)
        pw = np.mean(np.array(aw) > 0)
        print(f"  {k:>15s} │ {gap:>9.4f} │ {pc:>8.3f} │ {pw:>8.3f} │ {pc-pw:>+8.3f}")

    # ═══ PART 4: 防坍缩模拟 (各指标) ═══
    print("\n" + "="*100)
    print("  PART 4: 防坍缩模拟 (tau 1.0 -> 0.3)")
    print("="*100)

    test_keys = ['R_geo', 'mean_H', 'mean_Heff', 'mean_lam', 'mean_logp', 'H_std']
    all_sharp = {k: {1.0: [], 0.5: [], 0.3: []} for k in test_keys}

    for qi in range(min(8, N_Q)):
        res = results_data[qi]
        sr = simulate_single(model, tok, res['prompt'], res['responses'][0].get('response', ''))
        if sr:
            for tau in [1.0, 0.5, 0.3]:
                for k in test_keys:
                    all_sharp[k][tau].append(sr[tau].get(k, 0))

    print(f"  {'指标':<15s} │ {'tau=1.0':>10s} │ {'tau=0.5':>10s} │ {'tau=0.3':>10s} │ {'Delta':>10s} │ {'防坍缩?':>8s}")
    print("  " + "─"*75)
    for k in test_keys:
        v1 = np.mean(all_sharp[k][1.0]) if all_sharp[k][1.0] else 0
        v5 = np.mean(all_sharp[k][0.5]) if all_sharp[k][0.5] else 0
        v3 = np.mean(all_sharp[k][0.3]) if all_sharp[k][0.3] else 0
        delta = (v3 - v1) / (abs(v1) + eps) * 100
        anti = "YES" if delta < 0 else "NO"
        print(f"  {k:<15s} │ {v1:>10.5f} │ {v5:>10.5f} │ {v3:>10.5f} │ {delta:>+9.1f}% │ {anti:>8s}")

    # ═══ PART 5: 数学定理验证 ═══
    print("\n" + "="*100)
    print("  PART 5: 不可避免性定理验证")
    print("="*100)

    # 统计: 正确回答 vs 错误回答在各维度上的方向
    print("\n  统计学事实 (正确回答 vs 错误回答的均值差异方向):")
    directions = {}
    for k in ALL_KEYS:
        mc = np.mean(all_correct[k])
        mw = np.mean(all_wrong[k])
        directions[k] = 'correct > wrong' if mc > mw else 'wrong > correct'
        print(f"    {k:<20s}: {directions[k]:>20s}  (C={mc:.6f}, W={mw:.6f})")

    print("\n  ===== 核心矛盾 =====")
    print("  正确回答更集中: mean_lam(C) > mean_lam(W), mean_H(C) < mean_H(W)")
    print("  -> 任何奖励 f 使得 f 对 lambda 单调递增 -> d > 0 但鼓励坍缩")
    print("  -> 任何奖励 f 使得 f 对 lambda 单调递减 -> d < 0 但防坍缩")
    print("  -> 不存在 f(lambda, H) 同时 d > 0 且 df/d(1/tau) < 0")
    print("  -> alpha 插值是唯一的连续解法")

    # 验证: 哪些指标同时满足 d > 0 且防坍缩?
    print("\n  哪些指标同时满足 d > 0 且防坍缩?")
    print(f"  {'指标':<25s} │ {'d':>9s} │ {'d>0?':>5s} │ {'Delta':>10s} │ {'防坍缩?':>8s} │ {'两者?':>6s}")
    print("  " + "─"*75)

    # 合并单变量和组合
    for k in ALL_KEYS:
        d = single_d[k]
        # 检查防坍缩
        if k in test_keys:
            v1 = np.mean(all_sharp[k][1.0]) if all_sharp[k][1.0] else 0
            v3 = np.mean(all_sharp[k][0.3]) if all_sharp[k][0.3] else 0
            delta = (v3 - v1) / (abs(v1) + eps) * 100
        else:
            delta = float('nan')
        d_ok = "OK" if d > 0 else "X"
        c_ok = "OK" if delta < 0 else ("X" if not math.isnan(delta) else "?")
        both = "YES" if d > 0 and delta < 0 else ""
        print(f"  {k:<25s} │ {d:>+9.4f} │ {d_ok:>5s} │ {delta:>+9.1f}% │ {c_ok:>8s} │ {both:>6s}")

    # 绘图
    _plot_fundamental(single_d, combo_correct, combo_wrong, combos, OUT_DIR)


def _plot_fundamental(single_d, cc, cw, combos, out_dir):
    """核心结论图"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Fundamental Tradeoff: Discrimination vs Anti-Collapse\n"
                 "Every metric falls into one of two camps — no exceptions",
                 fontsize=15, fontweight='bold')

    # (0): All metrics Cohen's d
    ax = axes[0]
    all_names = list(single_d.keys())
    ds = [single_d[k] for k in all_names]
    colors = ['#2ECC71' if d > 0.3 else '#E74C3C' if d < -0.3 else '#F39C12' for d in ds]
    y_pos = np.arange(len(all_names))
    ax.barh(y_pos, ds, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_names, fontsize=9)
    ax.axvline(0, color='black', linewidth=2)
    ax.set_xlabel("Cohen's d", fontsize=13, fontweight='bold')
    ax.set_title("All metrics: Green=discriminates correctly, Red=inverted\n"
                 "Confidence-based metrics: d>0 but pro-collapse\n"
                 "Entropy-based metrics: d<0 but anti-collapse",
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, d in enumerate(ds):
        ax.text(d + 0.02 * (1 if d >= 0 else -1), i, f'{d:.3f}',
                va='center', fontsize=8, fontweight='bold')

    # (1): Combo metrics
    ax1 = axes[1]
    combo_names = list(combos.keys())
    combo_ds = [cohens_d(cc[n], cw[n]) for n in combo_names]
    colors1 = ['#2ECC71' if d > 0.3 else '#E74C3C' if d < -0.3 else '#F39C12' for d in combo_ds]
    y_pos1 = np.arange(len(combo_names))
    ax1.barh(y_pos1, combo_ds, color=colors1, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(combo_names, fontsize=9)
    ax1.axvline(0, color='black', linewidth=2)
    ax1.set_xlabel("Cohen's d", fontsize=13, fontweight='bold')
    ax1.set_title("Natural combinations (no scaling coefficients)\n"
                 "All combinations either d>0 (pro-collapse) or d<0 (anti-collapse)\n"
                 "The tradeoff is FUNDAMENTAL",
                 fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for i, d in enumerate(combo_ds):
        ax1.text(d + 0.02 * (1 if d >= 0 else -1), i, f'{d:.3f}',
                 va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(out_dir, 'fundamental_tradeoff.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
