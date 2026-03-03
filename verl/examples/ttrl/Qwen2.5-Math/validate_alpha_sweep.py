#!/usr/bin/env python3
"""
Alpha 插值扫描: R_alpha = geo_mean_H( H_eff / H^alpha )

alpha=1: R_geo (d=+1.23, 鼓励坍缩)
alpha=0: geo_mean(H_eff) (d~-0.62, 强力防坍缩)

寻找同时满足:
  1. Cohen's d > 0 (学习方向正确)
  2. 锐化下降 (防坍缩)
的最优 alpha*
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
plt.rcParams['font.family'] = 'sans-serif'

MODEL = "/data/user5/models/Qwen2.5-Math-1.5B"
FILE  = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json"
OUT_DIR = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/grpo_analysis"
DEV = "cuda"
eps = 1e-8
CHUNK = 256

# Alpha 值域扫描
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@torch.no_grad()
def compute_alpha_metrics(model, tok, prompt, response, alphas=ALPHAS, device=DEV):
    """计算所有 alpha 值的 R_alpha = geo_mean_H(H_eff / H^alpha)"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T = logits.size(0)

    empty = {f'R_alpha_{a:.1f}': 0.0 for a in alphas}
    empty.update({'mean_H': 0.0, 'mean_lam': 0.0})
    if T < 3:
        return empty

    H = torch.empty(T, device=device)
    Heff = torch.empty(T, device=device)
    sum_lam = 0.0
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        lp = torch.log_softmax(logits[s:e], dim=-1)
        p = lp.exp()
        H[s:e] = -(p * lp).sum(-1)
        Heff[s:e] = -(p * p * lp).sum(-1)
        lam = (p * p).sum(-1)
        sum_lam += float(lam.sum().item())
        del p, lp, lam

    sH = H.sum()
    if sH < eps:
        return empty

    results = {}
    log_Heff = torch.log(Heff.clamp(min=eps))
    log_H = torch.log(H.clamp(min=eps))

    for alpha in alphas:
        # R_alpha = geo_mean_H( H_eff / H^alpha )
        # log(R_alpha) = sum(H * (log(H_eff) - alpha * log(H))) / sum(H)
        log_ratio = log_Heff - alpha * log_H
        R_alpha = float(torch.exp((H * log_ratio).sum() / (sH + eps)).item())
        results[f'R_alpha_{alpha:.1f}'] = R_alpha

    results['mean_H'] = float(H.mean().item())
    results['mean_lam'] = sum_lam / max(T, 1)

    del logits, H, Heff, out
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def simulate_alpha_sharpening(model, tok, prompt, response, alphas=ALPHAS,
                               temps=[1.0, 0.7, 0.5, 0.3], device=DEV):
    """各 alpha 在不同温度下的变化"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T = logits_orig.size(0)
    if T < 3:
        return None

    results = {}  # alpha -> [R_at_tau1, R_at_tau2, ...]
    for alpha in alphas:
        results[alpha] = []

    for tau in temps:
        logits = logits_orig / tau
        H = torch.empty(T, device=device)
        Heff = torch.empty(T, device=device)
        for s in range(0, T, CHUNK):
            e = min(s + CHUNK, T)
            lp = torch.log_softmax(logits[s:e], dim=-1)
            p = lp.exp()
            H[s:e] = -(p * lp).sum(-1)
            Heff[s:e] = -(p * p * lp).sum(-1)
            del p, lp

        sH = H.sum()
        log_Heff = torch.log(Heff.clamp(min=eps))
        log_H = torch.log(H.clamp(min=eps))

        for alpha in alphas:
            log_ratio = log_Heff - alpha * log_H
            R = float(torch.exp((H * log_ratio).sum() / (sH + eps)).item())
            results[alpha].append(R)

    del logits_orig, out
    torch.cuda.empty_cache()
    return {'temps': temps, 'results': results}


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data ...")
    with open(FILE) as f:
        data = json.load(f)
    results_data = data['results']
    N_Q = len(results_data)
    print(f"  {N_Q} questions, {results_data[0]['n_samples']} responses/question")

    print("Loading model ...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # ── 收集每个 alpha 的 correct/wrong 分布 ──
    alpha_keys = [f'R_alpha_{a:.1f}' for a in ALPHAS]
    all_correct = {k: [] for k in alpha_keys}
    all_wrong   = {k: [] for k in alpha_keys}
    all_adv_c = {k: [] for k in alpha_keys}
    all_adv_w = {k: [] for k in alpha_keys}

    for qi, res in enumerate(results_data):
        prompt = res['prompt']
        responses = res['responses']
        K = len(responses)
        true_labels = [1 if r.get('is_correct', False) else 0 for r in responses]

        metrics_list = []
        for ri, r in enumerate(responses):
            m = compute_alpha_metrics(model, tok, prompt, r.get('response', ''))
            metrics_list.append(m)

        for ak in alpha_keys:
            vals = [m[ak] for m in metrics_list]
            adv = grpo_advantage(vals)
            for i in range(K):
                if true_labels[i]:
                    all_correct[ak].append(vals[i])
                    all_adv_c[ak].append(adv[i])
                else:
                    all_wrong[ak].append(vals[i])
                    all_adv_w[ak].append(adv[i])

        if (qi + 1) % 5 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}]  correct={sum(true_labels)}/{K}")

    # ── 统计表 ──
    print("\n" + "="*100)
    print("  Alpha Sweep: R_alpha = geo_mean_H( H_eff / H^alpha )")
    print("="*100)
    print(f"  {'alpha':<7s} │ {'Cohen d':>9s} │ {'Adv Gap':>9s} │ {'P(+|correct)':>12s} │ {'P(+|wrong)':>12s} │ "
          f"{'Mean(correct)':>14s} │ {'Mean(wrong)':>14s}")
    print("  " + "─"*90)

    summary = {}
    for alpha in ALPHAS:
        ak = f'R_alpha_{alpha:.1f}'
        rc, rw = all_correct[ak], all_wrong[ak]
        ac, aw = all_adv_c[ak], all_adv_w[ak]
        d = cohens_d(rc, rw)
        gap = np.mean(ac) - np.mean(aw) if ac and aw else 0
        p_c = np.mean(np.array(ac) > 0) if ac else 0
        p_w = np.mean(np.array(aw) > 0) if aw else 0
        mc = np.mean(rc) if rc else 0
        mw = np.mean(rw) if rw else 0
        print(f"  {alpha:<7.1f} │ {d:>+9.4f} │ {gap:>9.4f} │ {p_c:>12.3f} │ {p_w:>12.3f} │ {mc:>14.6f} │ {mw:>14.6f}")
        summary[alpha] = {'d': d, 'gap': gap, 'p_pos_c': p_c, 'p_pos_w': p_w, 'mc': mc, 'mw': mw}

    # ── 防坍缩模拟 (使用多个样本取平均) ──
    print("\n  Anti-collapse simulation (average over first 5 questions, response 0) ...")
    sharp_all = {a: [] for a in ALPHAS}
    n_sim = min(5, N_Q)
    for qi in range(n_sim):
        res = results_data[qi]
        sr = simulate_alpha_sharpening(model, tok, res['prompt'],
                                        res['responses'][0].get('response', ''))
        if sr is not None:
            for a in ALPHAS:
                sharp_all[a].append(sr['results'][a])

    temps = [1.0, 0.7, 0.5, 0.3]
    # 计算各 alpha 在 tau=0.3 时相对 tau=1.0 的平均变化率
    print(f"\n  {'alpha':<7s} │ {'tau=1.0':>10s} │ {'tau=0.7':>10s} │ {'tau=0.5':>10s} │ {'tau=0.3':>10s} │ {'Delta(0.3)':>10s}")
    print("  " + "─"*65)
    collapse_delta = {}
    for alpha in ALPHAS:
        vals = sharp_all[alpha]  # list of [R@1.0, R@0.7, R@0.5, R@0.3]
        if not vals:
            continue
        means = [np.mean([v[ti] for v in vals]) for ti in range(len(temps))]
        base = means[0] if means[0] != 0 else 1
        delta = (means[-1] - means[0]) / (abs(base) + eps) * 100
        collapse_delta[alpha] = delta
        print(f"  {alpha:<7.1f} │ {means[0]:>10.6f} │ {means[1]:>10.6f} │ {means[2]:>10.6f} │ {means[3]:>10.6f} │ {delta:>+9.1f}%")

    # ── 寻找最优 alpha ──
    print("\n" + "="*100)
    print("  Optimal alpha search: d > 0 AND delta < 0")
    print("="*100)
    candidates = []
    for alpha in ALPHAS:
        d = summary[alpha]['d']
        delta = collapse_delta.get(alpha, 999)
        ok_d = "OK" if d > 0 else "X"
        ok_c = "OK" if delta < 0 else "X"
        both = "*** CANDIDATE ***" if d > 0 and delta < 0 else ""
        print(f"  alpha={alpha:.1f}: d={d:+.4f} [{ok_d}], collapse_delta={delta:+.1f}% [{ok_c}]  {both}")
        if d > 0 and delta < 0:
            candidates.append((alpha, d, delta))

    if candidates:
        # 选择 d 最大的
        best = max(candidates, key=lambda x: x[1])
        print(f"\n  >>> Best candidate: alpha={best[0]:.1f}, d={best[1]:+.4f}, collapse_delta={best[2]:+.1f}%")
    else:
        # 没有同时满足两个条件的, 找 transition point
        print("\n  No alpha satisfies both conditions. Finding transition point ...")
        for i in range(len(ALPHAS)-1):
            d1 = collapse_delta.get(ALPHAS[i], 0)
            d2 = collapse_delta.get(ALPHAS[i+1], 0)
            if d1 < 0 and d2 >= 0:
                print(f"  Collapse transition between alpha={ALPHAS[i]:.1f} ({d1:+.1f}%) and alpha={ALPHAS[i+1]:.1f} ({d2:+.1f}%)")
            cd1 = summary[ALPHAS[i]]['d']
            cd2 = summary[ALPHAS[i+1]]['d']
            if cd1 < 0 and cd2 >= 0:
                print(f"  Cohen d transition between alpha={ALPHAS[i]:.1f} ({cd1:+.4f}) and alpha={ALPHAS[i+1]:.1f} ({cd2:+.4f})")

    # ── 绘图 ──
    print("\nGenerating plots ...")
    _plot_alpha_sweep(summary, collapse_delta, OUT_DIR)

    print(f"\nAll plots saved to: {OUT_DIR}/")


def _plot_alpha_sweep(summary, collapse_delta, out_dir):
    """Alpha sweep 综合图"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Alpha Sweep: R_alpha = geo_mean_H( H_eff / H^alpha )\n"
                 "alpha=0: anti-collapse but wrong direction | alpha=1: right direction but collapse",
                 fontsize=15, fontweight='bold', y=1.01)

    alphas = sorted(summary.keys())

    # (0,0): Cohen's d vs alpha
    ax = axes[0, 0]
    ds = [summary[a]['d'] for a in alphas]
    colors = ['#E74C3C' if d < 0 else '#2ECC71' for d in ds]
    bars = ax.bar(range(len(alphas)), ds, color=colors, edgecolor='black', linewidth=1, alpha=0.85)
    for bar, d, a in zip(bars, ds, alphas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * (1 if d >= 0 else -1),
                f'{d:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax.set_xlabel('alpha', fontsize=13, fontweight='bold')
    ax.set_ylabel("Cohen's d", fontsize=13, fontweight='bold')
    ax.set_title("Cohen's d (correct vs wrong)\nGreen=right direction, Red=inverted", fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # (0,1): Collapse delta vs alpha
    ax1 = axes[0, 1]
    deltas = [collapse_delta.get(a, 0) for a in alphas]
    colors1 = ['#2ECC71' if d < 0 else '#E74C3C' for d in deltas]
    bars1 = ax1.bar(range(len(alphas)), deltas, color=colors1, edgecolor='black', linewidth=1, alpha=0.85)
    for bar, d, a in zip(bars1, deltas, alphas):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1 * (1 if d >= 0 else -1),
                 f'{d:+.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(len(alphas)))
    ax1.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax1.set_xlabel('alpha', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Reward change at tau=0.3 (%)', fontsize=13, fontweight='bold')
    ax1.set_title("Anti-collapse: reward change under sharpening (tau=1->0.3)\nGreen=resists, Red=encourages collapse", fontsize=12, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # (1,0): P(adv>0|correct) vs alpha
    ax2 = axes[1, 0]
    pcs = [summary[a]['p_pos_c'] for a in alphas]
    pws = [summary[a]['p_pos_w'] for a in alphas]
    x = np.arange(len(alphas))
    w = 0.35
    ax2.bar(x - w/2, pcs, w, color='#2ECC71', alpha=0.8, label='P(adv>0 | correct)', edgecolor='black')
    ax2.bar(x + w/2, pws, w, color='#E74C3C', alpha=0.8, label='P(adv>0 | wrong)', edgecolor='black')
    for i, (pc, pw) in enumerate(zip(pcs, pws)):
        ax2.text(i - w/2, pc + 0.01, f'{pc:.3f}', ha='center', fontsize=8)
        ax2.text(i + w/2, pw + 0.01, f'{pw:.3f}', ha='center', fontsize=8)
    ax2.axhline(0.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax2.set_xlabel('alpha', fontsize=13, fontweight='bold')
    ax2.set_ylabel('P(adv>0)', fontsize=13, fontweight='bold')
    ax2.set_title("P(adv>0 | correct) vs P(adv>0 | wrong)\nWant: green > 0.5, gap between green and red", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.3, 0.75)

    # (1,1): Combined score: d vs collapse delta (scatter)
    ax3 = axes[1, 1]
    ds_arr = np.array([summary[a]['d'] for a in alphas])
    deltas_arr = np.array([collapse_delta.get(a, 0) for a in alphas])

    # Color by both conditions
    for i, a in enumerate(alphas):
        color = '#2ECC71' if ds_arr[i] > 0 and deltas_arr[i] < 0 else \
                '#F39C12' if ds_arr[i] > 0 or deltas_arr[i] < 0 else '#E74C3C'
        size = 200 if ds_arr[i] > 0 and deltas_arr[i] < 0 else 100
        ax3.scatter(deltas_arr[i], ds_arr[i], s=size, c=color, edgecolors='black',
                    linewidth=2, zorder=5)
        ax3.annotate(f'a={a:.1f}', (deltas_arr[i], ds_arr[i]),
                     textcoords='offset points', xytext=(8, 5), fontsize=10, fontweight='bold')

    ax3.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax3.axvline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)

    # Quadrant labels
    ax3.text(0.05, 0.95, 'IDEAL ZONE\nd>0, collapse<0', transform=ax3.transAxes,
             fontsize=12, fontweight='bold', color='#27AE60', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', alpha=0.8))
    ax3.text(0.65, 0.95, 'CURRENT\nd>0, collapse>0', transform=ax3.transAxes,
             fontsize=11, fontweight='bold', color='#E67E22', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7', alpha=0.8))
    ax3.text(0.05, 0.15, 'WORST\nd<0, collapse<0', transform=ax3.transAxes,
             fontsize=11, fontweight='bold', color='#C0392B', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', alpha=0.8))

    ax3.set_xlabel('Collapse delta at tau=0.3 (%)', fontsize=13, fontweight='bold')
    ax3.set_ylabel("Cohen's d", fontsize=13, fontweight='bold')
    ax3.set_title("Tradeoff: Discrimination vs Anti-collapse\nGreen = both criteria met", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, 'alpha_sweep_validation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
