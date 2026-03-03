#!/usr/bin/env python3
"""
精细 Alpha 扫描: 在 [0.30, 0.55] 区间精确搜索最优 alpha*
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

# 精细扫描区间
ALPHAS = [0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50, 0.55]


@torch.no_grad()
def compute_alpha_metrics(model, tok, prompt, response, alphas=ALPHAS, device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T = logits.size(0)

    empty = {f'R_a{a:.2f}': 0.0 for a in alphas}
    empty.update({'mean_H': 0.0})
    if T < 3:
        return empty

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
    if sH < eps:
        return empty

    results = {}
    log_Heff = torch.log(Heff.clamp(min=eps))
    log_H = torch.log(H.clamp(min=eps))

    for alpha in alphas:
        log_ratio = log_Heff - alpha * log_H
        R = float(torch.exp((H * log_ratio).sum() / (sH + eps)).item())
        results[f'R_a{alpha:.2f}'] = R

    results['mean_H'] = float(H.mean().item())

    del logits, H, Heff, out
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def simulate_sharpening(model, tok, prompt, response, alphas=ALPHAS,
                        temps=[1.0, 0.7, 0.5, 0.3], device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T = logits_orig.size(0)
    if T < 3:
        return None

    results = {a: [] for a in alphas}
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

    print("Loading model ...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    alpha_keys = [f'R_a{a:.2f}' for a in ALPHAS]
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
        for r in responses:
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

        if (qi + 1) % 10 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}]")

    # 防坍缩模拟
    print("  Anti-collapse simulation ...")
    sharp_all = {a: [] for a in ALPHAS}
    for qi in range(min(8, N_Q)):
        res = results_data[qi]
        sr = simulate_sharpening(model, tok, res['prompt'],
                                  res['responses'][0].get('response', ''))
        if sr:
            for a in ALPHAS:
                sharp_all[a].append(sr['results'][a])

    # 输出
    print("\n" + "="*105)
    print("  Fine Alpha Sweep: R_alpha = geo_mean_H( H_eff / H^alpha )")
    print("="*105)
    print(f"  {'alpha':<7s} │ {'Cohen d':>9s} │ {'Adv Gap':>9s} │ {'P(+|C)':>8s} │ {'P(+|W)':>8s} │ {'Gap P':>8s} │ {'Delta(0.3)':>10s} │ {'Result':>10s}")
    print("  " + "─"*85)

    summary = {}
    for alpha in ALPHAS:
        ak = f'R_a{alpha:.2f}'
        rc, rw = all_correct[ak], all_wrong[ak]
        ac, aw = all_adv_c[ak], all_adv_w[ak]
        d = cohens_d(rc, rw)
        gap = np.mean(ac) - np.mean(aw)
        p_c = np.mean(np.array(ac) > 0)
        p_w = np.mean(np.array(aw) > 0)
        p_gap = p_c - p_w

        vals = sharp_all[alpha]
        delta = 0
        if vals:
            means = [np.mean([v[ti] for v in vals]) for ti in range(4)]
            base = means[0] if means[0] != 0 else 1
            delta = (means[-1] - means[0]) / (abs(base) + eps) * 100

        ok_d = "d>0" if d > 0 else "d<0"
        ok_c = "anti" if delta < 0 else "pro"
        result = "IDEAL" if d > 0 and delta < 0 else ("OK-ish" if d > 0 and delta < 5 else "")

        print(f"  {alpha:<7.2f} │ {d:>+9.4f} │ {gap:>9.4f} │ {p_c:>8.3f} │ {p_w:>8.3f} │ {p_gap:>+8.3f} │ {delta:>+9.1f}% │ {result:>10s}")
        summary[alpha] = {'d': d, 'gap': gap, 'p_c': p_c, 'p_w': p_w, 'delta': delta}

    # 寻找 Pareto 最优
    print("\n" + "─"*60)
    print("  Pareto analysis: maximize d subject to delta <= 0")
    pareto = [(a, s['d'], s['delta']) for a, s in summary.items() if s['d'] > 0 and s['delta'] <= 0]
    if pareto:
        best = max(pareto, key=lambda x: x[1])
        print(f"  >>> BEST: alpha={best[0]:.2f}, d={best[1]:+.4f}, delta={best[2]:+.1f}%")

        # 也检查 "soft" Pareto (delta <= 3%)
        print("\n  Soft Pareto (delta <= +3%): maximize d")
        soft_pareto = [(a, s['d'], s['delta']) for a, s in summary.items() if s['d'] > 0 and s['delta'] <= 3]
        if soft_pareto:
            best_soft = max(soft_pareto, key=lambda x: x[1])
            print(f"  >>> SOFT BEST: alpha={best_soft[0]:.2f}, d={best_soft[1]:+.4f}, delta={best_soft[2]:+.1f}%")

    # Score: d * max(0, -delta) — 同时奖励高 d 和负 delta
    print("\n  Combined score: d * max(1, -delta/10) [balancing both goals]")
    for alpha in ALPHAS:
        s = summary[alpha]
        score = s['d'] * max(1.0, -s['delta'] / 10.0) if s['d'] > 0 else 0
        print(f"    alpha={alpha:.2f}: score = {score:.4f}  (d={s['d']:+.4f}, delta={s['delta']:+.1f}%)")

    # 绘图
    _plot_fine_sweep(summary, OUT_DIR)
    print(f"\nDone. Plots saved to {OUT_DIR}/")


def _plot_fine_sweep(summary, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Fine Alpha Sweep: Finding the Sweet Spot (d > 0 AND collapse < 0)",
                 fontsize=15, fontweight='bold')

    alphas = sorted(summary.keys())
    ds = [summary[a]['d'] for a in alphas]
    deltas = [summary[a]['delta'] for a in alphas]
    p_gaps = [summary[a]['p_c'] - summary[a]['p_w'] for a in alphas]

    # (0): d vs alpha (line)
    ax = axes[0]
    ax.plot(alphas, ds, 'o-', color='#3498DB', linewidth=3, markersize=10, label="Cohen's d")
    ax.axhline(0, color='red', linewidth=2, linestyle='--', alpha=0.5, label='d = 0 (threshold)')
    ax.fill_between(alphas, 0, ds, where=[d > 0 for d in ds], alpha=0.15, color='green')
    ax.fill_between(alphas, 0, ds, where=[d <= 0 for d in ds], alpha=0.15, color='red')
    for a, d in zip(alphas, ds):
        ax.annotate(f'{d:.3f}', (a, d), textcoords='offset points', xytext=(0, 12),
                    fontsize=9, ha='center', fontweight='bold')
    ax.set_xlabel('alpha', fontsize=13, fontweight='bold')
    ax.set_ylabel("Cohen's d", fontsize=13, fontweight='bold')
    ax.set_title("Discrimination Power", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (1): collapse delta vs alpha (line)
    ax1 = axes[1]
    ax1.plot(alphas, deltas, 's-', color='#E74C3C', linewidth=3, markersize=10, label='Collapse delta (%)')
    ax1.axhline(0, color='green', linewidth=2, linestyle='--', alpha=0.5, label='delta = 0 (threshold)')
    ax1.fill_between(alphas, 0, deltas, where=[d < 0 for d in deltas], alpha=0.15, color='green')
    ax1.fill_between(alphas, 0, deltas, where=[d >= 0 for d in deltas], alpha=0.15, color='red')
    for a, d in zip(alphas, deltas):
        ax1.annotate(f'{d:+.1f}%', (a, d), textcoords='offset points', xytext=(0, 12),
                     fontsize=9, ha='center', fontweight='bold')
    ax1.set_xlabel('alpha', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Reward change at tau=0.3 (%)', fontsize=13, fontweight='bold')
    ax1.set_title("Anti-Collapse Resistance", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # (2): Tradeoff scatter
    ax2 = axes[2]
    for i, a in enumerate(alphas):
        d, delta = ds[i], deltas[i]
        color = '#2ECC71' if d > 0 and delta < 0 else '#F39C12' if d > 0 else '#E74C3C'
        size = 250 if d > 0 and delta < 0 else 150
        ax2.scatter(delta, d, s=size, c=color, edgecolors='black', linewidth=2, zorder=5)
        ax2.annotate(f'{a:.2f}', (delta, d), textcoords='offset points',
                     xytext=(8, 5), fontsize=11, fontweight='bold')

    ax2.axhline(0, color='gray', linewidth=1.5, linestyle='--')
    ax2.axvline(0, color='gray', linewidth=1.5, linestyle='--')
    ax2.fill_between([-20, 0], [-1, -1], [2, 2], alpha=0.05, color='green')
    ax2.text(-15, 1.0, 'IDEAL\nd>0, collapse<0', fontsize=12, fontweight='bold', color='#27AE60')
    ax2.text(3, 1.0, 'DANGER\nd>0, collapse>0', fontsize=12, fontweight='bold', color='#E67E22')
    ax2.set_xlabel('Collapse delta (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel("Cohen's d", fontsize=13, fontweight='bold')
    ax2.set_title("Discrimination vs Anti-Collapse Tradeoff", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min(deltas)-3, max(deltas)+3)
    ax2.set_ylim(min(ds)-0.2, max(ds)+0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, 'alpha_fine_sweep.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
