#!/usr/bin/env python3
"""
验证 KL(logit || Gaussian) 即 negentropy 作为无监督奖励函数
核心性质: 标准化后的 logit 矩 (kurtosis, negentropy) 在温度缩放下严格不变
=> 理论上 collapse delta = 0%

与 R_geo, R_arith 对比区分力和防坍缩
"""

import json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
CHUNK = 128   # smaller chunk to avoid OOM with extra moment computation


@torch.no_grad()
def compute_metrics(model, tok, prompt, response, device=DEV):
    """Compute negentropy/kurtosis metrics + baselines for one response."""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()   # (T, V)
    T = logits.size(0)

    KEYS = ['kurt_mean', 'negent_mean', 'kurt_Hw', 'negent_Hw',
            'negent_geo', 'R_geo', 'R_arith']
    empty = {k: 0.0 for k in KEYS}
    if T < 3:
        return empty

    H       = torch.empty(T, device=device)
    Heff    = torch.empty(T, device=device)
    kurt_t  = torch.empty(T, device=device)
    negent_t = torch.empty(T, device=device)

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        chunk = logits[s:e]                   # (c, V)

        # ── entropy ──
        lp = torch.log_softmax(chunk, dim=-1)
        p = lp.exp()
        H[s:e]    = -(p * lp).sum(-1)
        Heff[s:e] = -(p * p * lp).sum(-1)
        del p, lp

        # ── logit moments (standardized) ──
        mu  = chunk.mean(dim=-1, keepdim=True)
        cen = chunk - mu
        var = cen.pow(2).mean(dim=-1, keepdim=True)
        std = var.sqrt().clamp(min=eps)
        z   = cen / std                       # standardized logits

        m3 = z.pow(3).mean(dim=-1)            # skewness
        m4 = z.pow(4).mean(dim=-1)            # raw 4th moment

        kurt_t[s:e]  = m4 - 3.0              # excess kurtosis
        # Hyvarinen negentropy: J ≈ skew²/12 + (kurt)²/48
        negent_t[s:e] = m3.pow(2) / 12.0 + (m4 - 3.0).pow(2) / 48.0

        del chunk, cen, z

    sH = H.sum()
    if sH < eps:
        return empty

    # ── baselines ──
    r = Heff / (H + eps)
    log_r = torch.log(r.clamp(min=eps))
    R_geo  = float(torch.exp((H * log_r).sum() / (sH + eps)).item())
    R_arith = float((Heff.sum() / (sH + eps)).item())

    # ── simple means ──
    kurt_mean   = float(kurt_t.mean().item())
    negent_mean = float(negent_t.mean().item())

    # ── H-weighted means ──
    w = H / (sH + eps)
    kurt_Hw   = float((w * kurt_t).sum().item())
    negent_Hw = float((w * negent_t).sum().item())

    # ── H-weighted geometric mean of negentropy ──
    log_neg = torch.log(negent_t.clamp(min=eps))
    negent_geo = float(torch.exp((H * log_neg).sum() / (sH + eps)).item())

    del logits, out, H, Heff, kurt_t, negent_t
    torch.cuda.empty_cache()

    return {
        'kurt_mean': kurt_mean, 'negent_mean': negent_mean,
        'kurt_Hw': kurt_Hw, 'negent_Hw': negent_Hw,
        'negent_geo': negent_geo,
        'R_geo': R_geo, 'R_arith': R_arith,
    }


@torch.no_grad()
def simulate_sharpening(model, tok, prompt, response,
                        temps=(1.0, 0.7, 0.5, 0.3), device=DEV):
    """Check temperature invariance of negentropy metrics."""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T = logits_orig.size(0)
    if T < 3:
        return None

    rows = []
    for tau in temps:
        logits = logits_orig / tau

        H       = torch.empty(T, device=device)
        Heff    = torch.empty(T, device=device)
        kurt_t  = torch.empty(T, device=device)
        negent_t = torch.empty(T, device=device)

        for s in range(0, T, CHUNK):
            e = min(s + CHUNK, T)
            chunk = logits[s:e]

            lp = torch.log_softmax(chunk, dim=-1)
            p = lp.exp()
            H[s:e]    = -(p * lp).sum(-1)
            Heff[s:e] = -(p * p * lp).sum(-1)
            del p, lp

            mu  = chunk.mean(dim=-1, keepdim=True)
            cen = chunk - mu
            var = cen.pow(2).mean(dim=-1, keepdim=True)
            std = var.sqrt().clamp(min=eps)
            z   = cen / std

            m3 = z.pow(3).mean(dim=-1)
            m4 = z.pow(4).mean(dim=-1)
            kurt_t[s:e]  = m4 - 3.0
            negent_t[s:e] = m3.pow(2) / 12.0 + (m4 - 3.0).pow(2) / 48.0

            del chunk, cen, z

        sH = H.sum()
        r = Heff / (H + eps)
        log_r = torch.log(r.clamp(min=eps))
        R_geo  = float(torch.exp((H * log_r).sum() / (sH + eps)).item())
        R_arith = float((Heff.sum() / (sH + eps)).item())

        kurt_mean   = float(kurt_t.mean().item())
        negent_mean = float(negent_t.mean().item())

        w = H / (sH + eps)
        kurt_Hw   = float((w * kurt_t).sum().item())
        negent_Hw = float((w * negent_t).sum().item())

        log_neg = torch.log(negent_t.clamp(min=eps))
        negent_geo = float(torch.exp((H * log_neg).sum() / (sH + eps)).item())

        rows.append({
            'tau': tau,
            'kurt_mean': kurt_mean, 'negent_mean': negent_mean,
            'kurt_Hw': kurt_Hw, 'negent_Hw': negent_Hw,
            'negent_geo': negent_geo,
            'R_geo': R_geo, 'R_arith': R_arith,
        })

    del logits_orig, out
    torch.cuda.empty_cache()
    return rows


# ─── helpers ───
def grpo_advantage(rewards):
    r = np.array(rewards, dtype=float)
    s = r.std()
    return (r - r.mean()) / (s + eps) if s > eps else np.zeros_like(r)

def cohens_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / (sp + 1e-12)


# ═══════════════════════════════════════════════════════════════════
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

    KEYS = ['kurt_mean', 'negent_mean', 'kurt_Hw', 'negent_Hw',
            'negent_geo', 'R_geo', 'R_arith']

    all_correct = {k: [] for k in KEYS}
    all_wrong   = {k: [] for k in KEYS}
    all_adv_c   = {k: [] for k in KEYS}
    all_adv_w   = {k: [] for k in KEYS}

    # ── 1) discrimination ──
    for qi, res in enumerate(results_data):
        prompt = res['prompt']
        responses = res['responses']
        K = len(responses)
        labels = [1 if r.get('is_correct', False) else 0 for r in responses]

        ml = [compute_metrics(model, tok, prompt, r.get('response', ''))
              for r in responses]

        for mk in KEYS:
            vals = [m[mk] for m in ml]
            adv  = grpo_advantage(vals)
            for i in range(K):
                (all_correct if labels[i] else all_wrong)[mk].append(vals[i])
                (all_adv_c   if labels[i] else all_adv_w)[mk].append(adv[i])

        if (qi + 1) % 10 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}]")

    # print table
    print("\n" + "="*110)
    print("  KL(logit || Gaussian) = negentropy  验证")
    print("="*110)
    hdr = f"  {'指标':<14s} │ {'Cohen d':>9s} │ {'Adv Gap':>9s} │ {'P(+|C)':>8s} │ {'P(+|W)':>8s} │ {'正确均值':>14s} │ {'错误均值':>14s}"
    print(hdr)
    print("  " + "─"*95)

    summary = {}
    for mk in KEYS:
        rc, rw = all_correct[mk], all_wrong[mk]
        ac, aw = all_adv_c[mk], all_adv_w[mk]
        d   = cohens_d(rc, rw)
        gap = np.mean(ac) - np.mean(aw)
        pc  = np.mean(np.array(ac) > 0)
        pw  = np.mean(np.array(aw) > 0)
        mc, mw = np.mean(rc), np.mean(rw)
        print(f"  {mk:<14s} │ {d:>+9.4f} │ {gap:>9.4f} │ {pc:>8.3f} │ {pw:>8.3f} │ {mc:>14.5f} │ {mw:>14.5f}")
        summary[mk] = dict(d=d, gap=gap, pc=pc, pw=pw, mc=mc, mw=mw)

    # ── 2) anti-collapse simulation ──
    print("\n  防坍缩模拟 (温度缩放, 8 个样本平均):")
    temps = [1.0, 0.7, 0.5, 0.3]
    sharp_all = {k: {t: [] for t in temps} for k in KEYS}

    for qi in range(min(8, N_Q)):
        res = results_data[qi]
        sr = simulate_sharpening(model, tok, res['prompt'],
                                  res['responses'][0].get('response', ''))
        if sr:
            for entry in sr:
                for k in KEYS:
                    sharp_all[k][entry['tau']].append(entry[k])

    print(f"\n  {'指标':<14s} │ {'tau=1.0':>12s} │ {'tau=0.7':>12s} │ {'tau=0.5':>12s} │ {'tau=0.3':>12s} │ {'Delta':>9s}")
    print("  " + "─"*75)

    collapse = {}
    for k in KEYS:
        means = [np.mean(sharp_all[k][t]) if sharp_all[k][t] else 0 for t in temps]
        delta = (means[-1] - means[0]) / (abs(means[0]) + eps) * 100
        collapse[k] = delta
        print(f"  {k:<14s} │ {means[0]:>12.5f} │ {means[1]:>12.5f} │ {means[2]:>12.5f} │ {means[3]:>12.5f} │ {delta:>+8.1f}%")

    # ── 3) summary ──
    print("\n  综合判断 (需要 d>0 且 delta<=0):")
    for k in KEYS:
        d = summary[k]['d']
        delta = collapse[k]
        tag = "*** IDEAL ***" if d > 0 and abs(delta) < 2 else \
              "*** YES ***"   if d > 0 and delta <= 0 else ""
        print(f"  {k:<14s}: d={d:>+.4f}, delta={delta:>+.1f}%  {tag}")

    # ── 4) temperature invariance proof ──
    print("\n  温度不变性验证 (negent_mean 在不同 tau 下的值):")
    for qi in range(min(3, N_Q)):
        res = results_data[qi]
        sr = simulate_sharpening(model, tok, res['prompt'],
                                  res['responses'][0].get('response', ''))
        if sr:
            vals = [f"tau={e['tau']:.1f}: {e['negent_mean']:.6f}" for e in sr]
            print(f"    Q{qi}: " + " | ".join(vals))

    # ── 5) plot ──
    _plot(summary, collapse, sharp_all, temps, KEYS, OUT_DIR)
    print("\nDone.")


def _plot(summary, collapse, sharp_all, temps, KEYS, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    fig.suptitle("KL(logit || Gaussian) = Negentropy  vs  Baselines",
                 fontsize=15, fontweight='bold')

    COLORS = {
        'kurt_mean': '#9B59B6', 'negent_mean': '#E74C3C',
        'kurt_Hw': '#8E44AD', 'negent_Hw': '#C0392B',
        'negent_geo': '#D35400',
        'R_geo': '#3498DB', 'R_arith': '#2ECC71',
    }
    LABELS = {
        'kurt_mean': 'Kurtosis (mean)',
        'negent_mean': 'Negentropy (mean)',
        'kurt_Hw': 'Kurtosis (H-wt)',
        'negent_Hw': 'Negentropy (H-wt)',
        'negent_geo': 'Negentropy (geo)',
        'R_geo': 'R_geo (baseline)',
        'R_arith': 'R_arith (baseline)',
    }

    # (0) Cohen's d
    ax = axes[0]
    ds = [summary[k]['d'] for k in KEYS]
    bars = ax.bar(range(len(KEYS)), ds,
                  color=[COLORS[k] for k in KEYS],
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    for bar, d in zip(bars, ds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02 * (1 if d >= 0 else -1),
                f'{d:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([LABELS[k] for k in KEYS], fontsize=8, rotation=25, ha='right')
    ax.set_ylabel("Cohen's d", fontsize=13, fontweight='bold')
    ax.set_title("Discrimination", fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # (1) Collapse delta
    ax1 = axes[1]
    deltas = [collapse[k] for k in KEYS]
    colors1 = ['#2ECC71' if abs(d) < 2 else ('#2ECC71' if d < 0 else '#E74C3C') for d in deltas]
    bars1 = ax1.bar(range(len(KEYS)), deltas,
                    color=colors1, edgecolor='black', linewidth=0.8, alpha=0.85)
    for bar, d in zip(bars1, deltas):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5 * (1 if d >= 0 else -1),
                 f'{d:+.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(len(KEYS)))
    ax1.set_xticklabels([LABELS[k] for k in KEYS], fontsize=8, rotation=25, ha='right')
    ax1.axhline(0, color='black', linewidth=2)
    ax1.set_ylabel('Delta at tau=0.3 (%)', fontsize=13, fontweight='bold')
    ax1.set_title("Anti-collapse (green=good)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # (2) Trajectory
    ax2 = axes[2]
    for k in KEYS:
        means = [np.mean(sharp_all[k][t]) if sharp_all[k][t] else 0 for t in temps]
        base = means[0] if abs(means[0]) > eps else 1
        norm = [m / (abs(base) + eps) for m in means]
        ax2.plot(temps, norm, color=COLORS[k], linewidth=2.5, marker='o',
                 markersize=8, label=LABELS[k], alpha=0.85)
    ax2.axhline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax2.invert_xaxis()
    ax2.set_xlabel('Temperature', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Normalized reward', fontsize=13, fontweight='bold')
    ax2.set_title("Reward trajectory under sharpening", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'kl_gaussian_validation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
