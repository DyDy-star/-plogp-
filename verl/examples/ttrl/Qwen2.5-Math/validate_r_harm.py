#!/usr/bin/env python3
"""
验证 R_harm = sum(H_t) / (sum(H_t^2 / H_eff_t) + eps)
即 r_t 的 H 加权调和平均

与 R_geo (几何平均), R_arith (算术平均), R_alpha(0.42) 对比
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
def compute_all(model, tok, prompt, response, device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T = logits.size(0)

    empty = {k: 0.0 for k in ['R_harm', 'R_geo', 'R_arith', 'R_alpha042', 'mean_H', 'mean_lam']}
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

    r = Heff / (H + eps)

    # ── R_harm = sum(H) / (sum(H^2 / H_eff) + eps) ──
    denom_terms = H * H / (Heff + eps)   # H_t^2 / H_eff_t = H_t / r_t
    R_harm = float((sH / (denom_terms.sum() + eps)).item())

    # ── R_geo ──
    log_r = torch.log(r.clamp(min=eps))
    R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    # ── R_arith ──
    R_arith = float((Heff.sum() / (sH + eps)).item())

    # ── R_alpha(0.42) ──
    log_Heff = torch.log(Heff.clamp(min=eps))
    log_H = torch.log(H.clamp(min=eps))
    log_ratio_042 = log_Heff - 0.42 * log_H
    R_alpha042 = float(torch.exp((H * log_ratio_042).sum() / (sH + eps)).item())

    mean_lam = sum_lam / max(T, 1)
    mean_H = float(H.mean().item())

    del logits, H, Heff, r, out
    torch.cuda.empty_cache()
    return {
        'R_harm': R_harm, 'R_geo': R_geo, 'R_arith': R_arith,
        'R_alpha042': R_alpha042, 'mean_H': mean_H, 'mean_lam': mean_lam,
    }


@torch.no_grad()
def simulate_sharpening(model, tok, prompt, response, temps=[1.0, 0.7, 0.5, 0.3], device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T = logits_orig.size(0)
    if T < 3:
        return None

    results = []
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
        r = Heff / (H + eps)

        # R_harm
        denom = (H * H / (Heff + eps)).sum()
        R_harm = float((sH / (denom + eps)).item())

        # R_geo
        log_r = torch.log(r.clamp(min=eps))
        R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

        # R_arith
        R_arith = float((Heff.sum() / (sH + eps)).item())

        # R_alpha(0.42)
        log_Heff = torch.log(Heff.clamp(min=eps))
        log_H = torch.log(H.clamp(min=eps))
        log_ratio = log_Heff - 0.42 * log_H
        R_alpha042 = float(torch.exp((H * log_ratio).sum() / (sH + eps)).item())

        results.append({
            'tau': tau, 'R_harm': R_harm, 'R_geo': R_geo,
            'R_arith': R_arith, 'R_alpha042': R_alpha042,
            'mean_H': float(H.mean().item()),
        })

    del logits_orig, out
    torch.cuda.empty_cache()
    return results


def grpo_advantage(rewards):
    r = np.array(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / (sp + 1e-12)


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

    KEYS = ['R_harm', 'R_geo', 'R_arith', 'R_alpha042']
    all_correct = {k: [] for k in KEYS}
    all_wrong   = {k: [] for k in KEYS}
    all_adv_c   = {k: [] for k in KEYS}
    all_adv_w   = {k: [] for k in KEYS}

    for qi, res in enumerate(results_data):
        prompt = res['prompt']
        responses = res['responses']
        K = len(responses)
        true_labels = [1 if r.get('is_correct', False) else 0 for r in responses]

        ml = [compute_all(model, tok, prompt, r.get('response', '')) for r in responses]

        for mk in KEYS:
            vals = [m[mk] for m in ml]
            adv = grpo_advantage(vals)
            for i in range(K):
                dst_c = all_correct if true_labels[i] else all_wrong
                dst_a = all_adv_c if true_labels[i] else all_adv_w
                dst_c[mk].append(vals[i])
                dst_a[mk].append(adv[i])

        if (qi + 1) % 10 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}]")

    # ═══ 区分力表 ═══
    print("\n" + "="*100)
    print("  R_harm vs R_geo vs R_arith vs R_alpha(0.42)")
    print("="*100)
    print(f"  {'指标':<15s} │ {'Cohen d':>9s} │ {'Adv Gap':>9s} │ {'P(+|C)':>8s} │ {'P(+|W)':>8s} │ {'正确均值':>12s} │ {'错误均值':>12s}")
    print("  " + "─"*80)

    summary = {}
    for mk in KEYS:
        rc, rw = all_correct[mk], all_wrong[mk]
        ac, aw = all_adv_c[mk], all_adv_w[mk]
        d = cohens_d(rc, rw)
        gap = np.mean(ac) - np.mean(aw)
        pc = np.mean(np.array(ac) > 0)
        pw = np.mean(np.array(aw) > 0)
        mc, mw = np.mean(rc), np.mean(rw)
        print(f"  {mk:<15s} │ {d:>+9.4f} │ {gap:>9.4f} │ {pc:>8.3f} │ {pw:>8.3f} │ {mc:>12.5f} │ {mw:>12.5f}")
        summary[mk] = {'d': d, 'gap': gap, 'pc': pc, 'mc': mc, 'mw': mw}

    # ═══ 防坍缩模拟 ═══
    print("\n  防坍缩模拟 (温度缩放, 8 个样本平均):")
    sharp_all = {k: {t: [] for t in [1.0, 0.7, 0.5, 0.3]} for k in KEYS}

    for qi in range(min(8, N_Q)):
        res = results_data[qi]
        sr = simulate_sharpening(model, tok, res['prompt'],
                                  res['responses'][0].get('response', ''))
        if sr:
            for entry in sr:
                for k in KEYS:
                    sharp_all[k][entry['tau']].append(entry[k])

    temps = [1.0, 0.7, 0.5, 0.3]
    print(f"\n  {'指标':<15s} │ {'tau=1.0':>10s} │ {'tau=0.7':>10s} │ {'tau=0.5':>10s} │ {'tau=0.3':>10s} │ {'Delta(0.3)':>10s}")
    print("  " + "─"*72)

    collapse = {}
    for k in KEYS:
        means = [np.mean(sharp_all[k][t]) for t in temps]
        delta = (means[-1] - means[0]) / (abs(means[0]) + eps) * 100
        collapse[k] = delta
        print(f"  {k:<15s} │ {means[0]:>10.5f} │ {means[1]:>10.5f} │ {means[2]:>10.5f} │ {means[3]:>10.5f} │ {delta:>+9.1f}%")

    # ═══ 综合 ═══
    print("\n  综合判断:")
    for k in KEYS:
        d = summary[k]['d']
        delta = collapse[k]
        both = "*** YES ***" if d > 0 and delta < 0 else ""
        print(f"  {k:<15s}: d={d:>+.4f}, delta={delta:>+.1f}%  {both}")

    # ═══ 数学关系验证: HM ≤ GM ≤ AM ═══
    print("\n  数学关系 HM ≤ GM ≤ AM 验证 (前 5 个正确回答):")
    for i in range(min(5, len(all_correct['R_harm']))):
        hm = all_correct['R_harm'][i]
        gm = all_correct['R_geo'][i]
        am = all_correct['R_arith'][i]
        ok = "OK" if hm <= gm + 0.001 and gm <= am + 0.001 else "FAIL"
        print(f"    HM={hm:.5f} ≤ GM={gm:.5f} ≤ AM={am:.5f}  [{ok}]")

    # ═══ 绘图 ═══
    _plot(summary, collapse, sharp_all, temps, OUT_DIR)
    print(f"\nDone.")


def _plot(summary, collapse, sharp_all, temps, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("R_harm (Harmonic Mean) vs R_geo (Geometric) vs R_arith (Arithmetic) vs R_alpha(0.42)",
                 fontsize=14, fontweight='bold')

    KEYS = ['R_harm', 'R_geo', 'R_arith', 'R_alpha042']
    COLORS = {'R_harm': '#9B59B6', 'R_geo': '#3498DB', 'R_arith': '#E67E22', 'R_alpha042': '#2ECC71'}
    LABELS = {'R_harm': 'R_harm (HM)', 'R_geo': 'R_geo (GM)', 'R_arith': 'R_arith (AM)', 'R_alpha042': 'R_alpha(0.42)'}

    # (0): Cohen's d
    ax = axes[0]
    ds = [summary[k]['d'] for k in KEYS]
    bars = ax.bar(range(len(KEYS)), ds, color=[COLORS[k] for k in KEYS],
                  edgecolor='black', linewidth=1, alpha=0.85)
    for bar, d, k in zip(bars, ds, KEYS):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{d:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([LABELS[k] for k in KEYS], fontsize=10)
    ax.set_ylabel("Cohen's d", fontsize=13, fontweight='bold')
    ax.set_title("Discrimination (d > 0 = correct direction)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=1)

    # (1): Collapse delta
    ax1 = axes[1]
    deltas = [collapse[k] for k in KEYS]
    colors1 = ['#2ECC71' if d < 0 else '#E74C3C' for d in deltas]
    bars1 = ax1.bar(range(len(KEYS)), deltas, color=colors1, edgecolor='black', linewidth=1, alpha=0.85)
    for bar, d, k in zip(bars1, deltas, KEYS):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5 * (1 if d >= 0 else -1),
                 f'{d:+.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(len(KEYS)))
    ax1.set_xticklabels([LABELS[k] for k in KEYS], fontsize=10)
    ax1.axhline(0, color='black', linewidth=2)
    ax1.set_ylabel('Delta at tau=0.3 (%)', fontsize=13, fontweight='bold')
    ax1.set_title("Anti-collapse (green=resists, red=encourages)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # (2): Trajectory
    ax2 = axes[2]
    for k in KEYS:
        means = [np.mean(sharp_all[k][t]) for t in temps]
        base = means[0] if means[0] != 0 else 1
        norm = [m / (abs(base) + eps) for m in means]
        ax2.plot(temps, norm, color=COLORS[k], linewidth=3, marker='o', markersize=10,
                 label=LABELS[k], alpha=0.9)
    ax2.axhline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax2.invert_xaxis()
    ax2.set_xlabel('Temperature', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Normalized reward', fontsize=13, fontweight='bold')
    ax2.set_title("Reward under sharpening", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'r_harm_validation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
