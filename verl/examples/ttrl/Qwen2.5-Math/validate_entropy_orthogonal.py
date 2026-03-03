#!/usr/bin/env python3
"""
验证熵正交 R_geo (Entropy-Orthogonal R_geo)

核心思想:
  R̃ = R_geo − β · mean_H,   β = cov(R_geo, mean_H) / var(mean_H)
  从 R_geo 中移除与 H 线性相关的分量 → 保留区分度, 消除坍缩偏差.

验证目标:
  1. d(R̃) > 0      → 区分度保留
  2. delta(R̃) ≈ 0  → 坍缩中性
  3. P(A>0|C) > P(A>0|W) → GRPO advantage 有效
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
CHUNK = 256


# ─────────────────── helpers ───────────────────

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


def grpo_adv(vals):
    r = np.asarray(vals, float)
    s = r.std()
    return (r - r.mean()) / (s + eps) if s > eps else np.zeros_like(r)


@torch.no_grad()
def _logits_to_metrics(logits, device):
    """从 logits (T, V) 计算 R_geo, mean_H, mean_lambda, R_alpha042"""
    T = logits.size(0)
    H    = torch.empty(T, device=device)
    Heff = torch.empty(T, device=device)
    lam  = torch.empty(T, device=device)

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        lp = torch.log_softmax(logits[s:e], dim=-1)
        p  = lp.exp()
        H[s:e]    = -(p * lp).sum(-1)
        Heff[s:e] = -(p * p * lp).sum(-1)
        lam[s:e]  = (p * p).sum(-1)
        del p, lp

    sH = H.sum()
    r = Heff / (H + eps)
    log_r = torch.log(r.clamp(min=eps))
    R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    log_Heff = torch.log(Heff.clamp(min=eps))
    log_H    = torch.log(H.clamp(min=eps))
    R_a042 = float(torch.exp((H * (log_Heff - 0.42 * log_H)).sum() / (sH + eps)).item())

    return {
        'R_geo':    R_geo,
        'mean_H':   float(H.mean().item()),
        'mean_lam': float(lam.mean().item()),
        'R_a042':   R_a042,
    }


@torch.no_grad()
def compute_all(model, tok, prompt, response, do_collapse=False,
                temps=(1.0, 0.7, 0.5, 0.3), device=DEV):
    """单次 forward → τ=1 指标 + (可选) 多温度坍缩指标"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl  = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl - 1:-1].float()
    T = logits_orig.size(0)
    if T < 3:
        return None, None

    sample = _logits_to_metrics(logits_orig, device)

    collapse_rows = None
    if do_collapse:
        collapse_rows = []
        for tau in temps:
            m = _logits_to_metrics(logits_orig / tau, device)
            m['tau'] = tau
            collapse_rows.append(m)

    del logits_orig, out
    torch.cuda.empty_cache()
    return sample, collapse_rows


# ─────────────────── main ───────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading ...")
    with open(FILE) as f:
        data = json.load(f)
    results = data['results']
    N_Q = len(results)

    tok   = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    N_COLLAPSE = min(8, N_Q)          # 坍缩模拟用前 8 题
    temps = [1.0, 0.7, 0.5, 0.3]

    # ═══ PART 1: 收集指标 ═══
    print("\n== PART 1: 收集所有样本指标 ==")
    all_Rgeo, all_H, all_lam, all_Ra, all_lbl = [], [], [], [], []
    group_idx = []                     # (start, end)
    # 坍缩模拟缓存
    collapse_buf = {t: {'R_geo': [], 'mean_H': [], 'mean_lam': [], 'R_a042': []}
                    for t in temps}

    ptr = 0
    for qi, res in enumerate(results):
        prompt = res['prompt']
        K = len(res['responses'])
        start = ptr
        for i, resp in enumerate(res['responses']):
            s, cr = compute_all(model, tok, prompt, resp.get('response', ''),
                                do_collapse=(qi < N_COLLAPSE), temps=temps)
            if s is None:
                s = {'R_geo': 0, 'mean_H': 0, 'mean_lam': 0, 'R_a042': 0}
            all_Rgeo.append(s['R_geo'])
            all_H.append(s['mean_H'])
            all_lam.append(s['mean_lam'])
            all_Ra.append(s['R_a042'])
            all_lbl.append(1 if resp.get('is_correct', False) else 0)
            if cr:
                for row in cr:
                    t = row['tau']
                    for k in ['R_geo', 'mean_H', 'mean_lam', 'R_a042']:
                        collapse_buf[t][k].append(row[k])
            ptr += 1
        group_idx.append((start, ptr))
        if (qi + 1) % 5 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}]  (n_correct={sum(all_lbl)}, n_wrong={len(all_lbl)-sum(all_lbl)})")

    R     = np.array(all_Rgeo)
    H_arr = np.array(all_H)
    L_arr = np.array(all_lam)
    Ra    = np.array(all_Ra)
    lbl   = np.array(all_lbl)

    # ═══ PART 2: 估计 β ═══
    print("\n== PART 2: 估计回归系数 β ==")
    ok = H_arr > eps
    Rm, Hm, Lm = R[ok], H_arr[ok], L_arr[ok]

    beta_H = np.cov(Rm, Hm)[0, 1] / (np.var(Hm) + 1e-12)
    beta_L = np.cov(Rm, Lm)[0, 1] / (np.var(Lm) + 1e-12)
    r2_H = np.corrcoef(Rm, Hm)[0, 1] ** 2
    r2_L = np.corrcoef(Rm, Lm)[0, 1] ** 2

    print(f"  β_H  = {beta_H:+.6f}   R²(R_geo, H) = {r2_H:.4f}")
    print(f"  β_λ  = {beta_L:+.6f}   R²(R_geo, λ) = {r2_L:.4f}")

    Rt_H = R - beta_H * H_arr      # R̃_H
    Rt_L = R - beta_L * L_arr      # R̃_λ

    chk_H = np.corrcoef(Rt_H[ok], Hm)[0, 1]
    chk_L = np.corrcoef(Rt_L[ok], Lm)[0, 1]
    print(f"  验证: corr(R̃_H, H)={chk_H:+.2e}   corr(R̃_λ, λ)={chk_L:+.2e}")

    # per-group β 稳定性
    betas_g = []
    for s, e in group_idx:
        Rg, Hg = R[s:e], H_arr[s:e]
        v = np.var(Hg)
        if v > 1e-12 and len(Rg) >= 4:
            betas_g.append(np.cov(Rg, Hg)[0, 1] / v)
    betas_g = np.array(betas_g)
    print(f"  组内 β_H: mean={betas_g.mean():+.4f}, std={betas_g.std():.4f}, "
          f"median={np.median(betas_g):+.4f}  (全局={beta_H:+.4f})")

    # ═══ PART 3: 区分度 ═══
    print("\n== PART 3: 区分度 (Cohen's d) ==")
    cm, wm = lbl == 1, lbl == 0
    metrics_map = {
        'R_geo':              R,
        'R̃_H  (detrend H)':  Rt_H,
        'R̃_λ  (detrend λ)':  Rt_L,
        'R_alpha(0.42)':      Ra,
    }
    disc = {}
    print(f"  {'指标':<24s} │ {'Cohen d':>9s} │ {'正确均值':>12s} │ {'错误均值':>12s}")
    print("  " + "─" * 62)
    for name, arr in metrics_map.items():
        d = cohens_d(arr[cm], arr[wm])
        mc, mw = arr[cm].mean(), arr[wm].mean()
        print(f"  {name:<24s} │ {d:>+9.4f} │ {mc:>12.6f} │ {mw:>12.6f}")
        disc[name] = d

    # ═══ PART 4: GRPO Advantage ═══
    print("\n== PART 4: GRPO Advantage ==")
    adv_s = {n: {'c': [], 'w': []} for n in metrics_map}
    for s, e in group_idx:
        gl = lbl[s:e]
        for name, arr in metrics_map.items():
            a = grpo_adv(arr[s:e])
            for i in range(e - s):
                adv_s[name]['c' if gl[i] else 'w'].append(a[i])

    print(f"  {'指标':<24s} │ {'Adv Gap':>9s} │ {'P(+|C)':>8s} │ {'P(+|W)':>8s} │ {'ΔP':>8s}")
    print("  " + "─" * 68)
    for name in metrics_map:
        ac, aw = np.array(adv_s[name]['c']), np.array(adv_s[name]['w'])
        gap = ac.mean() - aw.mean()
        pc, pw = (ac > 0).mean(), (aw > 0).mean()
        print(f"  {name:<24s} │ {gap:>+9.4f} │ {pc:>8.3f} │ {pw:>8.3f} │ {pc-pw:>+8.3f}")

    # ═══ PART 5: 坍缩模拟 ═══
    print("\n== PART 5: 防坍缩模拟 (β_H 固定) ==")
    col = {}
    for t in temps:
        Rt = np.array(collapse_buf[t]['R_geo'])
        Ht = np.array(collapse_buf[t]['mean_H'])
        Lt = np.array(collapse_buf[t]['mean_lam'])
        At = np.array(collapse_buf[t]['R_a042'])
        col[t] = {
            'R_geo':   Rt.mean(),
            'R̃_H':    (Rt - beta_H * Ht).mean(),
            'R̃_λ':    (Rt - beta_L * Lt).mean(),
            'R_a042':  At.mean(),
            'mean_H':  Ht.mean(),
        }

    ckeys = ['R_geo', 'R̃_H', 'R̃_λ', 'R_a042', 'mean_H']
    print(f"  {'指标':<12s} │ {'τ=1.0':>10s} │ {'τ=0.7':>10s} │ {'τ=0.5':>10s} │ {'τ=0.3':>10s} │ {'Δ(0.3/1.0)':>10s}")
    print("  " + "─" * 72)
    col_delta = {}
    for k in ckeys:
        v = [col[t][k] for t in temps]
        delta = (v[-1] - v[0]) / (abs(v[0]) + 1e-12) * 100
        col_delta[k] = delta
        print(f"  {k:<12s} │ {v[0]:>10.6f} │ {v[1]:>10.6f} │ {v[2]:>10.6f} │ {v[3]:>10.6f} │ {delta:>+9.1f}%")

    # adaptive β at each temp
    print("\n  自适应 β_H (每温度重估):")
    for t in temps:
        Rt_ = np.array(collapse_buf[t]['R_geo'])
        Ht_ = np.array(collapse_buf[t]['mean_H'])
        vv  = np.var(Ht_)
        b   = np.cov(Rt_, Ht_)[0, 1] / (vv + 1e-12) if vv > 1e-12 else 0.0
        print(f"    τ={t:.1f}: β_H = {b:+.6f}")

    # ═══ PART 6: 最终汇总 ═══
    print("\n" + "=" * 85)
    print("  ═══ 最终汇总 ═══")
    print("=" * 85)
    label_to_col = {
        'R_geo': 'R_geo', 'R̃_H  (detrend H)': 'R̃_H',
        'R̃_λ  (detrend λ)': 'R̃_λ', 'R_alpha(0.42)': 'R_a042',
    }
    for name in metrics_map:
        d = disc[name]
        ck = label_to_col[name]
        delta = col_delta.get(ck, float('nan'))
        ideal = "★ IDEAL ★" if d > 0 and abs(delta) < 5 else \
                "✓ d>0" if d > 0 else "✗"
        print(f"  {name:<24s}: d = {d:>+.4f},  delta = {delta:>+.1f}%  {ideal}")

    # ═══ PART 7: Plot ═══
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    fig.suptitle("Entropy-Orthogonal R_geo Validation", fontsize=15, fontweight='bold')

    COLORS = {'R_geo': '#3498DB', 'R̃_H  (detrend H)': '#E74C3C',
              'R̃_λ  (detrend λ)': '#9B59B6', 'R_alpha(0.42)': '#2ECC71'}
    names = list(metrics_map.keys())

    # (a) Cohen's d
    ax = axes[0]
    ds = [disc[n] for n in names]
    bars = ax.bar(range(len(names)), ds, color=[COLORS[n] for n in names],
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    for b, d in zip(bars, ds):
        ax.text(b.get_x() + b.get_width()/2, d + 0.02, f'{d:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel("Cohen's d", fontsize=13, fontweight='bold')
    ax.set_title("Discrimination (d > 0 = good)", fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Collapse delta
    ax1 = axes[1]
    deltas = [col_delta.get(label_to_col[n], 0) for n in names]
    c1 = ['#2ECC71' if abs(d) < 5 else ('#E74C3C' if d > 0 else '#3498DB') for d in deltas]
    bars1 = ax1.bar(range(len(names)), deltas, color=c1, edgecolor='black',
                    linewidth=0.8, alpha=0.85)
    for b, d in zip(bars1, deltas):
        ax1.text(b.get_x() + b.get_width()/2, d + 0.5 * (1 if d >= 0 else -1),
                 f'{d:+.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=8, rotation=20, ha='right')
    ax1.set_ylabel("Delta (%)", fontsize=13, fontweight='bold')
    ax1.set_title("Anti-collapse delta (green = |Δ|<5%)", fontsize=12, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=2)
    ax1.grid(True, alpha=0.3, axis='y')

    # (c) Sharpening trajectory
    ax2 = axes[2]
    traj_keys = {'R_geo': 'R_geo', 'R̃_H  (detrend H)': 'R̃_H',
                 'R̃_λ  (detrend λ)': 'R̃_λ', 'R_alpha(0.42)': 'R_a042'}
    for n in names:
        ck = traj_keys[n]
        vs = [col[t][ck] for t in temps]
        base = vs[0] if abs(vs[0]) > eps else 1.0
        norm = [v / abs(base) for v in vs]
        ax2.plot(temps, norm, color=COLORS[n], linewidth=2.5, marker='o',
                 markersize=9, label=n, alpha=0.85)
    ax2.axhline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax2.invert_xaxis()
    ax2.set_xlabel("Temperature τ", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Normalized reward (τ=1 → 1.0)", fontsize=13, fontweight='bold')
    ax2.set_title("Sharpening trajectory", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'entropy_orthogonal_validation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # scatter: R_geo vs mean_H with regression line
    fig2, ax3 = plt.subplots(figsize=(10, 8))
    ax3.scatter(H_arr[cm], R[cm], alpha=0.3, s=15, c='#2ECC71', label='Correct')
    ax3.scatter(H_arr[wm], R[wm], alpha=0.3, s=15, c='#E74C3C', label='Wrong')
    # regression line
    h_range = np.linspace(H_arr[ok].min(), H_arr[ok].max(), 100)
    ax3.plot(h_range, np.mean(Rm) + beta_H * (h_range - np.mean(Hm)),
             'k--', linewidth=2, label=f'β_H={beta_H:.4f}, R²={r2_H:.3f}')
    ax3.set_xlabel("mean_H", fontsize=13, fontweight='bold')
    ax3.set_ylabel("R_geo", fontsize=13, fontweight='bold')
    ax3.set_title(f"R_geo vs mean_H  (β={beta_H:+.4f}, removing this gives R̃_H)",
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    path2 = os.path.join(OUT_DIR, 'entropy_orthogonal_scatter.png')
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
