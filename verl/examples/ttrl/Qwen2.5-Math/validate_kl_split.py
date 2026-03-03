#!/usr/bin/env python3
"""
验证 KL 分割奖励: KL(G||emp)_低熵 − KL(G||emp)_高熵

思路: 按 H_t 将 token 分为低熵/高熵两组, 用 KL 差作为奖励.
      KL 本身温度不变 (标准化 logits), 但分组随温度变化.

验证指标:
  kl_split      = mean(KL|H<H̄) - mean(KL|H≥H̄)     (用户提案)
  neg_kl_split  = -kl_split                           (取负)
  neg_kl_mean   = -mean(KL)                           (基线)
  neg_kl_Hw     = -Σ(H/ΣH)·KL                        (H 加权基线)
  neg_kl_low    = -mean(KL|H<H̄)                      (仅低熵)
  neg_kl_high   = -mean(KL|H≥H̄)                      (仅高熵)
  R_geo                                                (对比基线)
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
CHUNK = 64
NBINS = 200
BIN_LO, BIN_HI = -6.0, 6.0


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = np.sqrt(((len(a)-1)*np.var(a, ddof=1)+(len(b)-1)*np.var(b, ddof=1))/(len(a)+len(b)-2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


def grpo_adv(vals):
    r = np.asarray(vals, float)
    s = r.std()
    return (r - r.mean()) / (s + eps) if s > eps else np.zeros_like(r)


@torch.no_grad()
def extract_per_token(model, tok, prompt, response, device=DEV):
    """单次 forward: 返回 per-token H_t 和 KL_t(G||emp)."""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl  = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T, V = logits.shape
    if T < 3:
        del out; torch.cuda.empty_cache()
        return None

    bw = (BIN_HI - BIN_LO) / NBINS
    centers = torch.linspace(BIN_LO + bw/2, BIN_HI - bw/2, NBINS, device=device)
    gauss = torch.exp(-0.5 * centers**2) / math.sqrt(2*math.pi) * bw
    gauss = gauss / gauss.sum()
    log_gauss = torch.log(gauss)

    H_arr   = torch.empty(T, device=device)
    Heff_arr= torch.empty(T, device=device)
    kl_arr  = torch.empty(T, device=device)

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        chunk = logits[s:e]; c = e - s

        # entropy
        lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
        H_arr[s:e]    = -(p * lp).sum(-1)
        Heff_arr[s:e] = -(p * p * lp).sum(-1)
        del p, lp

        # standardize → histogram → KL(G||emp)
        mu  = chunk.mean(-1, keepdim=True)
        cen = chunk - mu
        std = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
        z   = cen / std
        del chunk, cen, mu, std

        bi = ((z - BIN_LO) / bw).long().clamp(0, NBINS - 1)
        hist = torch.zeros(c, NBINS, device=device)
        hist.scatter_add_(1, bi, torch.ones_like(z))
        del bi, z

        p_sm = (hist + 1.0) / (V + NBINS)
        kl_arr[s:e] = (gauss.unsqueeze(0) * (log_gauss.unsqueeze(0)
                        - torch.log(p_sm))).sum(-1)
        del hist, p_sm

    result = {
        'H':    H_arr.cpu().numpy(),
        'Heff': Heff_arr.cpu().numpy(),
        'kl':   kl_arr.cpu().numpy(),
        'T':    T,
    }
    del logits, out; torch.cuda.empty_cache()
    return result


def aggregate(pt, H_override=None):
    """从 per-token 数据聚合为 sample 级指标. H_override 用于坍缩模拟."""
    H    = H_override if H_override is not None else pt['H']
    kl   = pt['kl']
    Heff = pt['Heff']
    T    = len(H)
    sH   = H.sum()
    if sH < eps or T < 3:
        return None

    H_bar = H.mean()
    low   = H < H_bar
    high  = H >= H_bar
    n_lo  = low.sum()
    n_hi  = high.sum()

    kl_lo = kl[low].mean()  if n_lo > 0 else 0.0
    kl_hi = kl[high].mean() if n_hi > 0 else 0.0
    kl_m  = kl.mean()

    w = H / (sH + eps)
    kl_Hw = float((w * kl).sum())

    r     = Heff / (H + eps)
    log_r = np.log(np.clip(r, eps, None))
    R_geo = float(np.exp((H * log_r).sum() / (sH + eps)))

    return {
        'kl_split':     float(kl_lo - kl_hi),    # 用户提案
        'neg_kl_split': float(kl_hi - kl_lo),     # 取负
        'neg_kl_mean':  float(-kl_m),
        'neg_kl_Hw':    float(-kl_Hw),
        'neg_kl_low':   float(-kl_lo),
        'neg_kl_high':  float(-kl_hi),
        'R_geo':        R_geo,
        'mean_H':       float(H_bar),
        'frac_low':     float(n_lo / T),
    }


@torch.no_grad()
def recompute_H_at_temp(logits_orig, tau, device):
    """给定原始 logits, 计算 τ 温度下的 H_t."""
    logits = logits_orig / tau
    T = logits.size(0)
    H = torch.empty(T, device=device)
    for s in range(0, T, 256):
        e = min(s + 256, T)
        lp = torch.log_softmax(logits[s:e], dim=-1)
        p  = lp.exp()
        H[s:e] = -(p * lp).sum(-1)
        del p, lp
    return H.cpu().numpy()


@torch.no_grad()
def extract_collapse(model, tok, prompt, response, temps=(1.0,0.7,0.5,0.3), device=DEV):
    """单次 forward: 对多个温度计算 H_t (KL 不变, 只需重算 H)."""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl  = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T = logits_orig.size(0)
    if T < 3:
        del out; torch.cuda.empty_cache()
        return None, None

    # 先提取 per-token kl (温度不变) 和 τ=1 的 H
    pt = extract_per_token.__wrapped__(model, tok, prompt, response, device) if hasattr(extract_per_token, '__wrapped__') else None
    # 手动计算 (避免重复 forward)
    bw = (BIN_HI - BIN_LO) / NBINS; V = logits_orig.size(1)
    centers = torch.linspace(BIN_LO + bw/2, BIN_HI - bw/2, NBINS, device=device)
    gauss = torch.exp(-0.5 * centers**2) / math.sqrt(2*math.pi) * bw
    gauss = gauss / gauss.sum()
    log_gauss = torch.log(gauss)

    kl_arr  = torch.empty(T, device=device)
    Heff_arr= torch.empty(T, device=device)
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T); chunk = logits_orig[s:e]; c = e - s
        lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
        Heff_arr[s:e] = -(p * p * lp).sum(-1)
        del p, lp
        mu = chunk.mean(-1, keepdim=True); cen = chunk - mu
        std = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
        z = cen / std; del chunk, cen, mu, std
        bi = ((z - BIN_LO) / bw).long().clamp(0, NBINS - 1)
        hist = torch.zeros(c, NBINS, device=device)
        hist.scatter_add_(1, bi, torch.ones_like(z)); del bi, z
        p_sm = (hist + 1.0) / (V + NBINS)
        kl_arr[s:e] = (gauss.unsqueeze(0) * (log_gauss.unsqueeze(0) - torch.log(p_sm))).sum(-1)
        del hist, p_sm

    kl_np   = kl_arr.cpu().numpy()
    Heff_np = Heff_arr.cpu().numpy()

    rows = []
    for tau in temps:
        H_np = recompute_H_at_temp(logits_orig, tau, device)
        pt_tau = {'H': H_np, 'Heff': Heff_np, 'kl': kl_np, 'T': T}
        m = aggregate(pt_tau)
        if m:
            m['tau'] = tau
            rows.append(m)

    del logits_orig, out; torch.cuda.empty_cache()
    return rows


# ═════════════════════════════════════════════

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

    KEYS = ['kl_split', 'neg_kl_split', 'neg_kl_mean', 'neg_kl_Hw',
            'neg_kl_low', 'neg_kl_high', 'R_geo']
    LABELS = {
        'kl_split':     'KL_lo−KL_hi',
        'neg_kl_split': '−(KL_lo−KL_hi)',
        'neg_kl_mean':  '−KL mean',
        'neg_kl_Hw':    '−KL H-wt',
        'neg_kl_low':   '−KL low-H',
        'neg_kl_high':  '−KL high-H',
        'R_geo':        'R_geo',
    }

    N_COLLAPSE = min(8, N_Q)
    temps = [1.0, 0.7, 0.5, 0.3]

    # ── Part 1: Collect ──
    print("\n== PART 1: 收集指标 ==")
    all_c = {k: [] for k in KEYS}
    all_w = {k: [] for k in KEYS}
    all_ac = {k: [] for k in KEYS}
    all_aw = {k: [] for k in KEYS}

    collapse_buf = {t: {k: [] for k in KEYS} for t in temps}

    for qi, res in enumerate(results):
        prompt = res['prompt']
        K = len(res['responses'])
        labels = [1 if r.get('is_correct', False) else 0 for r in res['responses']]

        group_vals = {k: [] for k in KEYS}

        for i, resp in enumerate(res['responses']):
            response_text = resp.get('response', '')

            # ── 正常指标 ──
            pt = extract_per_token(model, tok, prompt, response_text)
            if pt is None:
                for k in KEYS:
                    group_vals[k].append(0.0)
                continue
            m = aggregate(pt)
            if m is None:
                for k in KEYS:
                    group_vals[k].append(0.0)
                continue
            for k in KEYS:
                group_vals[k].append(m[k])

            # ── 坍缩模拟 (前 N_COLLAPSE 题) ──
            if qi < N_COLLAPSE:
                crow = extract_collapse(model, tok, prompt, response_text, temps=temps)
                if crow:
                    for row in crow:
                        t = row['tau']
                        for k in KEYS:
                            collapse_buf[t][k].append(row.get(k, 0.0))

        # GRPO advantage
        for k in KEYS:
            adv = grpo_adv(group_vals[k])
            for i in range(K):
                dst_v = all_c if labels[i] else all_w
                dst_a = all_ac if labels[i] else all_aw
                dst_v[k].append(group_vals[k][i])
                dst_a[k].append(adv[i])

        if (qi+1) % 5 == 0 or qi == N_Q-1:
            nc = sum(1 for l in labels if l)
            print(f"  [{qi+1}/{N_Q}]")

    # ── Part 2: Discrimination ──
    print("\n" + "="*110)
    print("  PART 2: 区分度 + GRPO Advantage")
    print("="*110)
    print(f"  {'指标':<16s} │ {'Cohen d':>9s} │ {'正确均值':>12s} │ {'错误均值':>12s} │"
          f" {'Adv Gap':>9s} │ {'P(+|C)':>7s} │ {'P(+|W)':>7s} │ {'ΔP':>7s}")
    print("  " + "─"*95)

    disc = {}
    for k in KEYS:
        d   = cohens_d(all_c[k], all_w[k])
        mc  = np.mean(all_c[k])
        mw  = np.mean(all_w[k])
        gap = np.mean(all_ac[k]) - np.mean(all_aw[k])
        pc  = np.mean(np.array(all_ac[k]) > 0)
        pw  = np.mean(np.array(all_aw[k]) > 0)
        disc[k] = {'d': d, 'gap': gap, 'pc': pc, 'pw': pw}
        print(f"  {LABELS[k]:<16s} │ {d:>+9.4f} │ {mc:>12.6f} │ {mw:>12.6f} │"
              f" {gap:>+9.4f} │ {pc:>7.3f} │ {pw:>7.3f} │ {pc-pw:>+7.3f}")

    # ── Part 3: Collapse ──
    print("\n" + "="*110)
    print("  PART 3: 防坍缩模拟")
    print("="*110)
    print(f"  {'指标':<16s} │ {'τ=1.0':>10s} │ {'τ=0.7':>10s} │ {'τ=0.5':>10s} │ {'τ=0.3':>10s} │ {'Delta':>8s}")
    print("  " + "─"*72)

    col_delta = {}
    for k in KEYS + ['mean_H', 'frac_low']:
        vs = []
        for t in temps:
            arr = collapse_buf[t].get(k, [])
            vs.append(np.mean(arr) if arr else 0.0)
        delta = (vs[-1] - vs[0]) / (abs(vs[0]) + 1e-12) * 100
        col_delta[k] = delta
        lb = LABELS.get(k, k)
        print(f"  {lb:<16s} │ {vs[0]:>10.6f} │ {vs[1]:>10.6f} │ {vs[2]:>10.6f} │ {vs[3]:>10.6f} │ {delta:>+7.1f}%")

    # ── Part 4: Summary ──
    print("\n" + "="*110)
    print("  ═══ 最终汇总 ═══")
    print("="*110)
    ranked = sorted(KEYS, key=lambda k: disc[k]['d'], reverse=True)
    for rank, k in enumerate(ranked, 1):
        d     = disc[k]['d']
        delta = col_delta.get(k, float('nan'))
        dp    = disc[k]['pc'] - disc[k]['pw']
        ideal = "★ IDEAL ★" if d > 0 and abs(delta) < 5 else \
                "✓ d>0" if d > 0 else "✗"
        print(f"  #{rank} {LABELS[k]:<16s}: d={d:>+.4f}, delta={delta:>+.1f}%, ΔP={dp:>+.3f}  {ideal}")

    # ── Part 5: Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    fig.suptitle("KL Split Reward Validation", fontsize=15, fontweight='bold')

    COLORS = dict(zip(KEYS, ['#E74C3C','#9B59B6','#3498DB','#2ECC71','#E67E22','#1ABC9C','#95A5A6']))

    # (a) Cohen's d
    ax = axes[0]
    ds = [disc[k]['d'] for k in KEYS]
    bars = ax.bar(range(len(KEYS)), ds, color=[COLORS[k] for k in KEYS],
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    for b, d in zip(bars, ds):
        ax.text(b.get_x()+b.get_width()/2, d + 0.02*(1 if d>=0 else -1),
                f'{d:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([LABELS[k] for k in KEYS], fontsize=7, rotation=30, ha='right')
    ax.set_ylabel("Cohen's d"); ax.set_title("Discrimination"); ax.axhline(0, color='k')
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Collapse delta
    ax1 = axes[1]
    deltas = [col_delta.get(k, 0) for k in KEYS]
    c1 = ['#2ECC71' if abs(d)<5 else ('#E74C3C' if d>0 else '#3498DB') for d in deltas]
    bars1 = ax1.bar(range(len(KEYS)), deltas, color=c1, edgecolor='black', linewidth=0.8, alpha=0.85)
    for b, d in zip(bars1, deltas):
        ax1.text(b.get_x()+b.get_width()/2, d + 1*(1 if d>=0 else -1),
                 f'{d:+.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(len(KEYS)))
    ax1.set_xticklabels([LABELS[k] for k in KEYS], fontsize=7, rotation=30, ha='right')
    ax1.set_ylabel("Delta (%)"); ax1.set_title("Anti-collapse (green=|Δ|<5%)")
    ax1.axhline(0, color='k', linewidth=2); ax1.grid(True, alpha=0.3, axis='y')

    # (c) Trajectory
    ax2 = axes[2]
    for k in KEYS:
        vs = [np.mean(collapse_buf[t].get(k, [0])) for t in temps]
        base = vs[0] if abs(vs[0]) > eps else 1.0
        norm = [v/abs(base) for v in vs]
        ax2.plot(temps, norm, color=COLORS[k], linewidth=2.5, marker='o',
                 markersize=8, label=LABELS[k], alpha=0.85)
    ax2.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax2.invert_xaxis()
    ax2.set_xlabel("Temperature τ"); ax2.set_ylabel("Normalized reward")
    ax2.set_title("Sharpening trajectory"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'kl_split_validation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
