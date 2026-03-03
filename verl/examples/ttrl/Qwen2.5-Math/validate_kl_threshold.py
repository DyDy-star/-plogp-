#!/usr/bin/env python3
"""
验证按 KL 阈值 vs 按 H 阈值分割的 -KL 奖励:

  按 H 分割 (已有):
    -KL low-H:  mean(-KL) for H_t < H̄
    -KL high-H: mean(-KL) for H_t ≥ H̄

  按 KL 分割 (新增):
    -KL low-KL:  mean(-KL) for KL_t < KL̄
    -KL high-KL: mean(-KL) for KL_t ≥ KL̄

  对三种标准化模式 (per-token / global / group) 各自测试.
"""

import json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
def extract(model, tok, prompt, response, device=DEV):
    """单次 forward: 提取 per-token H, Heff, σ, centered logits."""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl  = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T, V = logits.shape
    if T < 3:
        del out; torch.cuda.empty_cache()
        return None

    H_arr    = torch.empty(T, device=device)
    Heff_arr = torch.empty(T, device=device)
    sigma_arr = torch.empty(T, device=device)
    centered_chunks = []

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        chunk = logits[s:e]; c = e - s
        lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
        H_arr[s:e]    = -(p * lp).sum(-1)
        Heff_arr[s:e] = -(p * p * lp).sum(-1)
        del p, lp
        mu = chunk.mean(-1, keepdim=True)
        cen = chunk - mu
        std_t = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
        sigma_arr[s:e] = std_t.squeeze(-1)
        centered_chunks.append(cen.cpu())
        del chunk, mu

    del logits, out; torch.cuda.empty_cache()
    return {
        'H': H_arr.cpu().numpy(),
        'Heff': Heff_arr.cpu().numpy(),
        'sigma': sigma_arr.cpu().numpy(),
        'centered': centered_chunks,
        'T': T, 'V': V,
    }


def compute_kl_batch(centered_chunks, sigma_vec, T, V, device):
    """用给定 sigma 计算每个 token 的 KL(G||emp)."""
    bw = (BIN_HI - BIN_LO) / NBINS
    centers = torch.linspace(BIN_LO + bw/2, BIN_HI - bw/2, NBINS, device=device)
    gauss = torch.exp(-0.5 * centers**2) / math.sqrt(2*math.pi) * bw
    gauss = gauss / gauss.sum()
    log_gauss = torch.log(gauss)

    sig_t = torch.tensor(sigma_vec, device=device, dtype=torch.float32)
    kl = torch.empty(T, device=device)
    idx = 0
    for cen_cpu in centered_chunks:
        cen = cen_cpu.to(device); c = cen.size(0)
        z = cen / sig_t[idx:idx+c].unsqueeze(-1)
        del cen
        bi = ((z - BIN_LO) / bw).long().clamp(0, NBINS - 1)
        hist = torch.zeros(c, NBINS, device=device)
        hist.scatter_add_(1, bi, torch.ones_like(z)); del bi, z
        p_sm = (hist + 1.0) / (V + NBINS)
        kl[idx:idx+c] = (gauss.unsqueeze(0) * (log_gauss.unsqueeze(0)
                          - torch.log(p_sm))).sum(-1)
        del hist, p_sm; idx += c
    return kl.cpu().numpy()


def aggregate(pt, tau_logits_scale=None):
    """聚合 per-token → sample 级指标 (所有模式 × 所有分割方式)."""
    H = pt['H']; Heff = pt['Heff']; sigma = pt['sigma']
    T = pt['T']; V = pt['V']
    cen = pt['centered']
    device = DEV

    if T < 3 or H.sum() < eps:
        return None

    # ---- 三种标准化的 KL ----
    H_bar = H.mean()
    low_H  = H < H_bar
    high_H = H >= H_bar

    sig_global = sigma.mean()
    sig_low_H  = sigma[low_H].mean() if low_H.sum() > 0 else sig_global
    sig_high_H = sigma[high_H].mean() if high_H.sum() > 0 else sig_global

    modes = {
        'pt': sigma,                                            # per-token
        'gl': np.full(T, sig_global),                           # global
        'gp': np.where(low_H, sig_low_H, sig_high_H),          # group (by H)
    }

    result = {'mean_H': float(H_bar)}

    # R_geo
    sH = H.sum()
    r = Heff / (H + eps)
    log_r = np.log(np.clip(r, eps, None))
    result['R_geo'] = float(np.exp((H * log_r).sum() / (sH + eps)))

    for mode_name, sig_vec in modes.items():
        kl = compute_kl_batch(cen, sig_vec, T, V, device)

        kl_bar = kl.mean()
        low_KL  = kl < kl_bar    # 按 KL 值分割
        high_KL = kl >= kl_bar

        n_lo_H = low_H.sum();  n_hi_H = high_H.sum()
        n_lo_K = low_KL.sum(); n_hi_K = high_KL.sum()

        sfx = mode_name  # pt / gl / gp

        # 按 H 分割
        result[f'neg_kl_mean_{sfx}'] = float(-kl.mean())
        result[f'neg_kl_lowH_{sfx}'] = float(-kl[low_H].mean()) if n_lo_H > 0 else 0.0
        result[f'neg_kl_hiH_{sfx}']  = float(-kl[high_H].mean()) if n_hi_H > 0 else 0.0

        # 按 KL 分割
        result[f'neg_kl_lowK_{sfx}'] = float(-kl[low_KL].mean()) if n_lo_K > 0 else 0.0
        result[f'neg_kl_hiK_{sfx}']  = float(-kl[high_KL].mean()) if n_hi_K > 0 else 0.0

    return result


@torch.no_grad()
def extract_collapse(model, tok, prompt, response, temps=(1.0, 0.7, 0.5, 0.3), device=DEV):
    """多温度坍缩模拟."""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl  = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T, V = logits_orig.shape
    if T < 3:
        del out; torch.cuda.empty_cache()
        return None

    rows = []
    for tau in temps:
        logits = logits_orig / tau
        H_arr    = torch.empty(T, device=device)
        Heff_arr = torch.empty(T, device=device)
        sigma_arr = torch.empty(T, device=device)
        centered_chunks = []

        for s in range(0, T, CHUNK):
            e = min(s + CHUNK, T)
            chunk = logits[s:e]; c = e - s
            lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
            H_arr[s:e]    = -(p * lp).sum(-1)
            Heff_arr[s:e] = -(p * p * lp).sum(-1)
            del p, lp
            mu = chunk.mean(-1, keepdim=True)
            cen = chunk - mu
            std_t = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
            sigma_arr[s:e] = std_t.squeeze(-1)
            centered_chunks.append(cen.cpu())
            del chunk, mu

        pt_data = {
            'H': H_arr.cpu().numpy(),
            'Heff': Heff_arr.cpu().numpy(),
            'sigma': sigma_arr.cpu().numpy(),
            'centered': centered_chunks,
            'T': T, 'V': V,
        }
        m = aggregate(pt_data)
        if m:
            m['tau'] = tau
            rows.append(m)
        del centered_chunks

    del logits_orig, out; torch.cuda.empty_cache()
    return rows


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

    # 指标定义
    MODES = ['pt', 'gl', 'gp']
    SPLITS = ['mean', 'lowH', 'hiH', 'lowK', 'hiK']
    KEYS = [f'neg_kl_{sp}_{m}' for m in MODES for sp in SPLITS] + ['R_geo']

    MODE_NAMES = {'pt': 'per-tok', 'gl': 'global', 'gp': 'group'}
    SPLIT_NAMES = {'mean': 'mean', 'lowH': 'low-H', 'hiH': 'high-H', 'lowK': 'low-KL', 'hiK': 'high-KL'}
    LABELS = {}
    for m in MODES:
        for sp in SPLITS:
            LABELS[f'neg_kl_{sp}_{m}'] = f'-KL {SPLIT_NAMES[sp]} ({MODE_NAMES[m]})'
    LABELS['R_geo'] = 'R_geo'

    N_COLLAPSE = min(8, N_Q)
    temps = [1.0, 0.7, 0.5, 0.3]

    # ── Collect ──
    print("\n== 收集指标 ==")
    all_c  = {k: [] for k in KEYS}
    all_w  = {k: [] for k in KEYS}
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
            pt_data = extract(model, tok, prompt, response_text)
            if pt_data is None:
                for k in KEYS: group_vals[k].append(0.0)
                continue
            m = aggregate(pt_data)
            if m is None:
                for k in KEYS: group_vals[k].append(0.0)
                continue
            for k in KEYS:
                group_vals[k].append(m.get(k, 0.0))

            if qi < N_COLLAPSE:
                crow = extract_collapse(model, tok, prompt, response_text, temps=temps)
                if crow:
                    for row in crow:
                        t = row['tau']
                        for k in KEYS:
                            collapse_buf[t][k].append(row.get(k, 0.0))

        for k in KEYS:
            adv = grpo_adv(group_vals[k])
            for i in range(K):
                dst_v = all_c if labels[i] else all_w
                dst_a = all_ac if labels[i] else all_aw
                dst_v[k].append(group_vals[k][i])
                dst_a[k].append(adv[i])

        if (qi+1) % 5 == 0 or qi == N_Q-1:
            print(f"  [{qi+1}/{N_Q}]")

    # ── Results ──
    print("\n" + "="*130)
    print("  完整结果: 按 H 分割 vs 按 KL 分割, 三种标准化模式")
    print("="*130)

    disc = {}; col_delta = {}
    for m in MODES:
        mode_name = MODE_NAMES[m]
        print(f"\n  ── {mode_name} σ ──")
        print(f"  {'指标':<18s} │ {'d':>8s} │ {'delta':>8s} │ {'ΔP':>7s} │ 评估")
        print("  " + "─"*60)

        for sp in SPLITS:
            k = f'neg_kl_{sp}_{m}'
            d  = cohens_d(all_c[k], all_w[k])
            pc = np.mean(np.array(all_ac[k]) > 0)
            pw = np.mean(np.array(all_aw[k]) > 0)
            dp = pc - pw
            disc[k] = {'d': d, 'dp': dp}

            vs = [np.mean(collapse_buf[t].get(k, [0])) for t in temps]
            delta = (vs[-1] - vs[0]) / (abs(vs[0]) + 1e-12) * 100
            col_delta[k] = delta

            tags = []
            if d > 0 and delta < 0 and dp > 0: tags.append("★ IDEAL")
            else:
                if d > 0: tags.append("d>0")
                if delta < 0: tags.append("anti-col")
                if dp > 0: tags.append("ΔP>0")

            label = SPLIT_NAMES[sp]
            print(f"  {label:<18s} │ {d:>+8.3f} │ {delta:>+7.1f}% │ {dp:>+7.3f} │ {', '.join(tags)}")

    # R_geo baseline
    k = 'R_geo'
    d = cohens_d(all_c[k], all_w[k])
    pc = np.mean(np.array(all_ac[k]) > 0)
    pw = np.mean(np.array(all_aw[k]) > 0)
    dp = pc - pw
    disc[k] = {'d': d, 'dp': dp}
    vs = [np.mean(collapse_buf[t].get(k, [0])) for t in temps]
    delta = (vs[-1] - vs[0]) / (abs(vs[0]) + 1e-12) * 100
    col_delta[k] = delta
    print(f"\n  ── 对照 ──")
    print(f"  {'R_geo':<18s} │ {d:>+8.3f} │ {delta:>+7.1f}% │ {dp:>+7.3f}")

    # ── 关键对比: low-H vs low-KL, high-H vs high-KL ──
    print("\n" + "="*130)
    print("  ═══ 核心对比: H 分割 vs KL 分割 ═══")
    print("="*130)
    print(f"  {'标准化':<12s} │ {'分割方式':<12s} │ {'d':>8s} │ {'delta':>8s} │ {'ΔP':>7s} │ 评估")
    print("  " + "─"*70)
    for m in MODES:
        for sp in ['lowH', 'hiH', 'lowK', 'hiK']:
            k = f'neg_kl_{sp}_{m}'
            d, delta, dp = disc[k]['d'], col_delta[k], disc[k]['dp']
            tags = []
            if d > 0 and delta < 0 and dp > 0: tags.append("★ IDEAL")
            else:
                if d > 0: tags.append("d>0")
                if delta < 0: tags.append("anti-col")
                if dp > 0: tags.append("ΔP>0")
            print(f"  {MODE_NAMES[m]:<12s} │ {SPLIT_NAMES[sp]:<12s} │ {d:>+8.3f} │ {delta:>+7.1f}% │ {dp:>+7.3f} │ {', '.join(tags)}")
        print("  " + "─"*70)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    fig.suptitle("H-split vs KL-split Comparison", fontsize=15, fontweight='bold')

    # Focus on the 4 key splits × 3 modes = 12 metrics
    focus_keys = [f'neg_kl_{sp}_{m}' for m in MODES for sp in ['lowH', 'hiH', 'lowK', 'hiK']]

    colors = {
        'lowH': '#3498DB', 'hiH': '#E74C3C',
        'lowK': '#2ECC71', 'hiK': '#E67E22',
    }

    # (a) d vs delta scatter
    ax = axes[0]
    for k in focus_keys:
        parts = k.replace('neg_kl_', '').split('_')
        sp, m = parts[0], parts[1]
        marker = {'pt': 'o', 'gl': 's', 'gp': '^'}[m]
        ax.scatter(col_delta[k], disc[k]['d'], marker=marker, s=120,
                   c=colors[sp], edgecolors='black', zorder=5,
                   label=LABELS[k])
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.axvline(0, color='k', ls='-', alpha=0.3)
    ax.fill_betweenx([-0.5, 1.5], -50, 0, alpha=0.05, color='green')
    ax.set_xlabel("Collapse delta (%)")
    ax.set_ylabel("Cohen's d")
    ax.set_title("d vs delta")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    # (b) Bar: d
    ax1 = axes[1]
    ds = [disc[k]['d'] for k in focus_keys]
    clrs = [colors[k.replace('neg_kl_', '').split('_')[0]] for k in focus_keys]
    bars = ax1.bar(range(len(focus_keys)), ds, color=clrs, edgecolor='black', alpha=0.85)
    for b, d_val in zip(bars, ds):
        ax1.text(b.get_x()+b.get_width()/2, d_val + 0.02*(1 if d_val>=0 else -1),
                 f'{d_val:.3f}', ha='center', fontsize=6, fontweight='bold')
    ax1.set_xticks(range(len(focus_keys)))
    ax1.set_xticklabels([LABELS[k] for k in focus_keys], fontsize=5, rotation=60, ha='right')
    ax1.set_ylabel("Cohen's d"); ax1.set_title("Discrimination")
    ax1.axhline(0, color='k'); ax1.grid(True, alpha=0.3, axis='y')

    # (c) Bar: ΔP
    ax2 = axes[2]
    dps = [disc[k]['dp'] for k in focus_keys]
    bars2 = ax2.bar(range(len(focus_keys)), dps, color=clrs, edgecolor='black', alpha=0.85)
    for b, d_val in zip(bars2, dps):
        ax2.text(b.get_x()+b.get_width()/2, d_val + 0.003*(1 if d_val>=0 else -1),
                 f'{d_val:+.3f}', ha='center', fontsize=6, fontweight='bold')
    ax2.set_xticks(range(len(focus_keys)))
    ax2.set_xticklabels([LABELS[k] for k in focus_keys], fontsize=5, rotation=60, ha='right')
    ax2.set_ylabel("ΔP"); ax2.set_title("GRPO Signal")
    ax2.axhline(0, color='k', linewidth=2); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'kl_threshold_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
