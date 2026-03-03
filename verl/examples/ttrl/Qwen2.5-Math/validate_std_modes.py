#!/usr/bin/env python3
"""
验证三种标准化方式对 -KL(G||emp) 的影响:

  1. per-token σ (当前): 每个 token 用自己的 σ_t 标准化
  2. global σ:          全序列的 σ̄ = mean(σ_t) 作为统一分母
  3. group σ:           low-H 用 σ̄_low, high-H 用 σ̄_high

对每种标准化, 计算:
  -KL mean, -KL low-H, -KL high-H
  d (区分度), delta (防坍缩), ΔP (GRPO信号)
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
def extract_all_modes(model, tok, prompt, response, device=DEV):
    """
    单次 forward, 同时计算三种标准化模式的 KL.
    返回 per-token H, Heff, 和三种模式的 kl 数组.
    """
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

    # ---- Phase 1: 计算 H, Heff, per-token σ_t ----
    H_arr    = torch.empty(T, device=device)
    Heff_arr = torch.empty(T, device=device)
    sigma_arr = torch.empty(T, device=device)  # per-token σ_t
    # 存储 centered logits 用于后续不同标准化
    centered_chunks = []

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        chunk = logits[s:e].float(); c = e - s

        # entropy
        lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
        H_arr[s:e]    = -(p * lp).sum(-1)
        Heff_arr[s:e] = -(p * p * lp).sum(-1)
        del p, lp

        # per-token centering and σ
        mu  = chunk.mean(-1, keepdim=True)
        cen = chunk - mu
        std_t = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
        sigma_arr[s:e] = std_t.squeeze(-1)
        centered_chunks.append(cen.cpu())  # save for reuse
        del chunk, mu

    # ---- Phase 2: 计算三种标准化的 KL ----
    H_np = H_arr.cpu().numpy()
    H_bar = H_np.mean()
    low_mask  = H_np < H_bar
    high_mask = H_np >= H_bar

    sigma_np = sigma_arr.cpu().numpy()
    sigma_global = sigma_np.mean()
    sigma_low  = sigma_np[low_mask].mean() if low_mask.sum() > 0 else sigma_global
    sigma_high = sigma_np[high_mask].mean() if high_mask.sum() > 0 else sigma_global

    def compute_kl_with_sigma(cen_list, sigma_vec, T, V, device):
        """给定 centered logits 和每个 token 的 sigma, 计算 KL(G||emp)."""
        kl = torch.empty(T, device=device)
        idx = 0
        for cen_cpu in cen_list:
            cen = cen_cpu.to(device)
            c = cen.size(0)
            sig = sigma_vec[idx:idx+c].unsqueeze(-1)  # (c, 1)
            z = cen / sig  # standardize
            del cen

            bi = ((z - BIN_LO) / bw).long().clamp(0, NBINS - 1)
            hist = torch.zeros(c, NBINS, device=device)
            hist.scatter_add_(1, bi, torch.ones_like(z))
            del bi, z

            p_sm = (hist + 1.0) / (V + NBINS)
            kl[idx:idx+c] = (gauss.unsqueeze(0) * (log_gauss.unsqueeze(0)
                              - torch.log(p_sm))).sum(-1)
            del hist, p_sm
            idx += c
        return kl.cpu().numpy()

    # Mode 1: per-token σ (current)
    kl_pertoken = compute_kl_with_sigma(
        centered_chunks,
        sigma_arr.to(device),
        T, V, device
    )

    # Mode 2: global σ (全序列均值)
    sigma_global_vec = torch.full((T,), sigma_global, device=device)
    kl_global = compute_kl_with_sigma(
        centered_chunks,
        sigma_global_vec,
        T, V, device
    )

    # Mode 3: group σ (low-H 用 σ_low, high-H 用 σ_high)
    sigma_group_np = np.where(low_mask, sigma_low, sigma_high)
    sigma_group_vec = torch.tensor(sigma_group_np, device=device, dtype=torch.float32)
    kl_group = compute_kl_with_sigma(
        centered_chunks,
        sigma_group_vec,
        T, V, device
    )

    del centered_chunks, sigma_arr
    del logits, out; torch.cuda.empty_cache()

    return {
        'H': H_np,
        'Heff': Heff_arr.cpu().numpy(),
        'sigma': sigma_np,
        'kl_pertoken': kl_pertoken,
        'kl_global': kl_global,
        'kl_group': kl_group,
        'T': T,
    }


@torch.no_grad()
def extract_collapse_modes(model, tok, prompt, response, temps=(1.0, 0.7, 0.5, 0.3), device=DEV):
    """对多个温度, 用三种标准化计算 KL."""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl  = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T, V = logits_orig.shape
    if T < 3:
        del out; torch.cuda.empty_cache()
        return None

    bw = (BIN_HI - BIN_LO) / NBINS
    centers = torch.linspace(BIN_LO + bw/2, BIN_HI - bw/2, NBINS, device=device)
    gauss = torch.exp(-0.5 * centers**2) / math.sqrt(2*math.pi) * bw
    gauss = gauss / gauss.sum()
    log_gauss = torch.log(gauss)

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

        H_np = H_arr.cpu().numpy()
        Heff_np = Heff_arr.cpu().numpy()
        sigma_np = sigma_arr.cpu().numpy()
        H_bar = H_np.mean()
        low_mask  = H_np < H_bar
        high_mask = H_np >= H_bar

        sigma_global = sigma_np.mean()
        sigma_low  = sigma_np[low_mask].mean() if low_mask.sum() > 0 else sigma_global
        sigma_high = sigma_np[high_mask].mean() if high_mask.sum() > 0 else sigma_global

        def kl_with_sigma(sigma_vec_np):
            sig_t = torch.tensor(sigma_vec_np, device=device, dtype=torch.float32)
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

        kl_pt = kl_with_sigma(sigma_np)                      # per-token
        kl_gl = kl_with_sigma(np.full(T, sigma_global))      # global
        grp_sig = np.where(low_mask, sigma_low, sigma_high)
        kl_gp = kl_with_sigma(grp_sig)                       # group

        del centered_chunks

        # aggregate for each mode
        sH = H_np.sum()
        r = Heff_np / (H_np + eps)
        log_r = np.log(np.clip(r, eps, None))
        R_geo = float(np.exp((H_np * log_r).sum() / (sH + eps)))

        row = {'tau': tau, 'R_geo': R_geo, 'mean_H': float(H_bar)}
        for mode, kl in [('pt', kl_pt), ('gl', kl_gl), ('gp', kl_gp)]:
            n_lo = low_mask.sum(); n_hi = high_mask.sum()
            row[f'neg_kl_mean_{mode}'] = float(-kl.mean())
            row[f'neg_kl_low_{mode}']  = float(-kl[low_mask].mean()) if n_lo > 0 else 0.0
            row[f'neg_kl_high_{mode}'] = float(-kl[high_mask].mean()) if n_hi > 0 else 0.0
        rows.append(row)

    del logits_orig, out; torch.cuda.empty_cache()
    return rows


def aggregate_modes(pt_data):
    """从 per-token 数据聚合三种模式的 sample 级指标."""
    H    = pt_data['H']
    Heff = pt_data['Heff']
    T    = len(H)
    sH   = H.sum()
    if sH < eps or T < 3:
        return None

    H_bar = H.mean()
    low  = H < H_bar
    high = H >= H_bar

    r = Heff / (H + eps)
    log_r = np.log(np.clip(r, eps, None))
    R_geo = float(np.exp((H * log_r).sum() / (sH + eps)))

    result = {'R_geo': R_geo, 'mean_H': float(H_bar)}

    for mode in ['pertoken', 'global', 'group']:
        kl = pt_data[f'kl_{mode}']
        n_lo = low.sum(); n_hi = high.sum()
        sfx = {'pertoken': 'pt', 'global': 'gl', 'group': 'gp'}[mode]
        result[f'neg_kl_mean_{sfx}'] = float(-kl.mean())
        result[f'neg_kl_low_{sfx}']  = float(-kl[low].mean()) if n_lo > 0 else 0.0
        result[f'neg_kl_high_{sfx}'] = float(-kl[high].mean()) if n_hi > 0 else 0.0

    return result


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

    # 9 个指标 = 3 modes × 3 variants (mean, low-H, high-H)  + R_geo
    MODES = ['pt', 'gl', 'gp']
    VARIANTS = ['mean', 'low', 'high']
    KEYS = [f'neg_kl_{v}_{m}' for m in MODES for v in VARIANTS] + ['R_geo']

    LABELS = {}
    for m in MODES:
        mode_name = {'pt': 'per-tok', 'gl': 'global', 'gp': 'group'}[m]
        LABELS[f'neg_kl_mean_{m}'] = f'-KL mean ({mode_name})'
        LABELS[f'neg_kl_low_{m}']  = f'-KL low-H ({mode_name})'
        LABELS[f'neg_kl_high_{m}'] = f'-KL high-H ({mode_name})'
    LABELS['R_geo'] = 'R_geo'

    N_COLLAPSE = min(8, N_Q)
    temps = [1.0, 0.7, 0.5, 0.3]

    # ── Collect ──
    print("\n== 收集指标 ==")
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

            pt_data = extract_all_modes(model, tok, prompt, response_text)
            if pt_data is None:
                for k in KEYS:
                    group_vals[k].append(0.0)
                continue
            m = aggregate_modes(pt_data)
            if m is None:
                for k in KEYS:
                    group_vals[k].append(0.0)
                continue
            for k in KEYS:
                group_vals[k].append(m.get(k, 0.0))

            # collapse simulation
            if qi < N_COLLAPSE:
                crow = extract_collapse_modes(model, tok, prompt, response_text, temps=temps)
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
            print(f"  [{qi+1}/{N_Q}]")

    # ── Results ──
    print("\n" + "="*120)
    print("  区分度 + GRPO Advantage + 防坍缩")
    print("="*120)
    print(f"  {'指标':<26s} │ {'d':>8s} │ {'delta':>8s} │ {'ΔP':>7s} │ {'σ_w':>8s} │ 评估")
    print("  " + "─"*80)

    disc = {}
    col_delta = {}
    for k in KEYS:
        d  = cohens_d(all_c[k], all_w[k])
        pc = np.mean(np.array(all_ac[k]) > 0)
        pw = np.mean(np.array(all_aw[k]) > 0)
        dp = pc - pw
        sw = np.std(all_c[k] + all_w[k])
        disc[k] = {'d': d, 'dp': dp, 'sw': sw}

        # collapse delta
        vs = []
        for t in temps:
            arr = collapse_buf[t].get(k, [])
            vs.append(np.mean(arr) if arr else 0.0)
        delta = (vs[-1] - vs[0]) / (abs(vs[0]) + 1e-12) * 100
        col_delta[k] = delta

        # evaluation
        tags = []
        if d > 0 and delta < 0 and dp > 0:
            tags.append("★ IDEAL")
        else:
            if d > 0: tags.append("d>0")
            if delta < 0: tags.append("anti-col")
            if dp > 0: tags.append("ΔP>0")

        print(f"  {LABELS[k]:<26s} │ {d:>+8.3f} │ {delta:>+7.1f}% │ {dp:>+7.3f} │ {sw:>8.4f} │ {', '.join(tags)}")

    # ── 按模式分组汇总 ──
    print("\n" + "="*120)
    print("  按标准化模式汇总 (仅 -KL low-H, 最关键的指标)")
    print("="*120)
    for m in MODES:
        mode_name = {'pt': 'per-token σ (当前)', 'gl': 'global σ (全序列)', 'gp': 'group σ (分组)'}[m]
        k = f'neg_kl_low_{m}'
        d, delta, dp = disc[k]['d'], col_delta[k], disc[k]['dp']
        print(f"  {mode_name:<25s}: d={d:>+.3f}, delta={delta:>+.1f}%, ΔP={dp:>+.3f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    fig.suptitle("Standardization Mode Comparison", fontsize=15, fontweight='bold')

    colors_mode = {'pt': '#3498DB', 'gl': '#E74C3C', 'gp': '#2ECC71'}
    markers_var = {'mean': 'o', 'low': 's', 'high': '^'}

    # (a) d vs delta scatter
    ax = axes[0]
    for k in KEYS:
        if k == 'R_geo':
            ax.scatter(col_delta[k], disc[k]['d'], marker='D', s=120, c='gray',
                       edgecolors='black', zorder=5, label='R_geo')
            continue
        parts = k.replace('neg_kl_', '').split('_')
        var, mode = parts[0], parts[1]
        ax.scatter(col_delta[k], disc[k]['d'], marker=markers_var[var], s=100,
                   c=colors_mode[mode], edgecolors='black', zorder=5,
                   label=LABELS[k])
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.axvline(0, color='k', ls='-', alpha=0.3)
    ax.fill_betweenx([-0.5, 1.5], -50, 0, alpha=0.05, color='green')
    ax.set_xlabel("Collapse delta (%)")
    ax.set_ylabel("Cohen's d")
    ax.set_title("d vs delta (左下=ideal)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (b) Bar chart of d
    ax1 = axes[1]
    ds = [disc[k]['d'] for k in KEYS]
    colors = []
    for k in KEYS:
        if k == 'R_geo': colors.append('gray')
        else:
            mode = k.split('_')[-1]
            colors.append(colors_mode.get(mode, 'gray'))
    bars = ax1.bar(range(len(KEYS)), ds, color=colors, edgecolor='black', alpha=0.85)
    for b, d_val in zip(bars, ds):
        ax1.text(b.get_x()+b.get_width()/2, d_val + 0.02*(1 if d_val>=0 else -1),
                 f'{d_val:.3f}', ha='center', fontsize=7, fontweight='bold')
    ax1.set_xticks(range(len(KEYS)))
    ax1.set_xticklabels([LABELS[k] for k in KEYS], fontsize=6, rotation=45, ha='right')
    ax1.set_ylabel("Cohen's d")
    ax1.set_title("Discrimination")
    ax1.axhline(0, color='k')
    ax1.grid(True, alpha=0.3, axis='y')

    # (c) Bar chart of delta
    ax2 = axes[2]
    deltas = [col_delta.get(k, 0) for k in KEYS]
    c2 = ['#2ECC71' if d < 0 else ('#E74C3C' if d > 5 else '#FFD700') for d in deltas]
    bars2 = ax2.bar(range(len(KEYS)), deltas, color=c2, edgecolor='black', alpha=0.85)
    for b, d_val in zip(bars2, deltas):
        ax2.text(b.get_x()+b.get_width()/2, d_val + 1*(1 if d_val>=0 else -1),
                 f'{d_val:+.1f}%', ha='center', fontsize=7, fontweight='bold')
    ax2.set_xticks(range(len(KEYS)))
    ax2.set_xticklabels([LABELS[k] for k in KEYS], fontsize=6, rotation=45, ha='right')
    ax2.set_ylabel("Delta (%)")
    ax2.set_title("Collapse delta (green=anti-collapse)")
    ax2.axhline(0, color='k', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'std_mode_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
