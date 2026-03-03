#!/usr/bin/env python3
"""
验证 1/KL 权重的高熵 token 变体:

  用户提案:
    只对高熵 token (H_t ≥ H̄) 计算 1/KL,
    标准化 σ 使用高熵 token 的平均 σ_high (而非全序列).

  对比:
    1. 1/KL_gp (all)       — 全部 token, group σ (前次最优)
    2. 1/KL_hiH (σ_high)   — 只高熵 token, σ = σ_high  ← 用户提案
    3. 1/KL_loH (σ_low)    — 只低熵 token, σ = σ_low
    4. 1/KL_hiH (σ_global) — 只高熵 token, σ = 全序列均值

  每种在 unsup 和 weighted-correctness 模式下测量 d, ΔP, delta.
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
    if len(a) < 2 or len(b) < 2: return 0.0
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/(sp+1e-12)


def grpo_adv(vals):
    r = np.asarray(vals, float); s = r.std()
    return (r-r.mean())/(s+eps) if s > eps else np.zeros_like(r)


def kl_batch(centered_chunks, sigma_vec, T, V, device):
    bw = (BIN_HI-BIN_LO)/NBINS
    centers = torch.linspace(BIN_LO+bw/2, BIN_HI-bw/2, NBINS, device=device)
    gauss = torch.exp(-0.5*centers**2)/math.sqrt(2*math.pi)*bw
    gauss = gauss/gauss.sum(); log_gauss = torch.log(gauss)
    sig_t = torch.tensor(sigma_vec, device=device, dtype=torch.float32)
    kl = torch.empty(T, device=device); idx = 0
    for cc in centered_chunks:
        c2 = cc.to(device); c = c2.size(0)
        z = c2/sig_t[idx:idx+c].unsqueeze(-1); del c2
        bi = ((z-BIN_LO)/bw).long().clamp(0,NBINS-1)
        h2 = torch.zeros(c,NBINS,device=device)
        h2.scatter_add_(1,bi,torch.ones_like(z)); del bi,z
        ps = (h2+1.0)/(V+NBINS)
        kl[idx:idx+c] = (gauss.unsqueeze(0)*(log_gauss.unsqueeze(0)-torch.log(ps))).sum(-1)
        del h2,ps; idx+=c
    return kl.cpu().numpy()


@torch.no_grad()
def extract(model, tok, prompt, response, device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl  = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T, V = logits.shape
    if T < 3:
        del out; torch.cuda.empty_cache()
        return None

    H_arr = torch.empty(T, device=device)
    Heff_arr = torch.empty(T, device=device)
    sigma_arr = torch.empty(T, device=device)
    centered_chunks = []

    for s in range(0, T, CHUNK):
        e = min(s+CHUNK, T); chunk = logits[s:e]
        lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
        H_arr[s:e] = -(p*lp).sum(-1)
        Heff_arr[s:e] = -(p*p*lp).sum(-1)
        del p, lp
        mu = chunk.mean(-1, keepdim=True); cen = chunk - mu
        sigma_arr[s:e] = cen.pow(2).mean(-1).sqrt().clamp(min=eps)
        centered_chunks.append(cen.cpu())
        del chunk, mu

    H = H_arr.cpu().numpy(); Heff = Heff_arr.cpu().numpy()
    sigma = sigma_arr.cpu().numpy()

    H_bar = H.mean()
    low_H  = H < H_bar
    high_H = H >= H_bar
    sig_gl = sigma.mean()
    sig_lo = sigma[low_H].mean() if low_H.sum() > 0 else sig_gl
    sig_hi = sigma[high_H].mean() if high_H.sum() > 0 else sig_gl
    n_lo = low_H.sum(); n_hi = high_H.sum()

    # ── 计算不同模式的 KL ──
    # 1) group σ (全部 token)
    kl_gp = kl_batch(centered_chunks, np.where(low_H, sig_lo, sig_hi), T, V, device)
    # 2) σ_high (只用于高熵 token)
    kl_sigh = kl_batch(centered_chunks, np.full(T, sig_hi), T, V, device)
    # 3) σ_low (只用于低熵 token)
    kl_siglo = kl_batch(centered_chunks, np.full(T, sig_lo), T, V, device)
    # 4) σ_global
    kl_siggl = kl_batch(centered_chunks, np.full(T, sig_gl), T, V, device)

    del centered_chunks, logits, out; torch.cuda.empty_cache()

    result = {}

    # ── 1/KL_gp (all): 前次最优 ──
    inv_kl_gp_all = 1.0 / (kl_gp + eps)
    result['inv_kl_gp_all'] = float(inv_kl_gp_all.mean())

    # ── 1/KL_hiH (σ_high): 用户提案 ──
    if n_hi > 0:
        inv_kl_hiH_sigh = 1.0 / (kl_sigh[high_H] + eps)
        result['inv_kl_hiH_sigh'] = float(inv_kl_hiH_sigh.mean())
    else:
        result['inv_kl_hiH_sigh'] = 0.0

    # ── 1/KL_loH (σ_low): 对比 ──
    if n_lo > 0:
        inv_kl_loH_siglo = 1.0 / (kl_siglo[low_H] + eps)
        result['inv_kl_loH_siglo'] = float(inv_kl_loH_siglo.mean())
    else:
        result['inv_kl_loH_siglo'] = 0.0

    # ── 1/KL_hiH (σ_global): 隔离 σ 效果 ──
    if n_hi > 0:
        inv_kl_hiH_siggl = 1.0 / (kl_siggl[high_H] + eps)
        result['inv_kl_hiH_siggl'] = float(inv_kl_hiH_siggl.mean())
    else:
        result['inv_kl_hiH_siggl'] = 0.0

    # ── 额外: -KL hiKL (group σ) — 当前实际奖励 ──
    kl_bar = kl_gp.mean()
    hi_kl = kl_gp >= kl_bar
    result['neg_kl_hiKL_gp'] = float(-kl_gp[hi_kl].mean()) if hi_kl.sum() > 0 else 0.0

    # ── 额外: -KL lowH (global σ) — 之前的最优无监督 ──
    if n_lo > 0:
        result['neg_kl_lowH_gl'] = float(-kl_siggl[low_H].mean())
    else:
        result['neg_kl_lowH_gl'] = 0.0

    # ── 额外组合: 加权正确性下的 1/KL_hiH 变体 ──
    # (由 main 函数处理乘以 correctness)

    return result


@torch.no_grad()
def extract_collapse(model, tok, prompt, response, temps=(1.0,0.7,0.5,0.3), device=DEV):
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
        H_arr = torch.empty(T, device=device)
        sigma_arr = torch.empty(T, device=device)
        centered_chunks = []
        for s in range(0, T, CHUNK):
            e = min(s+CHUNK, T); chunk = logits[s:e]
            lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
            H_arr[s:e] = -(p*lp).sum(-1)
            del p, lp
            mu = chunk.mean(-1, keepdim=True); cen = chunk - mu
            sigma_arr[s:e] = cen.pow(2).mean(-1).sqrt().clamp(min=eps)
            centered_chunks.append(cen.cpu())
            del chunk, mu

        H = H_arr.cpu().numpy(); sigma = sigma_arr.cpu().numpy()
        H_bar = H.mean()
        low_H = H < H_bar; high_H = H >= H_bar
        sig_gl = sigma.mean()
        sig_lo = sigma[low_H].mean() if low_H.sum()>0 else sig_gl
        sig_hi = sigma[high_H].mean() if high_H.sum()>0 else sig_gl
        n_lo = low_H.sum(); n_hi = high_H.sum()

        kl_gp = kl_batch(centered_chunks, np.where(low_H, sig_lo, sig_hi), T, V, device)
        kl_sigh = kl_batch(centered_chunks, np.full(T, sig_hi), T, V, device)
        kl_siglo = kl_batch(centered_chunks, np.full(T, sig_lo), T, V, device)
        kl_siggl = kl_batch(centered_chunks, np.full(T, sig_gl), T, V, device)
        del centered_chunks

        row = {'tau': tau}
        row['inv_kl_gp_all']   = float((1.0/(kl_gp+eps)).mean())
        row['inv_kl_hiH_sigh'] = float((1.0/(kl_sigh[high_H]+eps)).mean()) if n_hi>0 else 0.0
        row['inv_kl_loH_siglo']= float((1.0/(kl_siglo[low_H]+eps)).mean()) if n_lo>0 else 0.0
        row['inv_kl_hiH_siggl']= float((1.0/(kl_siggl[high_H]+eps)).mean()) if n_hi>0 else 0.0
        rows.append(row)

    del logits_orig, out; torch.cuda.empty_cache()
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading ...")
    with open(FILE) as f:
        data = json.load(f)
    results = data['results']
    N_Q = len(results)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    KEYS = ['inv_kl_gp_all', 'inv_kl_hiH_sigh', 'inv_kl_loH_siglo',
            'inv_kl_hiH_siggl', 'neg_kl_hiKL_gp', 'neg_kl_lowH_gl']
    LABELS = {
        'inv_kl_gp_all':    '1/KL gp (all)',
        'inv_kl_hiH_sigh':  '1/KL hiH (σ_hi)',   # ← 用户提案
        'inv_kl_loH_siglo': '1/KL loH (σ_lo)',
        'inv_kl_hiH_siggl': '1/KL hiH (σ_gl)',
        'neg_kl_hiKL_gp':   '-KL hiKL (gp)',
        'neg_kl_lowH_gl':   '-KL loH (gl)',
    }

    N_COLLAPSE = min(8, N_Q)
    temps = [1.0, 0.7, 0.5, 0.3]

    # 两种模式
    modes = ['unsup', 'weighted']

    all_c  = {m: {k:[] for k in KEYS} for m in modes}
    all_w  = {m: {k:[] for k in KEYS} for m in modes}
    all_ac = {m: {k:[] for k in KEYS} for m in modes}
    all_aw = {m: {k:[] for k in KEYS} for m in modes}
    collapse_buf = {m: {t: {k:[] for k in KEYS} for t in temps} for m in modes}

    print("\n== 收集 ==")
    for qi, res in enumerate(results):
        prompt = res['prompt']; K = len(res['responses'])
        labels = [1 if r.get('is_correct', False) else 0 for r in res['responses']]

        gv_unsup = {k:[] for k in KEYS}
        gv_weighted = {k:[] for k in KEYS}

        for i, resp in enumerate(res['responses']):
            pt = extract(model, tok, prompt, resp.get('response', ''))
            if pt is None:
                for k in KEYS:
                    gv_unsup[k].append(0.0)
                    gv_weighted[k].append(0.0)
                continue

            ci = float(labels[i])
            for k in KEYS:
                v = pt.get(k, 0.0)
                gv_unsup[k].append(v)
                gv_weighted[k].append(ci * v)

            if qi < N_COLLAPSE:
                crow = extract_collapse(model, tok, prompt, resp.get('response',''), temps=temps)
                if crow:
                    for row in crow:
                        t = row['tau']
                        for k in KEYS:
                            if k in row:
                                collapse_buf['unsup'][t][k].append(row[k])
                                collapse_buf['weighted'][t][k].append(ci * row[k])

        for k in KEYS:
            for mode, gv in [('unsup', gv_unsup), ('weighted', gv_weighted)]:
                adv = grpo_adv(gv[k])
                for i in range(K):
                    dst_v = all_c[mode] if labels[i] else all_w[mode]
                    dst_a = all_ac[mode] if labels[i] else all_aw[mode]
                    dst_v[k].append(gv[k][i])
                    dst_a[k].append(adv[i])

        if (qi+1)%5==0 or qi==N_Q-1:
            print(f"  [{qi+1}/{N_Q}]")

    # ══════ 输出 ══════
    print(f"\n{'='*130}")
    print("  1/KL 高熵变体验证: 纯无监督 + 加权正确性")
    print(f"{'='*130}")

    disc = {}; col = {}
    for mode in modes:
        mode_label = {'unsup': '纯无监督', 'weighted': '加权正确性'}[mode]
        print(f"\n  ── {mode_label} ──")
        print(f"  {'指标':<20s} │ {'d':>8s} │ {'ΔP':>7s} │ {'delta':>8s} │ {'w̄_c':>10s} │ {'w̄_w':>10s} │ {'ratio':>7s} │ 评估")
        print("  " + "─"*95)

        for k in KEYS:
            d = cohens_d(all_c[mode][k], all_w[mode][k])
            pc = np.mean(np.array(all_ac[mode][k]) > 0)
            pw = np.mean(np.array(all_aw[mode][k]) > 0)
            dp = pc - pw
            vs = [np.mean(collapse_buf[mode][t].get(k, [0])) for t in temps]
            delta = (vs[-1]-vs[0])/(abs(vs[0])+1e-12)*100 if abs(vs[0])>1e-12 else 0.0
            mc = np.mean(all_c['unsup'][k]); mw = np.mean(all_w['unsup'][k])
            ratio = mc / (mw + 1e-12)

            disc[(mode,k)] = {'d': d, 'dp': dp}
            col[(mode,k)] = delta

            tags = []
            if d > 0 and delta < 0 and dp > 0: tags.append("★ IDEAL")
            else:
                if d > 0: tags.append("d>0")
                if delta < 0: tags.append("anti-col")
                if dp > 0: tags.append("ΔP>0")

            print(f"  {LABELS[k]:<20s} │ {d:>+8.3f} │ {dp:>+7.3f} │ {delta:>+7.1f}% │"
                  f" {mc:>10.4f} │ {mw:>10.4f} │ {ratio:>7.3f} │ {', '.join(tags)}")

    # ── 汇总对比 ──
    print(f"\n{'='*130}")
    print("  ═══ 核心对比: 1/KL 变体 ═══")
    print(f"{'='*130}")
    focus = ['inv_kl_gp_all', 'inv_kl_hiH_sigh', 'inv_kl_loH_siglo', 'inv_kl_hiH_siggl']
    print(f"  {'指标':<20s} │ {'unsup d':>9s} │ {'unsup ΔP':>9s} │ {'unsup δ':>8s} │"
          f" {'wt d':>9s} │ {'wt ΔP':>9s} │ {'wt δ':>8s} │ {'ratio':>7s}")
    print("  " + "─"*90)
    for k in focus:
        ud, udp = disc[('unsup',k)]['d'], disc[('unsup',k)]['dp']
        udl = col[('unsup',k)]
        wd, wdp = disc[('weighted',k)]['d'], disc[('weighted',k)]['dp']
        wdl = col[('weighted',k)]
        mc = np.mean(all_c['unsup'][k]); mw = np.mean(all_w['unsup'][k])
        ratio = mc/(mw+1e-12)
        print(f"  {LABELS[k]:<20s} │ {ud:>+9.3f} │ {udp:>+9.3f} │ {udl:>+7.1f}% │"
              f" {wd:>+9.3f} │ {wdp:>+9.3f} │ {wdl:>+7.1f}% │ {ratio:>7.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
