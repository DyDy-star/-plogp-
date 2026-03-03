#!/usr/bin/env python3
"""
验证 KL_low - KL_high (按 KL 阈值分割的差值奖励):

  R = mean(KL | KL<KL̄) - mean(KL | KL≥KL̄)   (始终 ≤ 0)
  -R = mean(KL | KL≥KL̄) - mean(KL | KL<KL̄)  (取负, ≥ 0)

  对三种标准化模式各自测试, 并与已有最优候选对比.
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
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/(sp+1e-12)


def grpo_adv(vals):
    r = np.asarray(vals, float); s = r.std()
    return (r-r.mean())/(s+eps) if s > eps else np.zeros_like(r)


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
        e = min(s+CHUNK, T); chunk = logits[s:e]; c = e-s
        lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
        H_arr[s:e] = -(p*lp).sum(-1)
        Heff_arr[s:e] = -(p*p*lp).sum(-1)
        del p, lp
        mu = chunk.mean(-1, keepdim=True)
        cen = chunk - mu
        std_t = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
        sigma_arr[s:e] = std_t.squeeze(-1)
        centered_chunks.append(cen.cpu())
        del chunk, mu

    del logits, out; torch.cuda.empty_cache()
    return {'H': H_arr.cpu().numpy(), 'Heff': Heff_arr.cpu().numpy(),
            'sigma': sigma_arr.cpu().numpy(), 'centered': centered_chunks,
            'T': T, 'V': V}


def compute_kl(centered_chunks, sigma_vec, T, V, device):
    bw = (BIN_HI-BIN_LO)/NBINS
    centers = torch.linspace(BIN_LO+bw/2, BIN_HI-bw/2, NBINS, device=device)
    gauss = torch.exp(-0.5*centers**2)/math.sqrt(2*math.pi)*bw
    gauss = gauss/gauss.sum(); log_gauss = torch.log(gauss)
    sig_t = torch.tensor(sigma_vec, device=device, dtype=torch.float32)
    kl = torch.empty(T, device=device); idx = 0
    for cen_cpu in centered_chunks:
        cen = cen_cpu.to(device); c = cen.size(0)
        z = cen/sig_t[idx:idx+c].unsqueeze(-1); del cen
        bi = ((z-BIN_LO)/bw).long().clamp(0, NBINS-1)
        hist = torch.zeros(c, NBINS, device=device)
        hist.scatter_add_(1, bi, torch.ones_like(z)); del bi, z
        p_sm = (hist+1.0)/(V+NBINS)
        kl[idx:idx+c] = (gauss.unsqueeze(0)*(log_gauss.unsqueeze(0)-torch.log(p_sm))).sum(-1)
        del hist, p_sm; idx += c
    return kl.cpu().numpy()


def aggregate(pt):
    H = pt['H']; Heff = pt['Heff']; sigma = pt['sigma']
    T = pt['T']; V = pt['V']; cen = pt['centered']
    if T < 3 or H.sum() < eps:
        return None

    H_bar = H.mean()
    low_H = H < H_bar; high_H = H >= H_bar
    sig_gl = sigma.mean()
    sig_lo_H = sigma[low_H].mean() if low_H.sum()>0 else sig_gl
    sig_hi_H = sigma[high_H].mean() if high_H.sum()>0 else sig_gl

    sH = H.sum()
    r = Heff/(H+eps); log_r = np.log(np.clip(r, eps, None))
    R_geo = float(np.exp((H*log_r).sum()/(sH+eps)))

    modes = {
        'pt': sigma,
        'gl': np.full(T, sig_gl),
        'gp': np.where(low_H, sig_lo_H, sig_hi_H),
    }

    result = {'R_geo': R_geo, 'mean_H': float(H_bar)}

    for mn, sig_vec in modes.items():
        kl = compute_kl(cen, sig_vec, T, V, DEV)
        kl_bar = kl.mean()
        low_KL = kl < kl_bar; high_KL = kl >= kl_bar
        n_lo_K = low_KL.sum(); n_hi_K = high_KL.sum()
        n_lo_H = low_H.sum(); n_hi_H = high_H.sum()

        kl_lo_K = kl[low_KL].mean() if n_lo_K>0 else 0.0
        kl_hi_K = kl[high_KL].mean() if n_hi_K>0 else 0.0
        kl_lo_H = kl[low_H].mean() if n_lo_H>0 else 0.0
        kl_hi_H = kl[high_H].mean() if n_hi_H>0 else 0.0

        # KL_low - KL_high (按 KL 分割)
        result[f'kl_diff_KL_{mn}']     = float(kl_lo_K - kl_hi_K)   # ≤ 0
        result[f'neg_kl_diff_KL_{mn}'] = float(kl_hi_K - kl_lo_K)   # ≥ 0

        # KL_low - KL_high (按 H 分割) — 对比
        result[f'kl_diff_H_{mn}']     = float(kl_lo_H - kl_hi_H)
        result[f'neg_kl_diff_H_{mn}'] = float(kl_hi_H - kl_lo_H)

        # 基线对比
        result[f'neg_kl_mean_{mn}']  = float(-kl.mean())
        result[f'neg_kl_lowH_{mn}']  = float(-kl_lo_H)
        result[f'neg_kl_hiKL_{mn}']  = float(-kl_hi_K)

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
        Heff_arr = torch.empty(T, device=device)
        sigma_arr = torch.empty(T, device=device)
        centered_chunks = []
        for s in range(0, T, CHUNK):
            e = min(s+CHUNK, T); chunk = logits[s:e]; c = e-s
            lp = torch.log_softmax(chunk, dim=-1); p = lp.exp()
            H_arr[s:e] = -(p*lp).sum(-1)
            Heff_arr[s:e] = -(p*p*lp).sum(-1)
            del p, lp
            mu = chunk.mean(-1, keepdim=True); cen = chunk-mu
            std_t = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
            sigma_arr[s:e] = std_t.squeeze(-1)
            centered_chunks.append(cen.cpu())
            del chunk, mu

        pt_data = {'H': H_arr.cpu().numpy(), 'Heff': Heff_arr.cpu().numpy(),
                    'sigma': sigma_arr.cpu().numpy(), 'centered': centered_chunks,
                    'T': T, 'V': V}
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

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    MODES = ['pt', 'gl', 'gp']
    # 核心对比指标
    KEYS = []
    for m in MODES:
        KEYS += [
            f'kl_diff_KL_{m}',      # KL_low-KL_high (按KL分)
            f'neg_kl_diff_KL_{m}',   # -(KL_low-KL_high) (按KL分)
            f'kl_diff_H_{m}',        # KL_low-KL_high (按H分)
            f'neg_kl_diff_H_{m}',    # -(KL_low-KL_high) (按H分)
            f'neg_kl_mean_{m}',      # -KL mean (基线)
            f'neg_kl_lowH_{m}',      # -KL low-H (已知最优)
            f'neg_kl_hiKL_{m}',      # -KL high-KL (已知最优KL分割)
        ]
    KEYS.append('R_geo')

    MODE_NAMES = {'pt': 'per-tok', 'gl': 'global', 'gp': 'group'}
    LABELS = {}
    for m in MODES:
        mn = MODE_NAMES[m]
        LABELS[f'kl_diff_KL_{m}']     = f'KLlo-KLhi [KL] ({mn})'
        LABELS[f'neg_kl_diff_KL_{m}'] = f'-(KLlo-KLhi) [KL] ({mn})'
        LABELS[f'kl_diff_H_{m}']      = f'KLlo-KLhi [H] ({mn})'
        LABELS[f'neg_kl_diff_H_{m}']  = f'-(KLlo-KLhi) [H] ({mn})'
        LABELS[f'neg_kl_mean_{m}']    = f'-KL mean ({mn})'
        LABELS[f'neg_kl_lowH_{m}']    = f'-KL low-H ({mn})'
        LABELS[f'neg_kl_hiKL_{m}']    = f'-KL hi-KL ({mn})'
    LABELS['R_geo'] = 'R_geo'

    N_COLLAPSE = min(8, N_Q)
    temps = [1.0, 0.7, 0.5, 0.3]

    print("\n== 收集 ==")
    all_c  = {k:[] for k in KEYS}
    all_w  = {k:[] for k in KEYS}
    all_ac = {k:[] for k in KEYS}
    all_aw = {k:[] for k in KEYS}
    collapse_buf = {t:{k:[] for k in KEYS} for t in temps}

    for qi, res in enumerate(results):
        prompt = res['prompt']; K = len(res['responses'])
        labels = [1 if r.get('is_correct',False) else 0 for r in res['responses']]
        group_vals = {k:[] for k in KEYS}

        for i, resp in enumerate(res['responses']):
            pt = extract(model, tok, prompt, resp.get('response',''))
            if pt is None:
                for k in KEYS: group_vals[k].append(0.0)
                continue
            m = aggregate(pt)
            if m is None:
                for k in KEYS: group_vals[k].append(0.0)
                continue
            for k in KEYS:
                group_vals[k].append(m.get(k, 0.0))

            if qi < N_COLLAPSE:
                crow = extract_collapse(model, tok, prompt, resp.get('response',''), temps=temps)
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

        if (qi+1)%5==0 or qi==N_Q-1:
            print(f"  [{qi+1}/{N_Q}]")

    # ── Results ──
    disc = {}; col_delta = {}
    for k in KEYS:
        d = cohens_d(all_c[k], all_w[k])
        pc = np.mean(np.array(all_ac[k])>0)
        pw = np.mean(np.array(all_aw[k])>0)
        dp = pc-pw
        disc[k] = {'d':d, 'dp':dp}
        vs = [np.mean(collapse_buf[t].get(k,[0])) for t in temps]
        delta = (vs[-1]-vs[0])/(abs(vs[0])+1e-12)*100
        col_delta[k] = delta

    # ── 核心输出: 按模式分组 ──
    for m in MODES:
        mn = MODE_NAMES[m]
        print(f"\n{'='*90}")
        print(f"  标准化: {mn} σ")
        print(f"{'='*90}")
        print(f"  {'指标':<28s} │ {'d':>8s} │ {'delta':>8s} │ {'ΔP':>7s} │ 评估")
        print("  "+"─"*70)

        focus = [
            f'kl_diff_KL_{m}',
            f'neg_kl_diff_KL_{m}',
            f'kl_diff_H_{m}',
            f'neg_kl_diff_H_{m}',
            f'neg_kl_mean_{m}',
            f'neg_kl_lowH_{m}',
            f'neg_kl_hiKL_{m}',
        ]
        for k in focus:
            d, delta, dp = disc[k]['d'], col_delta[k], disc[k]['dp']
            tags = []
            if d>0 and delta<0 and dp>0: tags.append("★ IDEAL")
            else:
                if d>0: tags.append("d>0")
                if delta<0: tags.append("anti-col")
                if dp>0: tags.append("ΔP>0")
            print(f"  {LABELS[k]:<28s} │ {d:>+8.3f} │ {delta:>+7.1f}% │ {dp:>+7.3f} │ {', '.join(tags)}")

    # R_geo
    k = 'R_geo'
    d, delta, dp = disc[k]['d'], col_delta[k], disc[k]['dp']
    print(f"\n  {'R_geo':<28s} │ {d:>+8.3f} │ {delta:>+7.1f}% │ {dp:>+7.3f}")

    # ── 最终汇总: KL_lo-KL_hi 核心对比 ──
    print(f"\n{'='*90}")
    print("  ═══ 核心对比: KL_low-KL_high ═══")
    print(f"{'='*90}")
    print(f"  {'标准化':<10s} │ {'分割':<6s} │ {'KLlo-KLhi d':>12s} │ {'delta':>8s} │ {'ΔP':>7s} │ {'-(KLlo-KLhi) d':>15s} │ {'delta':>8s} │ {'ΔP':>7s}")
    print("  "+"─"*95)
    for m in MODES:
        mn = MODE_NAMES[m]
        # KL split
        k1 = f'kl_diff_KL_{m}'; k2 = f'neg_kl_diff_KL_{m}'
        d1,dl1,dp1 = disc[k1]['d'],col_delta[k1],disc[k1]['dp']
        d2,dl2,dp2 = disc[k2]['d'],col_delta[k2],disc[k2]['dp']
        print(f"  {mn:<10s} │ {'KL':<6s} │ {d1:>+12.3f} │ {dl1:>+7.1f}% │ {dp1:>+7.3f} │ {d2:>+15.3f} │ {dl2:>+7.1f}% │ {dp2:>+7.3f}")
        # H split
        k1 = f'kl_diff_H_{m}'; k2 = f'neg_kl_diff_H_{m}'
        d1,dl1,dp1 = disc[k1]['d'],col_delta[k1],disc[k1]['dp']
        d2,dl2,dp2 = disc[k2]['d'],col_delta[k2],disc[k2]['dp']
        print(f"  {mn:<10s} │ {'H':<6s} │ {d1:>+12.3f} │ {dl1:>+7.1f}% │ {dp1:>+7.3f} │ {d2:>+15.3f} │ {dl2:>+7.1f}% │ {dp2:>+7.3f}")
        print("  "+"─"*95)

    print("\nDone.")


if __name__ == "__main__":
    main()
