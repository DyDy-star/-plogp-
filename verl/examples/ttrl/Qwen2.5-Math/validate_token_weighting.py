#!/usr/bin/env python3
"""
验证: 使用 per-token 指标作为优势 A 的权重 (token-level credit assignment)

核心对比:
  1. 纯无监督: reward = mean(w_t)            → 之前的做法
  2. 加权正确性: reward = correctness × mean(w_t) → 用户新提案
  3. 纯正确性: reward = correctness (1/0)      → 基线

per-token 权重候选:
  - uniform:  w_t = 1 (基线)
  - H_t:      熵 (不确定位置权重大)
  - inv_H:    1/(H+ε) (确定位置权重大)
  - KL_pt:    KL(G||emp) per-token σ (非高斯位置权重大)
  - KL_gp:    KL(G||emp) group σ
  - inv_KL:   1/(KL+ε) (高斯位置权重大)
  - sigma:    σ_t logit std (高置信位置权重大)
  - r_t:      H_eff/H (熵效率低的位置权重大)

对每种权重, 测量:
  d (区分度), ΔP (GRPO信号), delta (防坍缩)
  - 纯无监督 (仅 w̄ 作为 reward)
  - 加权正确性 (correctness × w̄ 作为 reward)
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


@torch.no_grad()
def extract_per_token(model, tok, prompt, response, device=DEV):
    """单次 forward: 提取多种 per-token 指标."""
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
    gauss = torch.exp(-0.5*centers**2)/math.sqrt(2*math.pi)*bw
    gauss = gauss/gauss.sum(); log_gauss = torch.log(gauss)

    H_arr    = torch.empty(T, device=device)
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

    H_np = H_arr.cpu().numpy(); Heff_np = Heff_arr.cpu().numpy()
    sigma_np = sigma_arr.cpu().numpy()

    # group σ
    H_bar = H_np.mean()
    low_H = H_np < H_bar; high_H = ~low_H
    sig_gl = sigma_np.mean()
    sig_lo = sigma_np[low_H].mean() if low_H.sum() > 0 else sig_gl
    sig_hi = sigma_np[high_H].mean() if high_H.sum() > 0 else sig_gl

    def kl_with_sigma(sig_vec):
        sig_t = torch.tensor(sig_vec, device=device, dtype=torch.float32)
        kl = torch.empty(T, device=device); idx = 0
        for cen_cpu in centered_chunks:
            cen = cen_cpu.to(device); c = cen.size(0)
            z = cen / sig_t[idx:idx+c].unsqueeze(-1); del cen
            bi = ((z-BIN_LO)/bw).long().clamp(0,NBINS-1)
            hist = torch.zeros(c, NBINS, device=device)
            hist.scatter_add_(1, bi, torch.ones_like(z)); del bi, z
            p_sm = (hist+1.0)/(V+NBINS)
            kl[idx:idx+c] = (gauss.unsqueeze(0)*(log_gauss.unsqueeze(0)-torch.log(p_sm))).sum(-1)
            del hist, p_sm; idx += c
        return kl.cpu().numpy()

    kl_pt = kl_with_sigma(sigma_np)                            # per-token σ
    kl_gp = kl_with_sigma(np.where(low_H, sig_lo, sig_hi))    # group σ
    kl_gl = kl_with_sigma(np.full(T, sig_gl))                 # global σ

    del centered_chunks, logits, out; torch.cuda.empty_cache()

    # 计算各种 per-token 权重的 response 级均值
    r_t = Heff_np / (H_np + eps)
    inv_H = 1.0 / (H_np + eps)
    inv_KL = 1.0 / (kl_gp + eps)

    return {
        'T': T,
        # per-token 权重均值 (response 级指标)
        'w_uniform': 1.0,
        'w_H':       float(H_np.mean()),
        'w_inv_H':   float(inv_H.mean()),
        'w_KL_pt':   float(kl_pt.mean()),
        'w_KL_gp':   float(kl_gp.mean()),
        'w_inv_KL':  float(inv_KL.mean()),
        'w_sigma':   float(sigma_np.mean()),
        'w_r_t':     float(r_t.mean()),
        # 已有的 response 级指标 (对比)
        'neg_kl_hiKL_gp': float(-kl_gp[kl_gp >= kl_gp.mean()].mean()) if (kl_gp >= kl_gp.mean()).sum() > 0 else 0.0,
        'neg_kl_lowH_gl': float(-kl_gl[low_H].mean()) if low_H.sum() > 0 else 0.0,
        'R_geo': float(np.exp((H_np * np.log(np.clip(r_t, eps, None))).sum() / (H_np.sum() + eps))),
    }


@torch.no_grad()
def extract_collapse(model, tok, prompt, response, temps=(1.0,0.7,0.5,0.3), device=DEV):
    """多温度模拟 — 对每种权重计算均值."""
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
    gauss = torch.exp(-0.5*centers**2)/math.sqrt(2*math.pi)*bw
    gauss = gauss/gauss.sum(); log_gauss = torch.log(gauss)

    rows = []
    for tau in temps:
        logits = logits_orig / tau
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

        H_np = H_arr.cpu().numpy(); Heff_np = Heff_arr.cpu().numpy()
        sigma_np = sigma_arr.cpu().numpy()
        H_bar = H_np.mean()
        low_H = H_np < H_bar; high_H = ~low_H
        sig_gl = sigma_np.mean()
        sig_lo = sigma_np[low_H].mean() if low_H.sum()>0 else sig_gl
        sig_hi = sigma_np[high_H].mean() if high_H.sum()>0 else sig_gl

        def kl_w(sv):
            st = torch.tensor(sv, device=device, dtype=torch.float32)
            kl = torch.empty(T, device=device); idx = 0
            for cc in centered_chunks:
                c2 = cc.to(device); c = c2.size(0)
                z2 = c2/st[idx:idx+c].unsqueeze(-1); del c2
                bi = ((z2-BIN_LO)/bw).long().clamp(0,NBINS-1)
                h2 = torch.zeros(c,NBINS,device=device)
                h2.scatter_add_(1,bi,torch.ones_like(z2)); del bi,z2
                ps = (h2+1.0)/(V+NBINS)
                kl[idx:idx+c] = (gauss.unsqueeze(0)*(log_gauss.unsqueeze(0)-torch.log(ps))).sum(-1)
                del h2,ps; idx+=c
            return kl.cpu().numpy()

        kl_gp = kl_w(np.where(low_H, sig_lo, sig_hi))
        r_t = Heff_np/(H_np+eps)

        row = {
            'tau': tau,
            'w_uniform': 1.0,
            'w_H': float(H_np.mean()),
            'w_inv_H': float((1.0/(H_np+eps)).mean()),
            'w_KL_gp': float(kl_gp.mean()),
            'w_inv_KL': float((1.0/(kl_gp+eps)).mean()),
            'w_sigma': float(sigma_np.mean()),
            'w_r_t': float(r_t.mean()),
        }
        rows.append(row)
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

    WEIGHTS = ['w_uniform', 'w_H', 'w_inv_H', 'w_KL_gp', 'w_inv_KL', 'w_sigma', 'w_r_t']
    W_LABELS = {
        'w_uniform': 'uniform',
        'w_H':       'H_t (熵)',
        'w_inv_H':   '1/H_t',
        'w_KL_gp':   'KL_gp',
        'w_inv_KL':  '1/KL_gp',
        'w_sigma':   'σ_t',
        'w_r_t':     'r_t=Heff/H',
    }

    # 对比指标 (之前的最优)
    BASELINES = ['neg_kl_hiKL_gp', 'neg_kl_lowH_gl', 'R_geo']
    BL_LABELS = {
        'neg_kl_hiKL_gp': '-KL hiKL(gp)',
        'neg_kl_lowH_gl': '-KL loH(gl)',
        'R_geo':          'R_geo',
    }

    N_COLLAPSE = min(8, N_Q)
    temps = [1.0, 0.7, 0.5, 0.3]

    # 收集
    # For "weighted correctness": reward = correct × w̄
    # For "unsupervised": reward = w̄
    # For "correctness only": reward = correct
    print("\n== 收集 ==")

    # 三种模式的 correct/wrong 值和 GRPO 优势
    modes = ['unsup', 'weighted', 'correctness']
    all_c  = {m: {k: [] for k in WEIGHTS + BASELINES} for m in modes}
    all_w  = {m: {k: [] for k in WEIGHTS + BASELINES} for m in modes}
    all_ac = {m: {k: [] for k in WEIGHTS + BASELINES} for m in modes}
    all_aw = {m: {k: [] for k in WEIGHTS + BASELINES} for m in modes}

    # 坍缩缓冲 (仅 unsup 和 weighted)
    collapse_buf = {
        m: {t: {k: [] for k in WEIGHTS} for t in temps}
        for m in ['unsup', 'weighted']
    }

    for qi, res in enumerate(results):
        prompt = res['prompt']
        K = len(res['responses'])
        labels = [1 if r.get('is_correct', False) else 0 for r in res['responses']]

        group_unsup     = {k: [] for k in WEIGHTS + BASELINES}
        group_weighted  = {k: [] for k in WEIGHTS + BASELINES}
        group_correct   = {k: [] for k in WEIGHTS + BASELINES}

        for i, resp in enumerate(res['responses']):
            pt = extract_per_token(model, tok, prompt, resp.get('response', ''))
            if pt is None:
                for k in WEIGHTS + BASELINES:
                    group_unsup[k].append(0.0)
                    group_weighted[k].append(0.0)
                    group_correct[k].append(float(labels[i]))
                continue

            correct_i = float(labels[i])
            for k in WEIGHTS:
                w_val = pt[k]
                group_unsup[k].append(w_val)                    # 纯无监督
                group_weighted[k].append(correct_i * w_val)     # 加权正确性
                group_correct[k].append(correct_i)              # 纯正确性

            for k in BASELINES:
                bl_val = pt.get(k, 0.0)
                group_unsup[k].append(bl_val)
                group_weighted[k].append(correct_i * bl_val)
                group_correct[k].append(correct_i)

            # 坍缩模拟
            if qi < N_COLLAPSE:
                crow = extract_collapse(model, tok, prompt, resp.get('response', ''), temps=temps)
                if crow:
                    for row in crow:
                        t = row['tau']
                        for k in WEIGHTS:
                            if k in row:
                                collapse_buf['unsup'][t][k].append(row[k])
                                collapse_buf['weighted'][t][k].append(correct_i * row[k])

        # GRPO advantage
        groups = {'unsup': group_unsup, 'weighted': group_weighted, 'correctness': group_correct}
        for mode, gv in groups.items():
            for k in WEIGHTS + BASELINES:
                adv = grpo_adv(gv[k])
                for i in range(K):
                    dst_v = all_c[mode] if labels[i] else all_w[mode]
                    dst_a = all_ac[mode] if labels[i] else all_aw[mode]
                    dst_v[k].append(gv[k][i])
                    dst_a[k].append(adv[i])

        if (qi+1) % 5 == 0 or qi == N_Q-1:
            print(f"  [{qi+1}/{N_Q}]")

    # ══════ 结果 ══════

    # ── Part 1: 纯无监督 vs 加权正确性 vs 纯正确性 ──
    print("\n" + "="*140)
    print("  Part 1: 三种模式对比 — 纯无监督 / 加权正确性(correctness×w̄) / 纯正确性")
    print("="*140)
    print(f"  {'权重':<14s} │ {'--- 纯无监督 ---':^26s} │ {'--- 加权正确性 ---':^26s} │ {'--- 纯正确性 ---':^26s}")
    print(f"  {'':14s} │ {'d':>8s} {'ΔP':>8s} │ {'d':>8s} {'ΔP':>8s} │ {'d':>8s} {'ΔP':>8s}")
    print("  " + "─"*85)

    results_table = {}
    for k in WEIGHTS:
        row = {}
        for mode in modes:
            d = cohens_d(all_c[mode][k], all_w[mode][k])
            pc = np.mean(np.array(all_ac[mode][k]) > 0)
            pw = np.mean(np.array(all_aw[mode][k]) > 0)
            dp = pc - pw
            row[mode] = {'d': d, 'dp': dp}
        results_table[k] = row
        print(f"  {W_LABELS[k]:<14s} │ {row['unsup']['d']:>+8.3f} {row['unsup']['dp']:>+8.3f} │"
              f" {row['weighted']['d']:>+8.3f} {row['weighted']['dp']:>+8.3f} │"
              f" {row['correctness']['d']:>+8.3f} {row['correctness']['dp']:>+8.3f}")

    # 基线对比
    print("  " + "─"*85)
    for k in BASELINES:
        row = {}
        for mode in modes:
            d = cohens_d(all_c[mode][k], all_w[mode][k])
            pc = np.mean(np.array(all_ac[mode][k]) > 0)
            pw = np.mean(np.array(all_aw[mode][k]) > 0)
            row[mode] = {'d': d, 'dp': pc - pw}
        results_table[k] = row
        print(f"  {BL_LABELS[k]:<14s} │ {row['unsup']['d']:>+8.3f} {row['unsup']['dp']:>+8.3f} │"
              f" {row['weighted']['d']:>+8.3f} {row['weighted']['dp']:>+8.3f} │"
              f" {row['correctness']['d']:>+8.3f} {row['correctness']['dp']:>+8.3f}")

    # ── Part 2: 加权正确性的坍缩分析 ──
    print(f"\n{'='*140}")
    print("  Part 2: 坍缩分析 — 加权正确性 (correctness × w̄)")
    print("="*140)
    print(f"  {'权重':<14s} │ {'unsup delta':>12s} │ {'weighted delta':>14s} │ 评估")
    print("  " + "─"*60)

    for k in WEIGHTS:
        deltas = {}
        for mode in ['unsup', 'weighted']:
            vs = [np.mean(collapse_buf[mode][t].get(k, [0])) for t in temps]
            deltas[mode] = (vs[-1]-vs[0])/(abs(vs[0])+1e-12)*100 if abs(vs[0]) > 1e-12 else 0.0
        d_w = results_table[k]['weighted']['d']
        dp_w = results_table[k]['weighted']['dp']
        tags = []
        if d_w > 0 and deltas['weighted'] < 0 and dp_w > 0:
            tags.append("★ IDEAL")
        else:
            if d_w > 0: tags.append("d>0")
            if deltas['weighted'] < 0: tags.append("anti-col")
            if dp_w > 0: tags.append("ΔP>0")
        print(f"  {W_LABELS[k]:<14s} │ {deltas['unsup']:>+11.1f}% │ {deltas['weighted']:>+13.1f}% │ {', '.join(tags)}")

    # ── Part 3: 信号放大分析 ──
    print(f"\n{'='*140}")
    print("  Part 3: 信号放大 — mean(w̄|correct) vs mean(w̄|incorrect)")
    print("="*140)
    print(f"  {'权重':<14s} │ {'w̄_correct':>12s} │ {'w̄_incorrect':>12s} │ {'ratio':>8s} │ 含义")
    print("  " + "─"*70)

    for k in WEIGHTS:
        mc = np.mean(all_c['unsup'][k])
        mw = np.mean(all_w['unsup'][k])
        ratio = mc / (mw + 1e-12)
        if ratio > 1.05:
            meaning = "放大正确 (↑ 学正确)"
        elif ratio < 0.95:
            meaning = "放大错误 (↑ 去错误)"
        else:
            meaning = "中性 (≈ 均匀)"
        print(f"  {W_LABELS[k]:<14s} │ {mc:>12.6f} │ {mw:>12.6f} │ {ratio:>8.3f} │ {meaning}")

    # ── Part 4: 综合排名 ──
    print(f"\n{'='*140}")
    print("  ═══ 综合排名: 加权正确性模式 (correctness × w̄) ═══")
    print("="*140)
    ranked = sorted(WEIGHTS, key=lambda k: results_table[k]['weighted']['dp'], reverse=True)
    for rank, k in enumerate(ranked, 1):
        d = results_table[k]['weighted']['d']
        dp = results_table[k]['weighted']['dp']
        print(f"  #{rank} {W_LABELS[k]:<14s}: d={d:>+.3f}, ΔP={dp:>+.3f}")

    # 对比纯正确性
    d_corr = results_table[WEIGHTS[0]]['correctness']['d']
    dp_corr = results_table[WEIGHTS[0]]['correctness']['dp']
    print(f"\n  基线 (纯正确性): d={d_corr:>+.3f}, ΔP={dp_corr:>+.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
