#!/usr/bin/env python3
"""
验证 JS(logit_empirical || Gaussian) 作为奖励函数
与 KL(negentropy) 和 R_geo 对比
使用 histogram 方法计算 JS 散度
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
CHUNK = 128
NBINS = 200
BIN_RANGE = (-6.0, 6.0)


def _js_from_hist(z_std_flat):
    """Compute JS(empirical || N(0,1)) from standardized logits using histogram."""
    hist, edges = np.histogram(z_std_flat, bins=NBINS, range=BIN_RANGE, density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total < 1:
        return 0.0
    p = hist / total                            # empirical pmf over bins
    centers = (edges[:-1] + edges[1:]) / 2.0
    bw = edges[1] - edges[0]
    # Gaussian reference (discretized to same bins)
    q = np.exp(-0.5 * centers**2) / np.sqrt(2 * np.pi) * bw
    q = q / q.sum()                             # normalize to pmf
    # JS = H(M) - (H(p) + H(q)) / 2
    m = 0.5 * (p + q)
    # avoid log(0)
    js = 0.0
    for i in range(NBINS):
        if p[i] > 0:
            js += 0.5 * p[i] * math.log(p[i] / (m[i] + 1e-30))
        if q[i] > 0:
            js += 0.5 * q[i] * math.log(q[i] / (m[i] + 1e-30))
    return max(js, 0.0)


@torch.no_grad()
def compute_metrics(model, tok, prompt, response, device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T = logits.size(0)

    KEYS = ['js_mean', 'negent_mean', 'js_Hw', 'negent_Hw', 'R_geo']
    empty = {k: 0.0 for k in KEYS}
    if T < 3:
        return empty

    H        = torch.empty(T, device=device)
    Heff     = torch.empty(T, device=device)
    negent_t = torch.empty(T, device=device)
    js_t     = np.empty(T, dtype=np.float64)

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        chunk = logits[s:e]

        # entropy
        lp = torch.log_softmax(chunk, dim=-1)
        p = lp.exp()
        H[s:e]    = -(p * lp).sum(-1)
        Heff[s:e] = -(p * p * lp).sum(-1)
        del p, lp

        # standardize logits
        mu  = chunk.mean(dim=-1, keepdim=True)
        cen = chunk - mu
        var = cen.pow(2).mean(dim=-1, keepdim=True)
        std = var.sqrt().clamp(min=eps)
        z   = cen / std

        # moment-based negentropy
        m3 = z.pow(3).mean(dim=-1)
        m4 = z.pow(4).mean(dim=-1)
        negent_t[s:e] = m3.pow(2) / 12.0 + (m4 - 3.0).pow(2) / 48.0

        # histogram-based JS (per token, on CPU)
        z_cpu = z.cpu().numpy()
        for i in range(e - s):
            js_t[s + i] = _js_from_hist(z_cpu[i])

        del chunk, cen, z

    sH = float(H.sum().item())
    if sH < eps:
        return empty

    # baselines
    r = Heff / (H + eps)
    log_r = torch.log(r.clamp(min=eps))
    R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    # simple means
    js_mean     = float(np.mean(js_t))
    negent_mean = float(negent_t.mean().item())

    # H-weighted
    H_np = H.cpu().numpy()
    w = H_np / (sH + eps)
    js_Hw     = float(np.sum(w * js_t))
    negent_Hw = float((H / (sH + eps) * negent_t).sum().item())

    del logits, out
    torch.cuda.empty_cache()
    return {
        'js_mean': js_mean, 'negent_mean': negent_mean,
        'js_Hw': js_Hw, 'negent_Hw': negent_Hw,
        'R_geo': R_geo,
    }


@torch.no_grad()
def simulate_sharpening(model, tok, prompt, response,
                        temps=(1.0, 0.7, 0.5, 0.3), device=DEV):
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
        negent_t = torch.empty(T, device=device)
        js_t     = np.empty(T, dtype=np.float64)

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
            negent_t[s:e] = m3.pow(2) / 12.0 + (m4 - 3.0).pow(2) / 48.0

            z_cpu = z.cpu().numpy()
            for i in range(e - s):
                js_t[s + i] = _js_from_hist(z_cpu[i])
            del chunk, cen, z

        sH = float(H.sum().item())
        r = Heff / (H + eps)
        log_r = torch.log(r.clamp(min=eps))
        R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

        H_np = H.cpu().numpy()
        w = H_np / (sH + eps)

        rows.append({
            'tau': tau,
            'js_mean': float(np.mean(js_t)),
            'negent_mean': float(negent_t.mean().item()),
            'js_Hw': float(np.sum(w * js_t)),
            'negent_Hw': float((H / (sH + eps) * negent_t).sum().item()),
            'R_geo': R_geo,
        })

    del logits_orig, out
    torch.cuda.empty_cache()
    return rows


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

    KEYS = ['js_mean', 'negent_mean', 'js_Hw', 'negent_Hw', 'R_geo']
    all_c = {k: [] for k in KEYS}
    all_w = {k: [] for k in KEYS}
    all_ac = {k: [] for k in KEYS}
    all_aw = {k: [] for k in KEYS}

    for qi, res in enumerate(results_data):
        prompt = res['prompt']
        responses = res['responses']
        K = len(responses)
        labels = [1 if r.get('is_correct', False) else 0 for r in responses]
        ml = [compute_metrics(model, tok, prompt, r.get('response', ''))
              for r in responses]
        for mk in KEYS:
            vals = [m[mk] for m in ml]
            adv = grpo_advantage(vals)
            for i in range(K):
                (all_c if labels[i] else all_w)[mk].append(vals[i])
                (all_ac if labels[i] else all_aw)[mk].append(adv[i])
        if (qi + 1) % 5 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}]")

    print("\n" + "="*105)
    print("  JS(logit || Gaussian) vs KL(negentropy) vs R_geo")
    print("="*105)
    hdr = f"  {'指标':<14s} │ {'Cohen d':>9s} │ {'Gap':>9s} │ {'P(+|C)':>7s} │ {'P(+|W)':>7s} │ {'正确均值':>14s} │ {'错误均值':>14s}"
    print(hdr)
    print("  " + "─"*90)

    summary = {}
    for mk in KEYS:
        rc, rw = all_c[mk], all_w[mk]
        ac, aw = all_ac[mk], all_aw[mk]
        d = cohens_d(rc, rw)
        gap = np.mean(ac) - np.mean(aw)
        pc = np.mean(np.array(ac) > 0)
        pw = np.mean(np.array(aw) > 0)
        mc, mw = np.mean(rc), np.mean(rw)
        print(f"  {mk:<14s} │ {d:>+9.4f} │ {gap:>9.4f} │ {pc:>7.3f} │ {pw:>7.3f} │ {mc:>14.6f} │ {mw:>14.6f}")
        summary[mk] = dict(d=d, mc=mc, mw=mw)

    # collapse sim (4 samples for speed since JS is expensive)
    print("\n  防坍缩模拟 (4 样本):")
    temps = [1.0, 0.7, 0.5, 0.3]
    sharp_all = {k: {t: [] for t in temps} for k in KEYS}
    for qi in range(min(4, N_Q)):
        sr = simulate_sharpening(model, tok,
                                  results_data[qi]['prompt'],
                                  results_data[qi]['responses'][0].get('response', ''))
        if sr:
            for entry in sr:
                for k in KEYS:
                    sharp_all[k][entry['tau']].append(entry[k])

    print(f"\n  {'指标':<14s} │ {'tau=1.0':>12s} │ {'tau=0.7':>12s} │ {'tau=0.5':>12s} │ {'tau=0.3':>12s} │ {'Delta':>9s}")
    print("  " + "─"*75)
    for k in KEYS:
        means = [np.mean(sharp_all[k][t]) if sharp_all[k][t] else 0 for t in temps]
        delta = (means[-1] - means[0]) / (abs(means[0]) + eps) * 100
        print(f"  {k:<14s} │ {means[0]:>12.6f} │ {means[1]:>12.6f} │ {means[2]:>12.6f} │ {means[3]:>12.6f} │ {delta:>+8.1f}%")

    print("\n  综合:")
    for k in KEYS:
        d = summary[k]['d']
        means = [np.mean(sharp_all[k][t]) if sharp_all[k][t] else 0 for t in temps]
        delta = (means[-1] - means[0]) / (abs(means[0]) + eps) * 100
        tag = "IDEAL" if d > 0 and abs(delta) < 2 else ""
        print(f"  {k:<14s}: d={d:>+.4f}, delta={delta:>+.1f}%  {tag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
