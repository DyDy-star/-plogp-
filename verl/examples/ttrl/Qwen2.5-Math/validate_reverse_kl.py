#!/usr/bin/env python3
"""
对比三种散度: KL(emp||G), KL(G||emp), JS(emp,G)
emp = standardized logit histogram, G = N(0,1)
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
CHUNK = 128
NBINS = 200
BIN_LO, BIN_HI = -6.0, 6.0

# precompute Gaussian reference pmf (shared across all tokens)
_edges = np.linspace(BIN_LO, BIN_HI, NBINS + 1)
_centers = 0.5 * (_edges[:-1] + _edges[1:])
_bw = _edges[1] - _edges[0]
_gauss = np.exp(-0.5 * _centers**2) / np.sqrt(2 * np.pi) * _bw
_gauss = _gauss / _gauss.sum()          # pmf


def _divergences(z_flat):
    """Return (fwd_kl, rev_kl, js) from standardized logit vector."""
    hist, _ = np.histogram(z_flat, bins=_edges, density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total < 1:
        return 0.0, 0.0, 0.0
    # Laplace smoothing for reverse KL stability
    alpha = 1.0
    p_raw = hist / total                              # empirical (unsmoothed)
    p_sm  = (hist + alpha) / (total + alpha * NBINS)  # smoothed
    q = _gauss

    # forward KL: KL(p || q) = sum p * log(p/q)
    fwd = 0.0
    for i in range(NBINS):
        if p_raw[i] > 0 and q[i] > 0:
            fwd += p_raw[i] * math.log(p_raw[i] / q[i])

    # reverse KL: KL(q || p)  (use smoothed p to avoid inf)
    rev = 0.0
    for i in range(NBINS):
        if q[i] > 0:
            rev += q[i] * math.log(q[i] / p_sm[i])

    # JS
    m = 0.5 * (p_raw + q)
    js = 0.0
    for i in range(NBINS):
        if p_raw[i] > 0:
            js += 0.5 * p_raw[i] * math.log(p_raw[i] / (m[i] + 1e-30))
        if q[i] > 0:
            js += 0.5 * q[i] * math.log(q[i] / (m[i] + 1e-30))

    return max(fwd, 0.0), max(rev, 0.0), max(js, 0.0)


@torch.no_grad()
def compute_metrics(model, tok, prompt, response, device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T = logits.size(0)

    KEYS = ['fwd_kl', 'rev_kl', 'js', 'fwd_Hw', 'rev_Hw', 'js_Hw', 'R_geo']
    empty = {k: 0.0 for k in KEYS}
    if T < 3:
        return empty

    H = torch.empty(T, device=device)
    Heff = torch.empty(T, device=device)
    fwd_arr = np.empty(T)
    rev_arr = np.empty(T)
    js_arr  = np.empty(T)

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
        std = cen.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
        z   = (cen / std).cpu().numpy()

        for i in range(e - s):
            fwd_arr[s+i], rev_arr[s+i], js_arr[s+i] = _divergences(z[i])

        del chunk, cen

    sH = float(H.sum().item())
    if sH < eps:
        return empty

    r = Heff / (H + eps)
    log_r = torch.log(r.clamp(min=eps))
    R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    H_np = H.cpu().numpy()
    w = H_np / (sH + eps)

    del logits, out
    torch.cuda.empty_cache()

    return {
        'fwd_kl': float(np.mean(fwd_arr)),
        'rev_kl': float(np.mean(rev_arr)),
        'js':     float(np.mean(js_arr)),
        'fwd_Hw': float(np.sum(w * fwd_arr)),
        'rev_Hw': float(np.sum(w * rev_arr)),
        'js_Hw':  float(np.sum(w * js_arr)),
        'R_geo':  R_geo,
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
        H = torch.empty(T, device=device)
        Heff = torch.empty(T, device=device)
        fwd_arr = np.empty(T)
        rev_arr = np.empty(T)
        js_arr  = np.empty(T)

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
            std = cen.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
            z   = (cen / std).cpu().numpy()
            for i in range(e - s):
                fwd_arr[s+i], rev_arr[s+i], js_arr[s+i] = _divergences(z[i])
            del chunk, cen

        sH = float(H.sum().item())
        r = Heff / (H + eps)
        log_r = torch.log(r.clamp(min=eps))
        R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())
        H_np = H.cpu().numpy()
        w = H_np / (sH + eps)

        rows.append({
            'tau': tau,
            'fwd_kl': float(np.mean(fwd_arr)),
            'rev_kl': float(np.mean(rev_arr)),
            'js':     float(np.mean(js_arr)),
            'fwd_Hw': float(np.sum(w * fwd_arr)),
            'rev_Hw': float(np.sum(w * rev_arr)),
            'js_Hw':  float(np.sum(w * js_arr)),
            'R_geo':  R_geo,
        })

    del logits_orig, out
    torch.cuda.empty_cache()
    return rows


def grpo_adv(r):
    r = np.array(r, dtype=float)
    s = r.std()
    return (r - r.mean()) / (s + eps) if s > eps else np.zeros_like(r)

def cohens_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


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

    KEYS = ['fwd_kl', 'rev_kl', 'js', 'fwd_Hw', 'rev_Hw', 'js_Hw', 'R_geo']
    all_c  = {k: [] for k in KEYS}
    all_w  = {k: [] for k in KEYS}
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
            adv  = grpo_adv(vals)
            for i in range(K):
                (all_c if labels[i] else all_w)[mk].append(vals[i])
                (all_ac if labels[i] else all_aw)[mk].append(adv[i])
        if (qi+1) % 5 == 0 or qi == N_Q-1:
            print(f"  [{qi+1}/{N_Q}]")

    # ── discrimination table ──
    print("\n" + "="*110)
    print("  Forward KL vs Reverse KL vs JS  (logit || Gaussian)")
    print("="*110)
    print(f"  {'指标':<10s} │ {'Cohen d':>9s} │ {'Gap':>9s} │ {'P(+|C)':>7s} │ {'P(+|W)':>7s} │ {'正确均值':>14s} │ {'错误均值':>14s}")
    print("  " + "─"*82)

    summary = {}
    for mk in KEYS:
        d   = cohens_d(all_c[mk], all_w[mk])
        gap = np.mean(all_ac[mk]) - np.mean(all_aw[mk])
        pc  = np.mean(np.array(all_ac[mk]) > 0)
        pw  = np.mean(np.array(all_aw[mk]) > 0)
        mc  = np.mean(all_c[mk])
        mw  = np.mean(all_w[mk])
        print(f"  {mk:<10s} │ {d:>+9.4f} │ {gap:>+9.4f} │ {pc:>7.3f} │ {pw:>7.3f} │ {mc:>14.6f} │ {mw:>14.6f}")
        summary[mk] = d

    # ── collapse sim (4 samples) ──
    print("\n  防坍缩模拟 (4 样本):")
    temps = [1.0, 0.7, 0.5, 0.3]
    sharp = {k: {t: [] for t in temps} for k in KEYS}
    for qi in range(min(4, N_Q)):
        sr = simulate_sharpening(model, tok,
                                  results_data[qi]['prompt'],
                                  results_data[qi]['responses'][0].get('response', ''))
        if sr:
            for e in sr:
                for k in KEYS:
                    sharp[k][e['tau']].append(e[k])

    print(f"\n  {'指标':<10s} │ {'tau=1.0':>12s} │ {'tau=0.7':>12s} │ {'tau=0.5':>12s} │ {'tau=0.3':>12s} │ {'Delta':>8s}")
    print("  " + "─"*68)
    collapse = {}
    for k in KEYS:
        ms = [np.mean(sharp[k][t]) if sharp[k][t] else 0 for t in temps]
        delta = (ms[-1] - ms[0]) / (abs(ms[0]) + eps) * 100
        collapse[k] = delta
        print(f"  {k:<10s} │ {ms[0]:>12.6f} │ {ms[1]:>12.6f} │ {ms[2]:>12.6f} │ {ms[3]:>12.6f} │ {delta:>+7.1f}%")

    # ── summary ──
    print("\n  综合 (d>0 且 |delta|<2% = IDEAL):")
    for k in KEYS:
        d = summary[k]
        delta = collapse[k]
        tag = "*** IDEAL ***" if d > 0 and abs(delta) < 2 else \
              "d>0" if d > 0 else ""
        print(f"  {k:<10s}: d={d:>+.4f}, delta={delta:>+.1f}%  {tag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
