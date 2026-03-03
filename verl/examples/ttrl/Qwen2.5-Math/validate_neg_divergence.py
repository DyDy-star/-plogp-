#!/usr/bin/env python3
"""
对比 -KL(emp||G), -KL(G||emp), -JS(emp,G) 作为奖励函数
取负 → 奖励更高斯的 logit 分布 → 正向区分 + 温度不变
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
BIN_LO, BIN_HI = -6.0, 6.0

_edges = np.linspace(BIN_LO, BIN_HI, NBINS + 1)
_centers = 0.5 * (_edges[:-1] + _edges[1:])
_bw = _edges[1] - _edges[0]
_gauss = np.exp(-0.5 * _centers**2) / np.sqrt(2 * np.pi) * _bw
_gauss /= _gauss.sum()


def _divergences(z_flat):
    hist, _ = np.histogram(z_flat, bins=_edges, density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total < 1:
        return 0.0, 0.0, 0.0
    p = hist / total
    p_sm = (hist + 1.0) / (total + NBINS)   # Laplace smooth for rev KL
    q = _gauss

    fwd = sum(p[i] * math.log(p[i] / q[i]) for i in range(NBINS) if p[i] > 0 and q[i] > 0)
    rev = sum(q[i] * math.log(q[i] / p_sm[i]) for i in range(NBINS) if q[i] > 0)
    m = 0.5 * (p + q)
    js = sum(0.5*p[i]*math.log(p[i]/(m[i]+1e-30)) for i in range(NBINS) if p[i]>0) + \
         sum(0.5*q[i]*math.log(q[i]/(m[i]+1e-30)) for i in range(NBINS) if q[i]>0)
    return max(fwd, 0.0), max(rev, 0.0), max(js, 0.0)


@torch.no_grad()
def compute_metrics(model, tok, prompt, response, device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T = logits.size(0)

    KEYS = ['neg_fwd', 'neg_rev', 'neg_js', 'R_geo', 'R_alpha042']
    empty = {k: 0.0 for k in KEYS}
    if T < 3:
        return empty

    H    = torch.empty(T, device=device)
    Heff = torch.empty(T, device=device)
    fwd  = np.empty(T)
    rev  = np.empty(T)
    js   = np.empty(T)

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        chunk = logits[s:e]

        lp = torch.log_softmax(chunk, dim=-1)
        p = lp.exp()
        H[s:e]    = -(p * lp).sum(-1)
        Heff[s:e] = -(p * p * lp).sum(-1)
        del p, lp

        mu  = chunk.mean(-1, keepdim=True)
        cen = chunk - mu
        std = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
        z_np = (cen / std).cpu().numpy()
        for i in range(e - s):
            fwd[s+i], rev[s+i], js[s+i] = _divergences(z_np[i])
        del chunk, cen

    sH = H.sum()
    if sH < eps:
        return empty

    # baselines
    r = Heff / (H + eps)
    log_r = torch.log(r.clamp(min=eps))
    R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    log_Heff = torch.log(Heff.clamp(min=eps))
    log_H = torch.log(H.clamp(min=eps))
    R_alpha042 = float(torch.exp((H * (log_Heff - 0.42*log_H)).sum() / (sH + eps)).item())

    del logits, out
    torch.cuda.empty_cache()

    return {
        'neg_fwd':     -float(np.mean(fwd)),
        'neg_rev':     -float(np.mean(rev)),
        'neg_js':      -float(np.mean(js)),
        'R_geo':       R_geo,
        'R_alpha042':  R_alpha042,
    }


@torch.no_grad()
def simulate_sharpening(model, tok, prompt, response,
                        temps=(1.0, 0.7, 0.5, 0.3), device=DEV):
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T_len = logits_orig.size(0)
    if T_len < 3:
        return None

    rows = []
    for tau in temps:
        logits = logits_orig / tau
        H    = torch.empty(T_len, device=device)
        Heff = torch.empty(T_len, device=device)
        fwd  = np.empty(T_len)
        rev  = np.empty(T_len)
        js   = np.empty(T_len)

        for s in range(0, T_len, CHUNK):
            e = min(s + CHUNK, T_len)
            chunk = logits[s:e]
            lp = torch.log_softmax(chunk, dim=-1)
            p = lp.exp()
            H[s:e]    = -(p * lp).sum(-1)
            Heff[s:e] = -(p * p * lp).sum(-1)
            del p, lp

            mu  = chunk.mean(-1, keepdim=True)
            cen = chunk - mu
            std = cen.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=eps)
            z_np = (cen / std).cpu().numpy()
            for i in range(e - s):
                fwd[s+i], rev[s+i], js[s+i] = _divergences(z_np[i])
            del chunk, cen

        sH = H.sum()
        r = Heff / (H + eps)
        log_r = torch.log(r.clamp(min=eps))
        R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

        log_Heff = torch.log(Heff.clamp(min=eps))
        log_H = torch.log(H.clamp(min=eps))
        R_alpha042 = float(torch.exp((H * (log_Heff - 0.42*log_H)).sum() / (sH + eps)).item())

        rows.append({
            'tau': tau,
            'neg_fwd': -float(np.mean(fwd)),
            'neg_rev': -float(np.mean(rev)),
            'neg_js':  -float(np.mean(js)),
            'R_geo': R_geo,
            'R_alpha042': R_alpha042,
        })

    del logits_orig, out
    torch.cuda.empty_cache()
    return rows


def grpo_adv(r):
    r = np.array(r, dtype=float); s = r.std()
    return (r - r.mean()) / (s + eps) if s > eps else np.zeros_like(r)

def cohens_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2: return 0.0
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading ...")
    with open(FILE) as f:
        data = json.load(f)
    rd = data['results']; N = len(rd)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    KEYS = ['neg_fwd', 'neg_rev', 'neg_js', 'R_geo', 'R_alpha042']
    LABELS = {
        'neg_fwd':    '-KL(emp||G)',
        'neg_rev':    '-KL(G||emp)',
        'neg_js':     '-JS(emp,G)',
        'R_geo':      'R_geo',
        'R_alpha042': 'R_alpha(0.42)',
    }

    all_c  = {k: [] for k in KEYS}
    all_w  = {k: [] for k in KEYS}
    all_ac = {k: [] for k in KEYS}
    all_aw = {k: [] for k in KEYS}

    for qi, res in enumerate(rd):
        prompt = res['prompt']
        K = len(res['responses'])
        labels = [1 if r.get('is_correct', False) else 0 for r in res['responses']]
        ml = [compute_metrics(model, tok, prompt, r.get('response', ''))
              for r in res['responses']]
        for mk in KEYS:
            vals = [m[mk] for m in ml]
            adv  = grpo_adv(vals)
            for i in range(K):
                (all_c if labels[i] else all_w)[mk].append(vals[i])
                (all_ac if labels[i] else all_aw)[mk].append(adv[i])
        if (qi+1) % 5 == 0 or qi == N-1:
            print(f"  [{qi+1}/{N}]")

    # ── table ──
    print("\n" + "="*115)
    print("  -KL(emp||G) vs -KL(G||emp) vs -JS vs R_geo vs R_alpha(0.42)")
    print("="*115)
    print(f"  {'指标':<14s} │ {'Cohen d':>9s} │ {'Gap':>9s} │ {'P(+|C)':>7s} │ {'P(+|W)':>7s} │ {'正确均值':>14s} │ {'错误均值':>14s}")
    print("  " + "─"*85)

    summary = {}
    for mk in KEYS:
        d   = cohens_d(all_c[mk], all_w[mk])
        gap = np.mean(all_ac[mk]) - np.mean(all_aw[mk])
        pc  = np.mean(np.array(all_ac[mk]) > 0)
        pw  = np.mean(np.array(all_aw[mk]) > 0)
        mc, mw = np.mean(all_c[mk]), np.mean(all_w[mk])
        print(f"  {LABELS[mk]:<14s} │ {d:>+9.4f} │ {gap:>+9.4f} │ {pc:>7.3f} │ {pw:>7.3f} │ {mc:>14.6f} │ {mw:>14.6f}")
        summary[mk] = dict(d=d, gap=gap, pc=pc, pw=pw, mc=mc, mw=mw)

    # ── collapse ──
    print("\n  防坍缩模拟 (6 样本):")
    temps = [1.0, 0.7, 0.5, 0.3]
    sharp = {k: {t: [] for t in temps} for k in KEYS}
    for qi in range(min(6, N)):
        sr = simulate_sharpening(model, tok, rd[qi]['prompt'],
                                  rd[qi]['responses'][0].get('response', ''))
        if sr:
            for e in sr:
                for k in KEYS:
                    sharp[k][e['tau']].append(e[k])

    print(f"\n  {'指标':<14s} │ {'tau=1.0':>12s} │ {'tau=0.7':>12s} │ {'tau=0.5':>12s} │ {'tau=0.3':>12s} │ {'Delta':>8s}")
    print("  " + "─"*72)
    collapse = {}
    for k in KEYS:
        ms = [np.mean(sharp[k][t]) if sharp[k][t] else 0 for t in temps]
        delta = (ms[-1] - ms[0]) / (abs(ms[0]) + eps) * 100
        collapse[k] = delta
        print(f"  {LABELS[k]:<14s} │ {ms[0]:>12.6f} │ {ms[1]:>12.6f} │ {ms[2]:>12.6f} │ {ms[3]:>12.6f} │ {delta:>+7.1f}%")

    # ── final ──
    print("\n  ═══ 最终排名 (d>0 且 |delta|<2%) ═══")
    ranked = sorted(KEYS, key=lambda k: summary[k]['d'], reverse=True)
    for rank, k in enumerate(ranked, 1):
        d = summary[k]['d']
        delta = collapse[k]
        ideal = "★ IDEAL ★" if d > 0 and abs(delta) < 2 else \
                "✓ d>0" if d > 0 else ""
        print(f"  #{rank} {LABELS[k]:<14s}: d={d:>+.4f}, delta={delta:>+.1f}%, P(+|C)={summary[k]['pc']:.3f}  {ideal}")

    # ── plot ──
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    fig.suptitle("-KL / -RevKL / -JS  vs  R_geo / R_alpha(0.42)",
                 fontsize=15, fontweight='bold')

    COLORS = {'neg_fwd': '#E74C3C', 'neg_rev': '#9B59B6', 'neg_js': '#E67E22',
              'R_geo': '#3498DB', 'R_alpha042': '#2ECC71'}

    # d bars
    ax = axes[0]
    ds = [summary[k]['d'] for k in KEYS]
    bars = ax.bar(range(len(KEYS)), ds, color=[COLORS[k] for k in KEYS],
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    for bar, d in zip(bars, ds):
        ax.text(bar.get_x()+bar.get_width()/2, d + 0.02*(1 if d>=0 else -1),
                f'{d:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([LABELS[k] for k in KEYS], fontsize=9, rotation=20, ha='right')
    ax.set_ylabel("Cohen's d", fontsize=13, fontweight='bold')
    ax.set_title("Discrimination (d > 0 = correct)", fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # delta bars
    ax1 = axes[1]
    deltas = [collapse[k] for k in KEYS]
    c1 = ['#2ECC71' if abs(d)<2 else ('#E74C3C' if d>0 else '#3498DB') for d in deltas]
    bars1 = ax1.bar(range(len(KEYS)), deltas, color=c1, edgecolor='black', linewidth=0.8, alpha=0.85)
    for bar, d in zip(bars1, deltas):
        ax1.text(bar.get_x()+bar.get_width()/2, d + 0.3*(1 if d>=0 else -1),
                 f'{d:+.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(len(KEYS)))
    ax1.set_xticklabels([LABELS[k] for k in KEYS], fontsize=9, rotation=20, ha='right')
    ax1.axhline(0, color='black', linewidth=2)
    ax1.set_ylabel('Delta (%)', fontsize=13, fontweight='bold')
    ax1.set_title("Anti-collapse (green = stable)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # trajectory
    ax2 = axes[2]
    for k in KEYS:
        ms = [np.mean(sharp[k][t]) if sharp[k][t] else 0 for t in temps]
        base = ms[0] if abs(ms[0]) > eps else 1
        norm = [m / (abs(base) + eps) for m in ms]
        ax2.plot(temps, norm, color=COLORS[k], linewidth=2.5, marker='o',
                 markersize=9, label=LABELS[k], alpha=0.85)
    ax2.axhline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax2.invert_xaxis()
    ax2.set_xlabel('Temperature', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Normalized reward', fontsize=13, fontweight='bold')
    ax2.set_title("Sharpening trajectory", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'neg_divergence_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
