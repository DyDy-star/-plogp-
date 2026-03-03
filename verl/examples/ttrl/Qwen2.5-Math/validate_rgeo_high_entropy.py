#!/usr/bin/env python3
"""
验证: 只对高熵 token 计算 R_geo

核心思想: 低熵位置的锐化是"无意义的坍缩", 高熵位置的效率才是"真正的推理质量"

候选:
  R_geo_high:  geo_mean_H(r_t) 仅限 H_t >= H̄ 的 token
  R_geo_low:   geo_mean_H(r_t) 仅限 H_t < H̄ 的 token
  R_geo_all:   标准 R_geo (全部 token)

也测试不同阈值:
  H̄ (均值), median(H), 固定分位数 (25%, 50%, 75%)
"""

import json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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


@torch.no_grad()
def compute_split_metrics(model, tok, prompt, response, device=DEV):
    """计算分区 R_geo: 高熵/低熵/全部"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()
    T = logits.size(0)

    empty = {k: 0.0 for k in [
        'R_geo_all', 'R_geo_high_mean', 'R_geo_low_mean',
        'R_geo_high_median', 'R_geo_high_q25', 'R_geo_high_q75',
        'n_high_mean', 'n_low_mean', 'mean_H', 'mean_lam',
        'mean_H_high', 'mean_H_low', 'mean_r_high', 'mean_r_low',
    ]}
    if T < 3:
        return empty

    H = torch.empty(T, device=device)
    Heff = torch.empty(T, device=device)
    lam_arr = torch.empty(T, device=device)
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        lp = torch.log_softmax(logits[s:e], dim=-1)
        p = lp.exp()
        H[s:e] = -(p * lp).sum(-1)
        Heff[s:e] = -(p * p * lp).sum(-1)
        lam_arr[s:e] = (p * p).sum(-1)
        del p, lp

    r = Heff / (H + eps)       # r_t = H_eff_t / H_t
    log_r = torch.log(r.clamp(min=eps))

    # ── 标准 R_geo (全部) ──
    sH = H.sum()
    R_geo_all = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    # ── 各种阈值分割 ──
    results = {'R_geo_all': R_geo_all, 'mean_H': float(H.mean().item()),
               'mean_lam': float(lam_arr.mean().item())}

    H_mean = H.mean()
    H_median = H.median()
    H_q25 = torch.quantile(H, 0.25)
    H_q75 = torch.quantile(H, 0.75)

    thresholds = {
        'mean': H_mean,
        'median': H_median,
        'q25': H_q25,
        'q75': H_q75,
    }

    for name, thresh in thresholds.items():
        mask_high = H >= thresh
        mask_low  = H < thresh

        n_high = int(mask_high.sum().item())
        n_low  = int(mask_low.sum().item())

        if n_high > 1:
            sH_high = H[mask_high].sum()
            R_high = float(torch.exp(
                (H[mask_high] * log_r[mask_high]).sum() / (sH_high + eps)
            ).item())
            results[f'R_geo_high_{name}'] = R_high
            results[f'mean_H_high_{name}'] = float(H[mask_high].mean().item())
            results[f'mean_r_high_{name}'] = float(r[mask_high].mean().item())
        else:
            results[f'R_geo_high_{name}'] = 0.0
            results[f'mean_H_high_{name}'] = 0.0
            results[f'mean_r_high_{name}'] = 0.0

        if n_low > 1:
            sH_low = H[mask_low].sum()
            R_low = float(torch.exp(
                (H[mask_low] * log_r[mask_low]).sum() / (sH_low + eps)
            ).item())
            results[f'R_geo_low_{name}'] = R_low
            results[f'mean_H_low_{name}'] = float(H[mask_low].mean().item())
            results[f'mean_r_low_{name}'] = float(r[mask_low].mean().item())
        else:
            results[f'R_geo_low_{name}'] = 0.0
            results[f'mean_H_low_{name}'] = 0.0
            results[f'mean_r_low_{name}'] = 0.0

        results[f'n_high_{name}'] = n_high
        results[f'n_low_{name}'] = n_low

    del logits, H, Heff, r, out
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def simulate_sharpening(model, tok, prompt, response, temps=[1.0, 0.7, 0.5, 0.3], device=DEV):
    """温度缩放测试"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()
    T = logits_orig.size(0)
    if T < 3:
        return None

    all_results = []
    for tau in temps:
        logits = logits_orig / tau
        H = torch.empty(T, device=device)
        Heff = torch.empty(T, device=device)
        for s in range(0, T, CHUNK):
            e = min(s + CHUNK, T)
            lp = torch.log_softmax(logits[s:e], dim=-1)
            p = lp.exp()
            H[s:e] = -(p * lp).sum(-1)
            Heff[s:e] = -(p * p * lp).sum(-1)
            del p, lp

        r = Heff / (H + eps)
        log_r = torch.log(r.clamp(min=eps))

        # 全部
        sH = H.sum()
        R_all = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

        # 高熵 (> 均值)
        H_mean = H.mean()
        mask_high = H >= H_mean
        n_high = int(mask_high.sum().item())
        if n_high > 1:
            sH_high = H[mask_high].sum()
            R_high = float(torch.exp(
                (H[mask_high] * log_r[mask_high]).sum() / (sH_high + eps)
            ).item())
        else:
            R_high = 0.0

        # 高熵 (> median)
        H_med = H.median()
        mask_med = H >= H_med
        n_med = int(mask_med.sum().item())
        if n_med > 1:
            sH_med = H[mask_med].sum()
            R_high_med = float(torch.exp(
                (H[mask_med] * log_r[mask_med]).sum() / (sH_med + eps)
            ).item())
        else:
            R_high_med = 0.0

        all_results.append({
            'tau': tau, 'R_geo_all': R_all,
            'R_geo_high_mean': R_high, 'R_geo_high_median': R_high_med,
            'mean_H': float(H.mean().item()),
            'n_high_mean': n_high, 'n_high_median': n_med,
            'frac_high_mean': n_high / T, 'frac_high_median': n_med / T,
        })

    del logits_orig, out
    torch.cuda.empty_cache()
    return all_results


def grpo_advantage(rewards):
    r = np.array(rewards, dtype=float)
    m, s = r.mean(), r.std()
    return (r - m) / (s + eps)


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
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

    # ═══ 收集所有指标 ═══
    METRIC_KEYS = [
        'R_geo_all',
        'R_geo_high_mean', 'R_geo_low_mean',
        'R_geo_high_median', 'R_geo_low_median',
        'R_geo_high_q25', 'R_geo_low_q25',
        'R_geo_high_q75', 'R_geo_low_q75',
    ]
    all_correct = {k: [] for k in METRIC_KEYS}
    all_wrong   = {k: [] for k in METRIC_KEYS}
    all_adv_c = {k: [] for k in METRIC_KEYS}
    all_adv_w = {k: [] for k in METRIC_KEYS}

    # 也收集一些调试信息
    frac_high_correct = []
    frac_high_wrong = []

    for qi, res in enumerate(results_data):
        prompt = res['prompt']
        responses = res['responses']
        K = len(responses)
        true_labels = [1 if r.get('is_correct', False) else 0 for r in responses]

        ml = []
        for r in responses:
            m = compute_split_metrics(model, tok, prompt, r.get('response', ''))
            ml.append(m)

        for mk in METRIC_KEYS:
            vals = [m.get(mk, 0.0) for m in ml]
            adv = grpo_advantage(vals)
            for i in range(K):
                if true_labels[i]:
                    all_correct[mk].append(vals[i])
                    all_adv_c[mk].append(adv[i])
                else:
                    all_wrong[mk].append(vals[i])
                    all_adv_w[mk].append(adv[i])

        for i in range(K):
            n_h = ml[i].get('n_high_mean', 0)
            n_tot = n_h + ml[i].get('n_low_mean', 0)
            frac = n_h / max(n_tot, 1)
            if true_labels[i]:
                frac_high_correct.append(frac)
            else:
                frac_high_wrong.append(frac)

        if (qi + 1) % 10 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}]")

    # ═══ 统计表 ═══
    print("\n" + "="*110)
    print("  分区 R_geo 验证: 只对高熵 token 计算")
    print("="*110)
    print(f"  {'指标':<25s} │ {'Cohen d':>9s} │ {'Adv Gap':>9s} │ {'P(+|C)':>8s} │ {'P(+|W)':>8s} │ {'Gap':>8s} │ {'正确均值':>10s} │ {'错误均值':>10s}")
    print("  " + "─"*100)

    summary = {}
    for mk in METRIC_KEYS:
        rc, rw = all_correct[mk], all_wrong[mk]
        ac, aw = all_adv_c[mk], all_adv_w[mk]
        d = cohens_d(rc, rw)
        gap = np.mean(ac) - np.mean(aw) if ac and aw else 0
        pc = np.mean(np.array(ac) > 0) if ac else 0
        pw = np.mean(np.array(aw) > 0) if aw else 0
        mc = np.mean(rc) if rc else 0
        mw = np.mean(rw) if rw else 0
        print(f"  {mk:<25s} │ {d:>+9.4f} │ {gap:>9.4f} │ {pc:>8.3f} │ {pw:>8.3f} │ {pc-pw:>+8.3f} │ {mc:>10.5f} │ {mw:>10.5f}")
        summary[mk] = {'d': d, 'gap': gap, 'pc': pc, 'pw': pw, 'mc': mc, 'mw': mw}

    print(f"\n  高熵 token 比例: correct={np.mean(frac_high_correct):.3f}, wrong={np.mean(frac_high_wrong):.3f}")

    # ═══ 防坍缩模拟 ═══
    print("\n" + "="*110)
    print("  防坍缩模拟 (温度缩放)")
    print("="*110)

    sharp_keys = ['R_geo_all', 'R_geo_high_mean', 'R_geo_high_median']
    all_sharp = {k: [] for k in sharp_keys}
    all_frac = {'frac_high_mean': [], 'frac_high_median': []}

    n_sim = min(8, N_Q)
    for qi in range(n_sim):
        res = results_data[qi]
        sr = simulate_sharpening(model, tok, res['prompt'],
                                  res['responses'][0].get('response', ''))
        if sr:
            for entry in sr:
                for k in sharp_keys:
                    all_sharp[k].append((entry['tau'], entry.get(k, 0)))
                for k in ['frac_high_mean', 'frac_high_median']:
                    all_frac[k].append((entry['tau'], entry.get(k, 0)))

    # 按 tau 聚合
    temps = [1.0, 0.7, 0.5, 0.3]
    print(f"\n  {'指标':<25s} │ {'tau=1.0':>10s} │ {'tau=0.7':>10s} │ {'tau=0.5':>10s} │ {'tau=0.3':>10s} │ {'Delta':>10s}")
    print("  " + "─"*80)

    collapse_delta = {}
    for k in sharp_keys:
        vals_by_tau = {}
        for tau, v in all_sharp[k]:
            vals_by_tau.setdefault(tau, []).append(v)
        means = [np.mean(vals_by_tau.get(t, [0])) for t in temps]
        delta = (means[-1] - means[0]) / (abs(means[0]) + eps) * 100
        collapse_delta[k] = delta
        print(f"  {k:<25s} │ {means[0]:>10.5f} │ {means[1]:>10.5f} │ {means[2]:>10.5f} │ {means[3]:>10.5f} │ {delta:>+9.1f}%")

    # 高熵比例随温度变化
    print(f"\n  高熵 token 比例随温度变化:")
    for k in ['frac_high_mean', 'frac_high_median']:
        vals_by_tau = {}
        for tau, v in all_frac[k]:
            vals_by_tau.setdefault(tau, []).append(v)
        means = [np.mean(vals_by_tau.get(t, [0])) for t in temps]
        print(f"  {k:<25s} │ {means[0]:>10.3f} │ {means[1]:>10.3f} │ {means[2]:>10.3f} │ {means[3]:>10.3f}")

    # ═══ 综合判断 ═══
    print("\n" + "="*110)
    print("  综合判断: d > 0 且 delta < 0?")
    print("="*110)
    for mk in ['R_geo_all', 'R_geo_high_mean', 'R_geo_high_median',
                'R_geo_high_q25', 'R_geo_high_q75',
                'R_geo_low_mean', 'R_geo_low_median']:
        d = summary.get(mk, {}).get('d', 0)
        delta = collapse_delta.get(mk, float('nan'))
        d_ok = "d>0" if d > 0 else "d<0"
        c_ok = "ANTI" if delta < 0 else ("PRO" if not math.isnan(delta) else "?")
        both = "*** YES ***" if d > 0 and delta < 0 else ""
        print(f"  {mk:<25s}: d={d:>+.4f} [{d_ok}], delta={delta:>+.1f}% [{c_ok}]  {both}")

    # ═══ 绘图 ═══
    _plot_results(summary, collapse_delta, sharp_keys, all_sharp, temps, OUT_DIR)
    print(f"\nDone. Plots saved to {OUT_DIR}/")


def _plot_results(summary, collapse_delta, sharp_keys, all_sharp, temps, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("R_geo on High-Entropy Tokens Only: Discrimination vs Anti-Collapse",
                 fontsize=15, fontweight='bold')

    # (0): Cohen's d comparison
    ax = axes[0]
    plot_keys = ['R_geo_all', 'R_geo_high_mean', 'R_geo_high_median',
                 'R_geo_high_q25', 'R_geo_high_q75',
                 'R_geo_low_mean', 'R_geo_low_median']
    ds = [summary.get(k, {}).get('d', 0) for k in plot_keys]
    colors = ['#3498DB'] + ['#2ECC71']*4 + ['#E74C3C']*2
    bars = ax.barh(range(len(plot_keys)), ds, color=colors, edgecolor='black', linewidth=1, alpha=0.85)
    ax.set_yticks(range(len(plot_keys)))
    ax.set_yticklabels([k.replace('R_geo_', '') for k in plot_keys], fontsize=10)
    ax.axvline(0, color='black', linewidth=2)
    for i, d in enumerate(ds):
        ax.text(d + 0.02, i, f'{d:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel("Cohen's d", fontsize=13, fontweight='bold')
    ax.set_title("Discrimination Power\nBlue=all, Green=high-H only, Red=low-H only", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # (1): Collapse delta
    ax1 = axes[1]
    ck = ['R_geo_all', 'R_geo_high_mean', 'R_geo_high_median']
    deltas = [collapse_delta.get(k, 0) for k in ck]
    colors1 = ['#E74C3C' if d > 0 else '#2ECC71' for d in deltas]
    bars1 = ax1.bar(range(len(ck)), deltas, color=colors1, edgecolor='black', linewidth=1, alpha=0.85)
    for bar, d in zip(bars1, deltas):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{d:+.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(ck)))
    ax1.set_xticklabels([k.replace('R_geo_', '') for k in ck], fontsize=11)
    ax1.axhline(0, color='black', linewidth=2)
    ax1.set_ylabel('Reward change at tau=0.3 (%)', fontsize=13, fontweight='bold')
    ax1.set_title("Anti-Collapse Test\nGreen=resists, Red=encourages", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # (2): Reward trajectory under sharpening
    ax2 = axes[2]
    for k, color, ls, label in [
        ('R_geo_all', '#3498DB', '--', 'R_geo (all tokens)'),
        ('R_geo_high_mean', '#2ECC71', '-', 'R_geo (H > mean only)'),
        ('R_geo_high_median', '#E67E22', '-', 'R_geo (H > median only)'),
    ]:
        vals_by_tau = {}
        for tau, v in all_sharp[k]:
            vals_by_tau.setdefault(tau, []).append(v)
        means = [np.mean(vals_by_tau.get(t, [0])) for t in temps]
        base = means[0] if means[0] != 0 else 1
        norm = [m / (abs(base) + eps) for m in means]
        ax2.plot(temps, norm, color=color, linewidth=3, linestyle=ls,
                 marker='o', markersize=10, label=label, alpha=0.9)

    ax2.axhline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax2.invert_xaxis()
    ax2.set_xlabel('Temperature (1.0=normal, <1.0=sharpened)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Normalized reward (relative to tau=1.0)', fontsize=13, fontweight='bold')
    ax2.set_title("Reward under sharpening\nUp=pro-collapse, Down=anti-collapse", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'rgeo_high_entropy_validation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
