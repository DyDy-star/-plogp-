#!/usr/bin/env python3
"""
验证候选奖励函数:
  A. mean(H_eff)        — 有效熵均值 (防坍缩)
  B. H_eff flow balance — 无损熵周转 (平衡探索/收敛)
  C. 组合 A × B

对比基线:
  - R_geo (几何平均 H_eff/H)
  - R_brake (几何平均 r(1-r))
  - MajVote (多数投票 0/1)
  - SC (-mean(H))

指标:
  - Cohen's d (正确 vs 错误)
  - GRPO 优势值差距
  - P(adv>0 | correct)
  - 防坍缩模拟 (均匀锐化后的奖励变化)
"""

import json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

MODEL = "/data/user5/models/Qwen2.5-Math-1.5B"
FILE  = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json"
OUT_DIR = "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/grpo_analysis"
DEV = "cuda"
eps = 1e-8
CHUNK = 256


# ════════════════════════════════════════════════
#  核心计算: 所有候选指标
# ════════════════════════════════════════════════
@torch.no_grad()
def compute_all_metrics(model, tok, prompt, response, device=DEV):
    """计算单个 response 的所有候选奖励指标"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()   # (T, V)
    T = logits.size(0)

    empty = {'R_geo': 0.0, 'R_brake': 0.0, 'R_arith': 0.0,
             'mean_Heff': 0.0, 'flow_balance': 0.0, 'combined': 0.0,
             'mean_lam': 0.0, 'mean_H': 0.0, 'SC': 0.0,
             'Heff_list': [], 'H_list': []}
    if T < 3:
        return empty

    H = torch.empty(T, device=device)
    Heff = torch.empty(T, device=device)
    sum_lam = 0.0
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        lp = torch.log_softmax(logits[s:e], dim=-1)
        p = lp.exp()
        H[s:e] = -(p * lp).sum(-1)
        Heff[s:e] = -(p * p * lp).sum(-1)
        lam = (p * p).sum(-1)
        sum_lam += float(lam.sum().item())
        del p, lp, lam

    sH = H.sum()
    if sH < eps:
        return empty

    r = Heff / (H + eps)

    # ── 基线指标 ──
    log_r = torch.log(r.clamp(min=eps))
    R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    r_brake_t = r * (1 - r)
    log_rb = torch.log(r_brake_t.clamp(min=eps))
    R_brake = float(torch.exp((H * log_rb).sum() / (sH + eps)).item())

    R_arith = float((Heff.sum() / (sH + eps)).item())

    # ── 候选 A: mean(H_eff) ──
    mean_Heff = float(Heff.sum().item()) / max(T, 1)

    # ── 候选 B: H_eff flow balance ──
    if T > 1:
        dHeff = Heff[1:] - Heff[:-1]               # (T-1,)
        F_up = float(dHeff.clamp(min=0).sum().item())
        F_down = float((-dHeff).clamp(min=0).sum().item())
        flow_total = F_up + F_down
        flow_balance = 2.0 * min(F_up, F_down) / (flow_total + eps)
    else:
        F_up = F_down = 0.0
        flow_balance = 0.0

    # ── 候选 C: 组合 ──
    combined = mean_Heff * flow_balance

    mean_lam = sum_lam / max(T, 1)
    mean_H = float(H.mean().item())
    SC = -mean_H

    # 采样 H_eff 和 H 用于绘图 (最多 200 个点)
    step = max(1, T // 200)
    Heff_list = Heff[::step].cpu().tolist()
    H_list = H[::step].cpu().tolist()

    del logits, H, Heff, r, r_brake_t, out
    torch.cuda.empty_cache()

    return {
        'R_geo': R_geo, 'R_brake': R_brake, 'R_arith': R_arith,
        'mean_Heff': mean_Heff, 'flow_balance': flow_balance, 'combined': combined,
        'mean_lam': mean_lam, 'mean_H': mean_H, 'SC': SC,
        'F_up': F_up, 'F_down': F_down,
        'Heff_list': Heff_list, 'H_list': H_list,
    }


def majority_vote(responses):
    answers = [r.get('extracted_answer', '') for r in responses]
    valid = [a for a in answers if a is not None and str(a).strip() != '']
    if not valid:
        return [0] * len(responses), None
    counter = Counter(valid)
    majority_ans = counter.most_common(1)[0][0]
    pseudo = [1 if str(a).strip() == str(majority_ans).strip() else 0 for a in answers]
    return pseudo, majority_ans


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


# ════════════════════════════════════════════════
#  防坍缩模拟: 均匀锐化后指标如何变化
# ════════════════════════════════════════════════
@torch.no_grad()
def simulate_sharpening(model, tok, prompt, response, temps=[1.0, 0.7, 0.5, 0.3], device=DEV):
    """模拟不同温度 (均匀锐化) 下各指标的变化"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits_orig = out.logits[0, pl-1:-1].float()   # (T, V)
    T = logits_orig.size(0)
    if T < 3:
        return None

    results = []
    for tau in temps:
        logits = logits_orig / tau
        H = torch.empty(T, device=device)
        Heff = torch.empty(T, device=device)
        sum_lam = 0.0
        for s in range(0, T, CHUNK):
            e = min(s + CHUNK, T)
            lp = torch.log_softmax(logits[s:e], dim=-1)
            p = lp.exp()
            H[s:e] = -(p * lp).sum(-1)
            Heff[s:e] = -(p * p * lp).sum(-1)
            lam = (p * p).sum(-1)
            sum_lam += float(lam.sum().item())
            del p, lp, lam

        sH = H.sum()
        r = Heff / (H + eps)
        log_r = torch.log(r.clamp(min=eps))
        R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())
        R_arith = float((Heff.sum() / (sH + eps)).item())
        mean_Heff = float(Heff.sum().item()) / max(T, 1)
        mean_lam = sum_lam / max(T, 1)
        mean_H = float(H.mean().item())

        if T > 1:
            dHeff = Heff[1:] - Heff[:-1]
            F_up = float(dHeff.clamp(min=0).sum().item())
            F_down = float((-dHeff).clamp(min=0).sum().item())
            flow_balance = 2.0 * min(F_up, F_down) / (F_up + F_down + eps)
        else:
            flow_balance = 0.0

        results.append({
            'tau': tau, 'R_geo': R_geo, 'R_arith': R_arith,
            'mean_Heff': mean_Heff, 'flow_balance': flow_balance,
            'combined': mean_Heff * flow_balance,
            'mean_lam': mean_lam, 'mean_H': mean_H,
        })

    del logits_orig, out
    torch.cuda.empty_cache()
    return results


# ════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("加载数据 ...")
    with open(FILE) as f:
        data = json.load(f)
    results = data['results']
    N_Q = len(results)
    print(f"  {N_Q} 题, {results[0]['n_samples']} 回答/题")

    print("加载模型 ...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # ── 遍历所有问题, 计算指标 ──
    METRICS = ['R_geo', 'R_brake', 'R_arith', 'mean_Heff', 'flow_balance', 'combined', 'SC']
    all_correct = {m: [] for m in METRICS}
    all_wrong   = {m: [] for m in METRICS}
    all_adv_correct = {m: [] for m in METRICS}
    all_adv_wrong   = {m: [] for m in METRICS}
    adv_mv_correct, adv_mv_wrong = [], []

    # 收集 H_eff 分布
    heff_correct, heff_wrong = [], []

    for qi, res in enumerate(results):
        prompt = res['prompt']
        gt = str(res['ground_truth'])
        responses = res['responses']
        K = len(responses)

        pseudo, majority_ans = majority_vote(responses)
        true_labels = [1 if r.get('is_correct', False) else 0 for r in responses]

        # 计算所有指标
        metrics_list = []
        for ri, r in enumerate(responses):
            m = compute_all_metrics(model, tok, prompt, r.get('response', ''))
            metrics_list.append(m)
            key = 'correct' if true_labels[ri] else 'wrong'
            if key == 'correct':
                heff_correct.extend(m['Heff_list'])
            else:
                heff_wrong.extend(m['Heff_list'])

        # GRPO 优势值
        for mk in METRICS:
            vals = [m[mk] for m in metrics_list]
            adv = grpo_advantage(vals)
            for i in range(K):
                if true_labels[i]:
                    all_correct[mk].append(vals[i])
                    all_adv_correct[mk].append(adv[i])
                else:
                    all_wrong[mk].append(vals[i])
                    all_adv_wrong[mk].append(adv[i])

        adv_mv = grpo_advantage(pseudo)
        for i in range(K):
            if true_labels[i]:
                adv_mv_correct.append(adv_mv[i])
            else:
                adv_mv_wrong.append(adv_mv[i])

        if (qi + 1) % 5 == 0 or qi == N_Q - 1:
            mg = metrics_list[0]
            print(f"  [{qi+1}/{N_Q}] correct={sum(true_labels)}/{K} "
                  f"mean_Heff=[{min(m['mean_Heff'] for m in metrics_list):.5f},"
                  f"{max(m['mean_Heff'] for m in metrics_list):.5f}] "
                  f"flow_bal=[{min(m['flow_balance'] for m in metrics_list):.4f},"
                  f"{max(m['flow_balance'] for m in metrics_list):.4f}]")

    # ════════════════════════════════════════════
    #  统计表
    # ════════════════════════════════════════════
    print("\n" + "="*90)
    print("  候选奖励函数验证结果")
    print("="*90)

    header = f"  {'指标':<22s} │ {'Cohen d':>8s} │ {'优势差距':>8s} │ {'P(+|正确)':>9s} │ {'P(+|错误)':>9s} │ {'正确均值':>10s} │ {'错误均值':>10s}"
    print(header)
    print("  " + "─"*86)

    summary = {}
    for mk in METRICS + ['MajVote']:
        if mk == 'MajVote':
            ac, aw = adv_mv_correct, adv_mv_wrong
            rc, rw = [1.0]*len(ac), [0.0]*len(aw)  # placeholder
            d_val = cohens_d(ac, aw)
        else:
            ac, aw = all_adv_correct[mk], all_adv_wrong[mk]
            rc, rw = all_correct[mk], all_wrong[mk]
            d_val = cohens_d(rc, rw)

        gap = np.mean(ac) - np.mean(aw)
        p_pos_c = np.mean(np.array(ac) > 0) if ac else 0
        p_pos_w = np.mean(np.array(aw) > 0) if aw else 0
        mc = np.mean(rc) if rc else 0
        mw = np.mean(rw) if rw else 0

        print(f"  {mk:<22s} │ {d_val:>+8.4f} │ {gap:>8.4f} │ {p_pos_c:>9.3f} │ {p_pos_w:>9.3f} │ {mc:>10.5f} │ {mw:>10.5f}")
        summary[mk] = {'d': d_val, 'gap': gap, 'p_pos_c': p_pos_c, 'p_pos_w': p_pos_w,
                        'mean_c': mc, 'mean_w': mw}

    print("  " + "─"*86)

    # Kendall tau 对比
    print("\n  Kendall tau (排序一致性):")
    for mk in ['mean_Heff', 'flow_balance', 'combined']:
        all_r = all_correct[mk] + all_wrong[mk]
        all_geo = all_correct['R_geo'] + all_wrong['R_geo']
        if len(all_r) > 10:
            tau, p = stats.kendalltau(all_r, all_geo)
            print(f"    tau({mk}, R_geo) = {tau:.4f}  p={p:.2e}")

    # ════════════════════════════════════════════
    #  防坍缩模拟
    # ════════════════════════════════════════════
    print("\n\n  防坍缩模拟 (均匀锐化, 温度缩放):")
    print("  选取第一题第一个 response 进行模拟 ...")

    res0 = results[0]
    sharp_results = simulate_sharpening(model, tok, res0['prompt'],
                                         res0['responses'][0].get('response', ''))
    if sharp_results:
        print(f"\n  {'温度':<6s} │ {'R_geo':>8s} │ {'R_arith':>8s} │ {'mean_Heff':>10s} │ {'flow_bal':>9s} │ {'组合':>10s} │ {'mean_lam':>9s} │ {'mean_H':>8s}")
        print("  " + "─"*85)
        for sr in sharp_results:
            print(f"  {sr['tau']:<6.2f} │ {sr['R_geo']:>8.5f} │ {sr['R_arith']:>8.5f} │ {sr['mean_Heff']:>10.6f} │ {sr['flow_balance']:>9.5f} │ {sr['combined']:>10.7f} │ {sr['mean_lam']:>9.5f} │ {sr['mean_H']:>8.4f}")

        # 计算变化率
        base = sharp_results[0]
        print(f"\n  相对于温度=1.0 的变化率:")
        print(f"  {'温度':<6s} │ {'R_geo':>10s} │ {'mean_Heff':>12s} │ {'flow_bal':>12s} │ {'组合':>12s}")
        print("  " + "─"*60)
        for sr in sharp_results[1:]:
            def pct(v, b):
                return (v - b) / (abs(b) + eps) * 100
            print(f"  {sr['tau']:<6.2f} │ {pct(sr['R_geo'], base['R_geo']):>+9.1f}% │ {pct(sr['mean_Heff'], base['mean_Heff']):>+11.1f}% │ {pct(sr['flow_balance'], base['flow_balance']):>+11.1f}% │ {pct(sr['combined'], base['combined']):>+11.1f}%")

    # ════════════════════════════════════════════
    #  绘图
    # ════════════════════════════════════════════
    print("\n生成图表 ...")
    _plot_main_comparison(summary, all_correct, all_wrong, all_adv_correct, all_adv_wrong,
                          adv_mv_correct, adv_mv_wrong, heff_correct, heff_wrong,
                          sharp_results, OUT_DIR)

    print(f"\n所有图表已保存至: {OUT_DIR}/")


def _plot_main_comparison(summary, all_correct, all_wrong, all_adv_correct, all_adv_wrong,
                          adv_mv_c, adv_mv_w, heff_c, heff_w, sharp_results, out_dir):
    """主对比图: 2×3 布局"""
    fig, axes = plt.subplots(2, 3, figsize=(26, 14))
    fig.suptitle("候选奖励函数验证: mean(H_eff) vs flow_balance vs R_geo vs R_brake\n"
                 "核心问题: 不除以 H 是否能防坍缩且保持区分力?",
                 fontsize=16, fontweight='bold', y=1.01)

    METRICS_PLOT = ['R_geo', 'R_brake', 'mean_Heff', 'flow_balance', 'combined', 'MajVote']
    COLORS = {'R_geo': '#3498DB', 'R_brake': '#9B59B6', 'mean_Heff': '#E74C3C',
              'flow_balance': '#2ECC71', 'combined': '#E67E22', 'MajVote': '#95A5A6', 'SC': '#7F8C8D'}

    # ═══ (0,0): Cohen's d 柱状图 ═══
    ax = axes[0, 0]
    names = METRICS_PLOT
    ds = [summary[m]['d'] for m in names]
    colors = [COLORS[m] for m in names]
    bars = ax.bar(range(len(names)), ds, color=colors, edgecolor='black', linewidth=1, alpha=0.85)
    for bar, d in zip(bars, ds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{d:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
    ax.set_ylabel("Cohen's d", fontsize=12, fontweight='bold')
    ax.set_title("区分力 (Cohen's d, 正确 vs 错误)\n越高 = 越能区分正确/错误回答", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # ═══ (0,1): P(adv>0|correct) 柱状图 ═══
    ax1 = axes[0, 1]
    p_vals = [summary[m]['p_pos_c'] for m in names]
    bars = ax1.bar(range(len(names)), p_vals, color=colors, edgecolor='black', linewidth=1, alpha=0.85)
    for bar, p in zip(bars, p_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{p:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax1.axhline(0.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
    ax1.set_ylabel("P(adv>0 | correct)", fontsize=12, fontweight='bold')
    ax1.set_title("正确回答获正优势的概率\n>0.5 = 学习方向正确", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.3, 0.85)

    # ═══ (0,2): H_eff 理论曲线 + 当前位置 ═══
    ax2 = axes[0, 2]
    p_arr = np.linspace(0.001, 0.999, 500)
    # 单主模式近似: H_eff ≈ p^2 * log(1/p)
    heff_theory = p_arr**2 * np.log(1.0 / p_arr)
    h_theory = p_arr * np.log(1.0 / p_arr) + (1-p_arr) * np.log(1.0 / (1-p_arr + 1e-12))
    r_theory = heff_theory / (h_theory + 1e-12)

    ax2.plot(p_arr, heff_theory, 'r-', linewidth=3, label='H_eff = p^2 log(1/p)', alpha=0.9)
    ax2.plot(p_arr, r_theory * 0.18, 'b--', linewidth=2, label='r_t = H_eff/H (scaled)', alpha=0.7)

    # 标注当前工作点
    ax2.axvspan(0.85, 0.95, alpha=0.15, color='orange', label='当前工作区间')
    ax2.axvline(0.607, color='gray', linewidth=1, linestyle=':', alpha=0.5, label='H_eff 峰值 (p=1/sqrt(e))')

    peak_x = 1.0 / np.sqrt(np.e)
    peak_y = peak_x**2 * np.log(1.0/peak_x)
    ax2.scatter([peak_x], [peak_y], s=120, color='red', zorder=5, edgecolors='black', linewidth=2)
    ax2.annotate(f'峰值 ({peak_x:.3f}, {peak_y:.4f})', xy=(peak_x, peak_y),
                 xytext=(0.3, peak_y + 0.02), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate('当前区间:\nH_eff 已过峰值\n锐化 = 降低 H_eff\n= 天然防坍缩!',
                 xy=(0.9, 0.04), fontsize=10, fontweight='bold', color='#C0392B',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC'))

    ax2.set_xlabel('p_max (主模式概率)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('值', fontsize=12, fontweight='bold')
    ax2.set_title('H_eff vs r_t 理论曲线\nH_eff 在 p=0.6 处达峰值, r_t 单调递增', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # ═══ (1,0): mean(H_eff) 分布 correct vs wrong ═══
    ax3 = axes[1, 0]
    mc = all_correct['mean_Heff']
    mw = all_wrong['mean_Heff']
    all_vals = mc + mw
    bins = np.linspace(min(all_vals)*0.9, max(all_vals)*1.1, 50) if all_vals else np.linspace(0, 0.2, 50)
    ax3.hist(mc, bins=bins, alpha=0.7, color='#2ECC71', density=True,
             label=f'正确 (n={len(mc)}, mu={np.mean(mc):.5f})', edgecolor='none')
    ax3.hist(mw, bins=bins, alpha=0.5, color='#E74C3C', density=True,
             label=f'错误 (n={len(mw)}, mu={np.mean(mw):.5f})', edgecolor='none')
    ax3.axvline(np.mean(mc), color='#27AE60', linewidth=2.5)
    ax3.axvline(np.mean(mw), color='#C0392B', linewidth=2.5)
    ax3.set_xlabel('mean(H_eff)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('密度', fontsize=12, fontweight='bold')
    ax3.set_title('候选 A: mean(H_eff) 分布\n正确 vs 错误回答', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ═══ (1,1): flow_balance 分布 ═══
    ax4 = axes[1, 1]
    fc = all_correct['flow_balance']
    fw = all_wrong['flow_balance']
    all_fb = fc + fw
    bins_fb = np.linspace(min(all_fb)*0.9, max(all_fb)*1.1, 50) if all_fb else np.linspace(0, 1, 50)
    ax4.hist(fc, bins=bins_fb, alpha=0.7, color='#2ECC71', density=True,
             label=f'正确 (n={len(fc)}, mu={np.mean(fc):.4f})', edgecolor='none')
    ax4.hist(fw, bins=bins_fb, alpha=0.5, color='#E74C3C', density=True,
             label=f'错误 (n={len(fw)}, mu={np.mean(fw):.4f})', edgecolor='none')
    ax4.axvline(np.mean(fc), color='#27AE60', linewidth=2.5)
    ax4.axvline(np.mean(fw), color='#C0392B', linewidth=2.5)
    ax4.set_xlabel('flow_balance', fontsize=12, fontweight='bold')
    ax4.set_ylabel('密度', fontsize=12, fontweight='bold')
    ax4.set_title('候选 B: H_eff 流量平衡\n正确 vs 错误回答', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ═══ (1,2): 防坍缩模拟 ═══
    ax5 = axes[1, 2]
    if sharp_results:
        temps = [sr['tau'] for sr in sharp_results]
        base = sharp_results[0]

        for mk, label, color, ls in [
            ('R_geo', 'R_geo (基线)', '#3498DB', '--'),
            ('mean_Heff', 'mean(H_eff) (候选A)', '#E74C3C', '-'),
            ('flow_balance', 'flow_balance (候选B)', '#2ECC71', '-'),
            ('combined', '组合 A×B (候选C)', '#E67E22', '-'),
        ]:
            vals = [sr[mk] for sr in sharp_results]
            # 归一化到温度=1.0的值
            base_val = vals[0] if vals[0] != 0 else 1
            norm_vals = [v / (abs(base_val) + eps) for v in vals]
            ax5.plot(temps, norm_vals, color=color, linewidth=2.5, linestyle=ls,
                     marker='o', markersize=8, label=label, alpha=0.9)

        ax5.set_xlabel('温度 (1.0=正常, <1.0=锐化)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('归一化奖励值 (相对于 tau=1.0)', fontsize=12, fontweight='bold')
        ax5.set_title('防坍缩测试: 均匀锐化下的奖励变化\n上升=鼓励坍缩, 下降=抵抗坍缩', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(1.0, color='black', linewidth=1, linestyle=':', alpha=0.3)
        ax5.invert_xaxis()  # 温度从高到低

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, 'heff_candidates_validation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {path}")


if __name__ == "__main__":
    main()
