#!/usr/bin/env python3
"""
GRPO 组内优势值对比分析:
  - R_geo (当前无监督奖励) vs R_brake (自刹车) vs 多数投票 0-1 奖励
  - 伪标签 vs 真实标签
  - GRPO 优势值分布
  - 逐位置 r_t 分布与刹车效果分析

绘图风格参考 analyze_step_entropy.py
"""

import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 风格设置 (参考 analyze_step_entropy.py) ──
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
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
#  1. 计算 R_geo + R_brake
# ════════════════════════════════════════════════
@torch.no_grad()
def compute_r_geo(model, tok, prompt, response, device=DEV):
    """计算单个 response 的 R_geo, R_brake(geo_mean(r(1-r))), 及 r_t 分布"""
    full = prompt + response
    enc = tok(full, return_tensors='pt', truncation=True, max_length=4096).to(device)
    pl = tok(prompt, return_tensors='pt', truncation=True, max_length=4096)['input_ids'].size(1)
    out = model(**enc)
    logits = out.logits[0, pl-1:-1].float()   # (T, V)
    T = logits.size(0)
    if T < 3:
        return {'R_geo': 0.0, 'R_brake': 0.0, 'R_arith': 0.0,
                'mean_lam': 0.0, 'mean_H': 0.0,
                'r_t_above_05': 0.0, 'mean_r': 0.0, 'r_t_list': []}

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
        return {'R_geo': 0.0, 'R_brake': 0.0, 'R_arith': 0.0,
                'mean_lam': 0.0, 'mean_H': 0.0,
                'r_t_above_05': 0.0, 'mean_r': 0.0, 'r_t_list': []}

    r = Heff / (H + eps)                           # r_t

    # R_geo = geo_mean_H(r_t)
    log_r = torch.log(r.clamp(min=eps))
    R_geo = float(torch.exp((H * log_r).sum() / (sH + eps)).item())

    # R_brake = geo_mean_H(r_t * (1 - r_t))
    r_brake_t = r * (1 - r)                        # 逐位置刹车值
    log_rb = torch.log(r_brake_t.clamp(min=eps))
    R_brake = float(torch.exp((H * log_rb).sum() / (sH + eps)).item())

    R_arith = float((Heff.sum() / (sH + eps)).item())
    mean_lam = sum_lam / max(T, 1)
    mean_H = float(H.mean().item())

    # r_t 统计 (采样最多 200 个位置用于绘图)
    r_np = r.cpu().numpy()
    above_05 = float((r > 0.5).float().mean().item())
    mean_r = float(r.mean().item())
    step = max(1, len(r_np) // 200)
    r_t_list = r_np[::step].tolist()

    del logits, H, Heff, r, r_brake_t, out
    torch.cuda.empty_cache()
    return {'R_geo': R_geo, 'R_brake': R_brake, 'R_arith': R_arith,
            'mean_lam': mean_lam, 'mean_H': mean_H,
            'r_t_above_05': above_05, 'mean_r': mean_r, 'r_t_list': r_t_list}


# ════════════════════════════════════════════════
#  2. 多数投票伪标签
# ════════════════════════════════════════════════
def majority_vote(responses):
    """返回伪标签列表 (1=匹配多数, 0=不匹配) 和多数答案"""
    answers = [r.get('extracted_answer', '') for r in responses]
    # 过滤空答案
    valid = [a for a in answers if a is not None and str(a).strip() != '']
    if not valid:
        return [0] * len(responses), None
    counter = Counter(valid)
    majority_ans = counter.most_common(1)[0][0]
    pseudo = [1 if str(a).strip() == str(majority_ans).strip() else 0 for a in answers]
    return pseudo, majority_ans


def grpo_advantage(rewards):
    """GRPO 优势值: A_i = (R_i - mean) / (std + eps)"""
    r = np.array(rewards, dtype=float)
    m, s = r.mean(), r.std()
    return (r - m) / (s + eps)


# ════════════════════════════════════════════════
#  3. 主分析
# ════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data ...")
    with open(FILE) as f:
        data = json.load(f)
    results = data['results']
    N_Q = len(results)
    print(f"  {N_Q} questions, {results[0]['n_samples']} responses/question")

    print("Loading model ...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # ── 遍历所有 question, 计算指标 ──
    all_groups = []   # list of dicts, 每个 question 一个
    all_r_t = {'correct': [], 'wrong': []}   # 收集 r_t 分布

    for qi, res in enumerate(results):
        prompt = res['prompt']
        gt = str(res['ground_truth'])
        responses = res['responses']
        K = len(responses)

        # 多数投票
        pseudo, majority_ans = majority_vote(responses)

        # 真实标签
        true_labels = [1 if r.get('is_correct', False) else 0 for r in responses]

        # R_geo + R_brake
        r_geos, r_brakes, r_t_lists = [], [], []
        for ri, r in enumerate(responses):
            m = compute_r_geo(model, tok, prompt, r.get('response', ''))
            r_geos.append(m['R_geo'])
            r_brakes.append(m['R_brake'])
            # 收集 r_t 分布
            key = 'correct' if true_labels[ri] else 'wrong'
            all_r_t[key].extend(m['r_t_list'])
            r_t_lists.append(m['r_t_list'])

        # 优势值
        adv_geo = grpo_advantage(r_geos)
        adv_brake = grpo_advantage(r_brakes)
        adv_mv  = grpo_advantage(pseudo)

        group = {
            'qid': res['id'], 'gt': gt, 'majority_ans': majority_ans,
            'K': K, 'n_correct': sum(true_labels),
            'pseudo': pseudo, 'true': true_labels,
            'r_geo': r_geos, 'r_brake': r_brakes,
            'adv_geo': adv_geo.tolist(), 'adv_brake': adv_brake.tolist(),
            'adv_mv': adv_mv.tolist(),
            'pseudo_correct': (majority_ans is not None and
                               str(majority_ans).strip() == gt.strip()),
        }
        all_groups.append(group)

        if (qi + 1) % 5 == 0 or qi == N_Q - 1:
            print(f"  [{qi+1}/{N_Q}] qid={res['id']} correct={sum(true_labels)}/{K} "
                  f"pseudo_ok={group['pseudo_correct']} "
                  f"r_geo=[{min(r_geos):.4f},{max(r_geos):.4f}] "
                  f"r_brake=[{min(r_brakes):.5f},{max(r_brakes):.5f}]")

    # ════════════════════════════════════════════
    #  4. 绘图
    # ════════════════════════════════════════════
    print("\nGenerating plots ...")

    # ── 4.1 选取代表性 GRPO 组 (详细柱状图) ──
    cats = {'pseudo_right_mixed': None, 'pseudo_right_all': None, 'pseudo_wrong': None}
    for g in all_groups:
        nc = g['n_correct']
        K = g['K']
        if g['pseudo_correct'] and 0 < nc < K and cats['pseudo_right_mixed'] is None:
            cats['pseudo_right_mixed'] = g
        if g['pseudo_correct'] and nc == K and cats['pseudo_right_all'] is None:
            cats['pseudo_right_all'] = g
        if not g['pseudo_correct'] and cats['pseudo_wrong'] is None:
            cats['pseudo_wrong'] = g
    for key in cats:
        if cats[key] is None:
            for g in all_groups:
                if g['pseudo_correct'] and key.startswith('pseudo_right'):
                    cats[key] = g; break
                if not g['pseudo_correct'] and key == 'pseudo_wrong':
                    cats[key] = g; break
        if cats[key] is None:
            cats[key] = all_groups[0]

    for cat_name, g in cats.items():
        _plot_single_group(g, cat_name, OUT_DIR)

    # ── 4.2 全局: 三方优势值分布 ──
    _plot_global_advantage_distribution(all_groups, OUT_DIR)

    # ── 4.3 伪标签 vs 真实标签一致性 ──
    _plot_pseudo_vs_true(all_groups, OUT_DIR)

    # ── 4.4 奖励分布对比 ──
    _plot_reward_distribution(all_groups, OUT_DIR)

    # ── 4.5 新增: r_t 位置分布 + 刹车效果 ──
    _plot_rt_distribution_and_brake(all_r_t, all_groups, OUT_DIR)

    print(f"\n所有图像已保存至: {OUT_DIR}/")


# ════════════════════════════════════════════════
#  绘图函数
# ════════════════════════════════════════════════
def _plot_single_group(g, title_tag, out_dir):
    """单个 GRPO 组的详细对比柱状图: R_geo / R_brake / MajVote"""
    K = g['K']
    idx = np.arange(K)
    w = 0.25  # 三组柱

    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
    fig.suptitle(f'GRPO Group — Q{g["qid"]} ({title_tag})\n'
                 f'GT={g["gt"]}  Majority={g["majority_ans"]}  '
                 f'Pseudo {"CORRECT" if g["pseudo_correct"] else "WRONG"}  '
                 f'Acc={g["n_correct"]}/{K}',
                 fontsize=14, fontweight='bold', y=1.01)

    # ── 上图: 奖励值 ──
    ax = axes[0]
    colors_t = ['#2ECC71' if t else '#E74C3C' for t in g['true']]
    ax.bar(idx - w, g['r_geo'], w, label='R_geo',
           color=colors_t, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(idx, g['r_brake'], w, label='R_brake = geo(r(1-r))',
           color=['#9B59B6' if t else '#E8DAEF' for t in g['true']],
           edgecolor='black', linewidth=0.8, alpha=0.85)
    colors_mv = ['#3498DB' if p else '#BDC3C7' for p in g['pseudo']]
    ax.bar(idx + w, g['pseudo'], w, label='MajVote (0/1)',
           color=colors_mv, edgecolor='black', linewidth=0.8, alpha=0.85)

    for i in range(K):
        marker = 'o' if g['true'][i] else 'x'
        color = '#27AE60' if g['true'][i] else '#C0392B'
        ymax = max(g['r_geo'][i], g['r_brake'][i], g['pseudo'][i])
        ax.scatter(i, ymax + 0.02, marker=marker, s=80, color=color,
                   zorder=5, linewidths=2)

    ax.set_ylabel('Reward', fontsize=13, fontweight='bold')
    ax.set_title('Reward Values (green/purple=correct, red/light=wrong)', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.02, max(max(g['r_geo']), 1.0) + 0.08)

    # ── 下图: 优势值 ──
    ax2 = axes[1]
    ax2.bar(idx - w, g['adv_geo'], w, label='Adv (R_geo)',
            color=['#2ECC71' if a > 0 else '#E74C3C' for a in g['adv_geo']],
            edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.bar(idx, g['adv_brake'], w, label='Adv (R_brake)',
            color=['#9B59B6' if a > 0 else '#D7BDE2' for a in g['adv_brake']],
            edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.bar(idx + w, g['adv_mv'], w, label='Adv (MajVote)',
            color=['#3498DB' if a > 0 else '#95A5A6' for a in g['adv_mv']],
            edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=1)

    for i in range(K):
        sym = 'T' if g['true'][i] else 'F'
        psym = 'P' if g['pseudo'][i] else '-'
        ax2.text(i, ax2.get_ylim()[0] - 0.08, f'{sym}/{psym}', ha='center',
                 fontsize=8, fontweight='bold', color='#2C3E50')

    ax2.set_xlabel('Response Index', fontsize=13, fontweight='bold')
    ax2.set_ylabel('GRPO Advantage', fontsize=13, fontweight='bold')
    ax2.set_title('GRPO Advantages  (T/F=True label, P/-=Pseudo label)', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f'R{i}' for i in range(K)])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f'grpo_group_{title_tag}_q{g["qid"]}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_global_advantage_distribution(groups, out_dir):
    """所有组的优势值分布: R_geo vs R_brake vs MajVote, 按真实正确性着色"""
    adv_geo_c, adv_geo_w = [], []
    adv_brake_c, adv_brake_w = [], []
    adv_mv_c, adv_mv_w = [], []

    for g in groups:
        for i in range(g['K']):
            if g['true'][i]:
                adv_geo_c.append(g['adv_geo'][i])
                adv_brake_c.append(g['adv_brake'][i])
                adv_mv_c.append(g['adv_mv'][i])
            else:
                adv_geo_w.append(g['adv_geo'][i])
                adv_brake_w.append(g['adv_brake'][i])
                adv_mv_w.append(g['adv_mv'][i])

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('GRPO Advantage Distribution: R_geo vs R_brake vs MajVote\n'
                 '(grouped by TRUE label)',
                 fontsize=16, fontweight='bold', y=1.01)

    bins = np.linspace(-3, 3, 50)

    def _hist(ax, adv_c, adv_w, title):
        ax.hist(adv_c, bins=bins, alpha=0.7, color='#2ECC71',
                label=f'Correct (n={len(adv_c)})',
                edgecolor='black', linewidth=0.5, density=True)
        ax.hist(adv_w, bins=bins, alpha=0.5, color='#E74C3C',
                label=f'Wrong (n={len(adv_w)})',
                edgecolor='black', linewidth=0.5, density=True)
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
        mc, mw = np.mean(adv_c), np.mean(adv_w)
        ax.axvline(mc, color='#27AE60', linewidth=2, linestyle='-',
                   label=f'Correct mean={mc:.3f}')
        ax.axvline(mw, color='#C0392B', linewidth=2, linestyle='-',
                   label=f'Wrong mean={mw:.3f}')
        ax.set_xlabel('GRPO Advantage', fontsize=13, fontweight='bold')
        ax.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _hist(axes[0], adv_geo_c, adv_geo_w, 'R_geo')
    _hist(axes[1], adv_brake_c, adv_brake_w, 'R_brake = geo(r(1-r))')
    _hist(axes[2], adv_mv_c, adv_mv_w, 'Majority Vote (0/1)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, 'advantage_distribution_global.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── 统计表 ──
    print(f"\n  ┌─ Advantage Statistics ──────────────────────────────────────────────┐")
    print(f"  │ {'Metric':<22s} │ {'Correct':>10s} │ {'Wrong':>10s} │ {'Gap':>10s} │")
    print(f"  ├────────────────────────┼────────────┼────────────┼────────────┤")
    for name, ac, aw in [('R_geo', adv_geo_c, adv_geo_w),
                          ('R_brake geo(r(1-r))', adv_brake_c, adv_brake_w),
                          ('MajVote', adv_mv_c, adv_mv_w)]:
        mc, mw = np.mean(ac), np.mean(aw)
        print(f"  │ {name:<22s} │ {mc:>+10.4f} │ {mw:>+10.4f} │ {mc-mw:>10.4f} │")
    print(f"  └────────────────────────┴────────────┴────────────┴────────────┘")

    for name, ac, aw in [('R_geo', adv_geo_c, adv_geo_w),
                          ('R_brake', adv_brake_c, adv_brake_w),
                          ('MajVote', adv_mv_c, adv_mv_w)]:
        pc = np.mean(np.array(ac) > 0)
        pw = np.mean(np.array(aw) > 0)
        print(f"  P(adv>0 | correct): {name:>12s} = {pc:.3f}    "
              f"P(adv>0 | wrong): {pw:.3f}")


def _plot_pseudo_vs_true(groups, out_dir):
    """伪标签 vs 真实标签的混淆矩阵 + 问题级别一致性"""
    # token 级别混淆
    tp = fp = fn = tn = 0
    for g in groups:
        for i in range(g['K']):
            t, p = g['true'][i], g['pseudo'][i]
            if t and p: tp += 1
            elif not t and p: fp += 1
            elif t and not p: fn += 1
            else: tn += 1

    # 问题级别: 伪标签 (多数答案) 的准确率
    n_pseudo_correct = sum(1 for g in groups if g['pseudo_correct'])
    n_total = len(groups)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Pseudo-Label vs True Label Analysis',
                 fontsize=16, fontweight='bold', y=1.01)

    # ── 混淆矩阵 ──
    ax = axes[0]
    cm = np.array([[tn, fp], [fn, tp]])
    labels = np.array([[f'TN\n{tn}', f'FP\n{fp}'],
                       [f'FN\n{fn}', f'TP\n{tp}']])
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Pseudo=0', 'Pseudo=1'],
                yticklabels=['True=Wrong', 'True=Correct'],
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 16, 'fontweight': 'bold'})
    ax.set_title('Response-Level Confusion Matrix', fontsize=14, fontweight='bold')
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    ax.set_xlabel(f'Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}',
                  fontsize=11, fontweight='bold')

    # ── 问题级别 ──
    ax2 = axes[1]
    q_correct_counts = [g['n_correct'] for g in groups]
    q_pseudo_correct = [g['pseudo_correct'] for g in groups]

    colors = ['#2ECC71' if pc else '#E74C3C' for pc in q_pseudo_correct]
    ax2.bar(range(n_total), q_correct_counts, color=colors,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.axhline(y=np.mean(q_correct_counts), color='#3498DB', linestyle='--',
                linewidth=2, label=f'Mean correct={np.mean(q_correct_counts):.1f}')
    ax2.set_xlabel('Question Index', fontsize=13, fontweight='bold')
    ax2.set_ylabel('# Correct Responses (out of 16)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Per-Question Accuracy (green=pseudo correct, red=pseudo wrong)\n'
                  f'Pseudo-label accuracy: {n_pseudo_correct}/{n_total} = {n_pseudo_correct/max(n_total,1):.1%}',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, 'pseudo_vs_true_labels.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_reward_distribution(groups, out_dir):
    """R_geo / R_brake 奖励分布: correct vs wrong"""
    r_geo_c, r_geo_w = [], []
    r_brake_c, r_brake_w = [], []
    for g in groups:
        for i in range(g['K']):
            if g['true'][i]:
                r_geo_c.append(g['r_geo'][i])
                r_brake_c.append(g['r_brake'][i])
            else:
                r_geo_w.append(g['r_geo'][i])
                r_brake_w.append(g['r_brake'][i])

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Reward Distribution: R_geo vs R_brake = geo(r(1-r))',
                 fontsize=16, fontweight='bold', y=1.01)

    # ── R_geo 分布 ──
    ax = axes[0]
    bins_geo = np.linspace(0, 0.45, 50)
    ax.hist(r_geo_c, bins=bins_geo, alpha=0.7, color='#2ECC71',
            label=f'Correct (n={len(r_geo_c)}, μ={np.mean(r_geo_c):.4f})',
            edgecolor='black', linewidth=0.5, density=True)
    ax.hist(r_geo_w, bins=bins_geo, alpha=0.5, color='#E74C3C',
            label=f'Wrong (n={len(r_geo_w)}, μ={np.mean(r_geo_w):.4f})',
            edgecolor='black', linewidth=0.5, density=True)
    ax.axvline(np.mean(r_geo_c), color='#27AE60', linewidth=2.5)
    ax.axvline(np.mean(r_geo_w), color='#C0392B', linewidth=2.5)
    ax.set_xlabel('R_geo', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title('R_geo Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── R_brake 分布 ──
    ax1 = axes[1]
    all_brake = r_brake_c + r_brake_w
    bmin = min(all_brake) * 0.9 if all_brake else 0
    bmax = max(all_brake) * 1.1 if all_brake else 0.3
    bins_br = np.linspace(bmin, bmax, 50)
    ax1.hist(r_brake_c, bins=bins_br, alpha=0.7, color='#2ECC71',
             label=f'Correct (μ={np.mean(r_brake_c):.5f})',
             edgecolor='black', linewidth=0.5, density=True)
    ax1.hist(r_brake_w, bins=bins_br, alpha=0.5, color='#E74C3C',
             label=f'Wrong (μ={np.mean(r_brake_w):.5f})',
             edgecolor='black', linewidth=0.5, density=True)
    ax1.axvline(np.mean(r_brake_c), color='#27AE60', linewidth=2.5)
    ax1.axvline(np.mean(r_brake_w), color='#C0392B', linewidth=2.5)
    ax1.set_xlabel('R_brake = geo(r(1-r))', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax1.set_title('R_brake Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Scatter: correct 优势 vs 难度 ──
    ax2 = axes[2]
    x_acc, y_geo, y_brake, y_mv = [], [], [], []
    for g in groups:
        acc = g['n_correct'] / g['K']
        adv_c_g = [g['adv_geo'][i] for i in range(g['K']) if g['true'][i]]
        adv_c_b = [g['adv_brake'][i] for i in range(g['K']) if g['true'][i]]
        adv_c_m = [g['adv_mv'][i] for i in range(g['K']) if g['true'][i]]
        if adv_c_g:
            x_acc.append(acc)
            y_geo.append(np.mean(adv_c_g))
            y_brake.append(np.mean(adv_c_b))
            y_mv.append(np.mean(adv_c_m))

    ax2.scatter(x_acc, y_geo, alpha=0.7, s=70, color='#3498DB',
                edgecolors='black', linewidth=0.5, label='R_geo', zorder=3)
    ax2.scatter(x_acc, y_brake, alpha=0.7, s=70, color='#9B59B6',
                edgecolors='black', linewidth=0.5, marker='^', label='R_brake', zorder=3)
    ax2.scatter(x_acc, y_mv, alpha=0.5, s=70, color='#E67E22',
                edgecolors='black', linewidth=0.5, marker='s', label='MajVote', zorder=3)
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')
    ax2.set_xlabel('Question Accuracy (correct/K)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Advantage of Correct', fontsize=13, fontweight='bold')
    ax2.set_title('Correct Adv. vs Difficulty', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    if len(x_acc) > 3:
        for yy, c in [(y_geo, '#3498DB'), (y_brake, '#9B59B6'), (y_mv, '#E67E22')]:
            z = np.polyfit(x_acc, yy, 1)
            xs = np.linspace(0, 1, 50)
            ax2.plot(xs, np.polyval(z, xs), '--', color=c, linewidth=2, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, 'reward_distribution.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_rt_distribution_and_brake(all_r_t, groups, out_dir):
    """
    核心分析: r_t 逐位置分布 + r(1-r) 刹车理论曲线 + 性能影响评估
    回答: r_t 最大到 0.5 是否损害性能?
    """
    fig, axes = plt.subplots(2, 3, figsize=(26, 14))
    fig.suptitle('Anti-Collapse Brake Analysis: R_brake = geo_mean(r_t · (1 - r_t))\n'
                 'Does capping at r=0.5 hurt performance?',
                 fontsize=16, fontweight='bold', y=1.01)

    # ═══ (0,0): r_t 分布 correct vs wrong ═══
    ax = axes[0, 0]
    rt_c = np.array(all_r_t['correct'])
    rt_w = np.array(all_r_t['wrong'])
    bins = np.linspace(0, 0.7, 80)
    ax.hist(rt_c, bins=bins, alpha=0.7, color='#2ECC71', density=True,
            label=f'Correct (n={len(rt_c)}, μ={rt_c.mean():.4f})', edgecolor='none')
    ax.hist(rt_w, bins=bins, alpha=0.5, color='#E74C3C', density=True,
            label=f'Wrong (n={len(rt_w)}, μ={rt_w.mean():.4f})', edgecolor='none')
    ax.axvline(0.5, color='black', linewidth=2, linestyle='--', label='r=0.5 (brake peak)')
    pct_c_above = (rt_c > 0.5).mean() * 100
    pct_w_above = (rt_w > 0.5).mean() * 100
    ax.axvline(rt_c.mean(), color='#27AE60', linewidth=2)
    ax.axvline(rt_w.mean(), color='#C0392B', linewidth=2)
    ax.set_title(f'Per-Token r_t = H_eff/H Distribution\n'
                 f'r_t > 0.5: correct={pct_c_above:.1f}%, wrong={pct_w_above:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('r_t', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ═══ (0,1): 理论曲线 r vs r(1-r) ═══
    ax1 = axes[0, 1]
    r_th = np.linspace(0, 1, 500)
    f_brake = r_th * (1 - r_th)
    f_linear = r_th
    ax1.plot(r_th, f_linear, 'b--', linewidth=2, label='R_geo: f(r)=r', alpha=0.7)
    ax1.plot(r_th, f_brake, 'r-', linewidth=3, label='R_brake: f(r)=r(1-r)', alpha=0.9)
    # 标注当前工作区间
    rt_all = np.concatenate([rt_c, rt_w])
    q10, q50, q90 = np.percentile(rt_all, [10, 50, 90])
    ax1.axvspan(q10, q90, alpha=0.15, color='#3498DB',
                label=f'Current 10-90% range [{q10:.3f}, {q90:.3f}]')
    ax1.axvline(q50, color='#3498DB', linewidth=2, linestyle=':',
                label=f'Current median = {q50:.3f}')
    ax1.axvline(0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    # 标注斜率
    ax1.annotate('slope > 0\n(learning)', xy=(0.2, 0.16), fontsize=11,
                 fontweight='bold', color='#27AE60',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1'))
    ax1.annotate('slope < 0\n(braking)', xy=(0.7, 0.21), fontsize=11,
                 fontweight='bold', color='#C0392B',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC'))
    ax1.set_title('Reward Function Shape\nr(1-r) peaks at 0.5, provides natural brake',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('r_t = H_eff / H', fontsize=12, fontweight='bold')
    ax1.set_ylabel('f(r_t)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ═══ (0,2): 梯度对比 ═══
    ax2 = axes[0, 2]
    grad_linear = np.ones_like(r_th)
    grad_brake = 1 - 2 * r_th
    ax2.plot(r_th, grad_linear, 'b--', linewidth=2, label="R_geo: f'=1 (unbounded)", alpha=0.7)
    ax2.plot(r_th, grad_brake, 'r-', linewidth=3, label="R_brake: f'=1-2r (self-braking)", alpha=0.9)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axvline(0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5,
                label='r=0.5 (zero gradient)')
    ax2.axvspan(q10, q90, alpha=0.15, color='#3498DB', label='Current operating range')
    # 标注当前中位梯度
    med_grad = 1 - 2 * q50
    ax2.scatter([q50], [med_grad], s=120, color='#E67E22', zorder=5, edgecolors='black',
                linewidth=2, label=f'Current median: grad={med_grad:.3f}')
    ax2.set_title(f'Gradient Analysis\nAt median r={q50:.3f}, brake gradient = {med_grad:.3f} (>0, still learning)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('r_t', fontsize=12, fontweight='bold')
    ax2.set_ylabel("f'(r_t)", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ═══ (1,0): R_geo vs R_brake 散点 (每个 response 一个点) ═══
    ax3 = axes[1, 0]
    geo_vals, brake_vals, is_correct = [], [], []
    for g in groups:
        for i in range(g['K']):
            geo_vals.append(g['r_geo'][i])
            brake_vals.append(g['r_brake'][i])
            is_correct.append(g['true'][i])
    geo_vals = np.array(geo_vals)
    brake_vals = np.array(brake_vals)
    is_correct = np.array(is_correct)

    ax3.scatter(geo_vals[is_correct == 1], brake_vals[is_correct == 1],
                alpha=0.5, s=30, color='#2ECC71', label='Correct', edgecolors='none')
    ax3.scatter(geo_vals[is_correct == 0], brake_vals[is_correct == 0],
                alpha=0.3, s=30, color='#E74C3C', label='Wrong', edgecolors='none')
    # 理论曲线 R_brake ≈ R_geo(1-R_geo) 的近似
    rr = np.linspace(0, 0.45, 100)
    ax3.plot(rr, rr * (1 - rr), 'k--', linewidth=2, alpha=0.5,
             label='R·(1-R) approx')
    ax3.set_xlabel('R_geo', fontsize=12, fontweight='bold')
    ax3.set_ylabel('R_brake', fontsize=12, fontweight='bold')
    ax3.set_title('R_geo vs R_brake per response\n(strong correlation = brake preserves ordering)',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Kendall tau
    from scipy import stats
    tau, pval = stats.kendalltau(geo_vals, brake_vals)
    ax3.text(0.05, 0.95, f'Kendall τ = {tau:.4f}\np = {pval:.2e}',
             transform=ax3.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F9E79F'))

    # ═══ (1,1): R_brake 是否损害区分度? (Cohen's d 对比) ═══
    ax4 = axes[1, 1]
    geo_c = geo_vals[is_correct == 1]
    geo_w = geo_vals[is_correct == 0]
    brake_c = brake_vals[is_correct == 1]
    brake_w = brake_vals[is_correct == 0]

    def cohens_d(a, b):
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * np.var(a) + (nb - 1) * np.var(b)) / (na + nb - 2))
        return (np.mean(a) - np.mean(b)) / (sp + 1e-12)

    d_geo = cohens_d(geo_c, geo_w)
    d_brake = cohens_d(brake_c, brake_w)

    # 也计算优势值的 d
    adv_geo_c_all = [g['adv_geo'][i] for g in groups for i in range(g['K']) if g['true'][i]]
    adv_geo_w_all = [g['adv_geo'][i] for g in groups for i in range(g['K']) if not g['true'][i]]
    adv_brake_c_all = [g['adv_brake'][i] for g in groups for i in range(g['K']) if g['true'][i]]
    adv_brake_w_all = [g['adv_brake'][i] for g in groups for i in range(g['K']) if not g['true'][i]]
    adv_mv_c_all = [g['adv_mv'][i] for g in groups for i in range(g['K']) if g['true'][i]]
    adv_mv_w_all = [g['adv_mv'][i] for g in groups for i in range(g['K']) if not g['true'][i]]

    d_adv_geo = cohens_d(adv_geo_c_all, adv_geo_w_all)
    d_adv_brake = cohens_d(adv_brake_c_all, adv_brake_w_all)
    d_adv_mv = cohens_d(adv_mv_c_all, adv_mv_w_all)

    names = ['R_geo\n(reward)', 'R_brake\n(reward)', 'R_geo\n(advantage)',
             'R_brake\n(advantage)', 'MajVote\n(advantage)']
    ds = [d_geo, d_brake, d_adv_geo, d_adv_brake, d_adv_mv]
    colors = ['#3498DB', '#9B59B6', '#3498DB', '#9B59B6', '#E67E22']
    bars = ax4.bar(names, ds, color=colors, edgecolor='black', linewidth=1, alpha=0.85)
    for bar, d in zip(bars, ds):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{d:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax4.set_ylabel("Cohen's d (correct vs wrong)", fontsize=12, fontweight='bold')
    ax4.set_title("Discrimination Power Comparison\n"
                  "(higher = better at distinguishing correct/wrong)",
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # ═══ (1,2): 综合结论面板 ═══
    ax5 = axes[1, 2]
    ax5.axis('off')
    # 计算关键数据
    p_geo_pos = np.mean(np.array(adv_geo_c_all) > 0)
    p_brake_pos = np.mean(np.array(adv_brake_c_all) > 0)
    p_mv_pos = np.mean(np.array(adv_mv_c_all) > 0)

    text = (
        "═══ Key Findings ═══\n\n"
        f"1. Current r_t distribution:\n"
        f"   median = {q50:.4f}  (far below 0.5)\n"
        f"   10th pct = {q10:.4f}\n"
        f"   90th pct = {q90:.4f}\n"
        f"   % above 0.5: correct={pct_c_above:.1f}%, wrong={pct_w_above:.1f}%\n\n"
        f"2. Does r_t cap at 0.5 hurt?\n"
        f"   Brake gradient at median: {med_grad:.3f} > 0\n"
        f"   → Still fully learning in current range\n"
        f"   → Brake only activates for r_t > 0.5\n\n"
        f"3. Discrimination (Cohen's d):\n"
        f"   R_geo reward d   = {d_geo:.4f}\n"
        f"   R_brake reward d = {d_brake:.4f}\n"
        f"   R_brake adv d    = {d_adv_brake:.4f}\n"
        f"   MajVote adv d    = {d_adv_mv:.4f}\n\n"
        f"4. P(adv>0 | correct):\n"
        f"   R_geo   = {p_geo_pos:.3f}\n"
        f"   R_brake = {p_brake_pos:.3f}\n"
        f"   MajVote = {p_mv_pos:.3f}\n\n"
        f"5. Ordering preservation:\n"
        f"   Kendall τ(R_geo, R_brake) = {tau:.4f}\n"
        f"   → Brake preserves relative ordering"
    )
    ax5.text(0.05, 0.95, text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9',
                       edgecolor='#7F8C8D', linewidth=2))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, 'brake_analysis.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
