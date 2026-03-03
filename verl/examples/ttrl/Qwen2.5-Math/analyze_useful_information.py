#!/usr/bin/env python3
"""
深度分析：什么是"有用信息"？

核心问题：用户想把熵效率拆成：
  - 熵增时：最大化增加有用信息（不乱增熵）
  - 熵减时：最小化丢弃有用信息（不为锐化丢重要部分）

但"有用"如何定义？本脚本从数据中回答这个问题：
1. 收集所有可计算特征（熵、KL、结构、分布）
2. 计算每个特征与正确性的相关性
3. 拆解"熵增"和"熵减"阶段各自的信号
4. 找出哪些特征真正区分正确/错误推理
"""

import json, os
import numpy as np
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "eval_results_aime_full_entropy")

# 加载评估 JSON（包含 step-level 详细数据）
EVAL_JSON = os.path.join(EVAL_DIR, "aime_eval_full_entropy_20260206_131034.json")
# 加载 token-level B0 结果
TOKEN_B0_JSON = os.path.join(EVAL_DIR, "token_level_b0_results.json")


def load_all_data():
    """加载并融合所有数据源"""
    with open(EVAL_JSON) as f:
        eval_data = json.load(f)
    
    with open(TOKEN_B0_JSON) as f:
        token_data = json.load(f)
    
    trained_token = {(s['qid'], s['extracted_answer'], s['is_correct']): s 
                     for s in token_data['Trained']['samples']}
    
    samples = []
    for result in eval_data['results']:
        qid = result['id']
        for ridx, resp in enumerate(result['responses']):
            ea = resp.get('entropy_analysis', {})
            trans = ea.get('step_transitions', [])
            steps = ea.get('steps', [])
            ib = ea.get('ib_summary', {})
            
            is_correct = resp.get('is_correct', False)
            extracted = resp.get('extracted_answer')
            response = resp.get('response', '')
            
            # === Step-Level 特征 ===
            step_entropies = [s.get('mean_entropy', 0) for s in steps]
            n_steps = len(steps)
            
            # 熵趋势
            if n_steps >= 2:
                entropy_trend = np.polyfit(range(n_steps), step_entropies, 1)[0]
                entropy_start = np.mean(step_entropies[:3]) if n_steps >= 3 else step_entropies[0]
                entropy_end = np.mean(step_entropies[-3:]) if n_steps >= 3 else step_entropies[-1]
                entropy_mid = np.mean(step_entropies[n_steps//3:2*n_steps//3]) if n_steps >= 3 else np.mean(step_entropies)
            else:
                entropy_trend = 0
                entropy_start = entropy_end = entropy_mid = np.mean(step_entropies) if step_entropies else 0
            
            # === Transition 特征 ===
            kl_fwds = [t['kl_forward'] for t in trans]
            kl_revs = [t['kl_reverse'] for t in trans]
            js_divs = [t['js_divergence'] for t in trans]
            ent_deltas = [t['entropy_delta'] for t in trans]
            
            # 拆分熵增/熵减阶段
            up_indices = [i for i, d in enumerate(ent_deltas) if d > 0]
            down_indices = [i for i, d in enumerate(ent_deltas) if d <= 0]
            
            # 熵增阶段特征
            if up_indices:
                up_kl_fwd = np.mean([kl_fwds[i] for i in up_indices])
                up_kl_rev = np.mean([kl_revs[i] for i in up_indices])
                up_js = np.mean([js_divs[i] for i in up_indices])
                up_delta = np.mean([ent_deltas[i] for i in up_indices])
                up_ratio = up_kl_rev / (up_kl_fwd + 1e-10)  # 逆向/正向
                up_asym = (up_kl_rev - up_kl_fwd) / (up_kl_rev + up_kl_fwd + 1e-10)
                up_frac = len(up_indices) / len(ent_deltas)
            else:
                up_kl_fwd = up_kl_rev = up_js = up_delta = up_ratio = up_asym = 0
                up_frac = 0
            
            # 熵减阶段特征
            if down_indices:
                down_kl_fwd = np.mean([kl_fwds[i] for i in down_indices])
                down_kl_rev = np.mean([kl_revs[i] for i in down_indices])
                down_js = np.mean([js_divs[i] for i in down_indices])
                down_delta = np.mean([ent_deltas[i] for i in down_indices])
                down_ratio = down_kl_fwd / (down_kl_rev + 1e-10)
                down_asym = (down_kl_fwd - down_kl_rev) / (down_kl_fwd + down_kl_rev + 1e-10)
                down_frac = len(down_indices) / len(ent_deltas)
            else:
                down_kl_fwd = down_kl_rev = down_js = down_delta = down_ratio = down_asym = 0
                down_frac = 0
            
            # === 用户的"熵效率"概念 ===
            # 尝试多种定义：
            # V1: 原始定义 - q = (c-d)/(c+d)
            qs_v1 = []
            for t in trans:
                kf = max(t['kl_forward'], 0)
                kr = max(t['kl_reverse'], 0)
                ed = t['entropy_delta']
                if ed > 0:
                    c, d = kr, kf
                else:
                    c, d = kf, kr
                denom = c + d
                qs_v1.append((c - d) / denom if denom > 1e-10 else 0)
            
            # V2: 用 JS 加权的效率
            qs_v2 = []
            for i, t in enumerate(trans):
                q = qs_v1[i]
                qs_v2.append(q * js_divs[i])
            
            # V3: 用 entropy_delta 大小加权
            qs_v3 = []
            for i, t in enumerate(trans):
                q = qs_v1[i]
                qs_v3.append(q * abs(ent_deltas[i]))
            
            # === 分布形状特征 ===
            # 从 step_transitions 提取 cosine similarity
            cos_sims = [t.get('cosine_similarity', 0) for t in trans]
            
            # === IB summary ===
            ib_total_info = ib.get('total_information_gain', 0)
            ib_total_comp = ib.get('total_compression', 0)
            
            # === Token-level B0 特征（从 token 数据） ===
            token_key = (qid, extracted, is_correct)
            token_info = trained_token.get(token_key, {})
            
            sample = {
                'qid': qid,
                'response_idx': ridx,
                'is_correct': is_correct,
                'extracted_answer': extracted,
                
                # ── 基础统计 ──
                'response_length': len(response),
                'n_steps': n_steps,
                'mean_entropy': np.mean(step_entropies) if step_entropies else 0,
                'std_entropy': np.std(step_entropies) if step_entropies else 0,
                'min_entropy': min(step_entropies) if step_entropies else 0,
                'max_entropy': max(step_entropies) if step_entropies else 0,
                'entropy_range': max(step_entropies) - min(step_entropies) if step_entropies else 0,
                
                # ── 熵趋势 ──
                'entropy_trend': entropy_trend,
                'entropy_start': entropy_start,
                'entropy_end': entropy_end,
                'entropy_mid': entropy_mid,
                'entropy_end_minus_start': entropy_end - entropy_start,
                'final_step_entropy': step_entropies[-1] if step_entropies else 0,
                
                # ── 全局 KL/JS ──
                'mean_kl_fwd': np.mean(kl_fwds) if kl_fwds else 0,
                'mean_kl_rev': np.mean(kl_revs) if kl_revs else 0,
                'mean_js': np.mean(js_divs) if js_divs else 0,
                'mean_cos_sim': np.mean(cos_sims) if cos_sims else 0,
                'kl_asymmetry': np.mean([(kf - kr)/(kf + kr + 1e-10) for kf, kr in zip(kl_fwds, kl_revs)]) if kl_fwds else 0,
                
                # ── 熵增阶段 ──
                'up_frac': up_frac,
                'up_kl_fwd': up_kl_fwd,
                'up_kl_rev': up_kl_rev,
                'up_js': up_js,
                'up_delta': up_delta,
                'up_ratio': up_ratio,
                'up_asym': up_asym,
                
                # ── 熵减阶段 ──
                'down_frac': down_frac,
                'down_kl_fwd': down_kl_fwd,
                'down_kl_rev': down_kl_rev,
                'down_js': down_js,
                'down_delta': down_delta,
                'down_ratio': down_ratio,
                'down_asym': down_asym,
                
                # ── 用户的"效率"指标 ──
                'eff_v1_mean': np.mean(qs_v1) if qs_v1 else 0,
                'eff_v1_std': np.std(qs_v1) if qs_v1 else 0,
                'eff_v1_frac_pos': np.mean([1 if q > 0 else 0 for q in qs_v1]) if qs_v1 else 0,
                'eff_v2_js_weighted': np.mean(qs_v2) if qs_v2 else 0,
                'eff_v3_delta_weighted': np.mean(qs_v3) if qs_v3 else 0,
                
                # ── 只看熵增部分的效率 ──
                'eff_up_only': np.mean([qs_v1[i] for i in up_indices]) if up_indices else 0,
                # ── 只看熵减部分的效率 ──
                'eff_down_only': np.mean([qs_v1[i] for i in down_indices]) if down_indices else 0,
                
                # ── IB ──
                'ib_total_info': ib_total_info,
                'ib_total_comp': ib_total_comp,
                'ib_ratio': ib_total_info / (ib_total_comp + 1e-10) if ib_total_comp > 0 else 0,
                
                # ── Token-level ──
                'token_b0_mean': token_info.get('token_b0_mean', 0),
                'token_b0_std': token_info.get('token_b0_std', 0),
                'n_tokens': token_info.get('n_tokens', 0),
            }
            
            samples.append(sample)
    
    # 添加 consensus
    by_qid = defaultdict(list)
    for s in samples:
        by_qid[s['qid']].append(s)
    
    for qid, group in by_qid.items():
        answers = [s['extracted_answer'] for s in group]
        counter = Counter(a for a in answers if a is not None and a != '')
        gs = len(group)
        for s in group:
            a = s['extracted_answer']
            s['consensus'] = counter.get(a, 1) / gs if a else 1.0 / gs
    
    return samples, by_qid


def analyze_features(samples, by_qid):
    """分析所有特征与正确性的关系"""
    
    # 获取所有数值特征
    skip_keys = {'qid', 'response_idx', 'is_correct', 'extracted_answer'}
    feature_keys = [k for k in samples[0].keys() if k not in skip_keys]
    
    print(f"\n{'='*80}")
    print(f"  深度分析：什么区分正确和错误的推理？ ({len(samples)} samples)")
    print(f"{'='*80}")
    
    # ── 1. 全局相关性分析 ──
    print(f"\n  ╔{'═'*78}╗")
    print(f"  ║  Part 1: 全局特征与正确性的 Pearson 相关                                     ║")
    print(f"  ╚{'═'*78}╝")
    
    correlations = {}
    for k in feature_keys:
        vals = np.array([s[k] for s in samples])
        corr_arr = np.array([1.0 if s['is_correct'] else 0.0 for s in samples])
        
        if np.std(vals) < 1e-10:
            correlations[k] = 0
            continue
        
        r = np.corrcoef(vals, corr_arr)[0, 1]
        correlations[k] = r if not np.isnan(r) else 0
    
    # 按相关性绝对值排序
    sorted_features = sorted(correlations.items(), key=lambda x: -abs(x[1]))
    
    print(f"\n  {'Rank':<5} {'特征':<30} {'Pearson r':>10} {'方向':>6} {'类别':<15}")
    print(f"  {'-'*68}")
    
    categories = {
        'consensus': '共识',
        'response_length': '结构', 'n_steps': '结构', 'n_tokens': '结构',
        'mean_entropy': '熵-全局', 'std_entropy': '熵-全局', 'min_entropy': '熵-全局',
        'max_entropy': '熵-全局', 'entropy_range': '熵-全局',
        'entropy_trend': '熵-趋势', 'entropy_start': '熵-趋势', 'entropy_end': '熵-趋势',
        'entropy_mid': '熵-趋势', 'entropy_end_minus_start': '熵-趋势', 'final_step_entropy': '熵-趋势',
        'mean_kl_fwd': 'KL全局', 'mean_kl_rev': 'KL全局', 'mean_js': 'KL全局',
        'mean_cos_sim': 'KL全局', 'kl_asymmetry': 'KL全局',
        'up_frac': '熵增阶段', 'up_kl_fwd': '熵增阶段', 'up_kl_rev': '熵增阶段',
        'up_js': '熵增阶段', 'up_delta': '熵增阶段', 'up_ratio': '熵增阶段', 'up_asym': '熵增阶段',
        'down_frac': '熵减阶段', 'down_kl_fwd': '熵减阶段', 'down_kl_rev': '熵减阶段',
        'down_js': '熵减阶段', 'down_delta': '熵减阶段', 'down_ratio': '熵减阶段', 'down_asym': '熵减阶段',
        'eff_v1_mean': '效率指标', 'eff_v1_std': '效率指标', 'eff_v1_frac_pos': '效率指标',
        'eff_v2_js_weighted': '效率指标', 'eff_v3_delta_weighted': '效率指标',
        'eff_up_only': '效率指标', 'eff_down_only': '效率指标',
        'ib_total_info': 'IB', 'ib_total_comp': 'IB', 'ib_ratio': 'IB',
        'token_b0_mean': 'Token级', 'token_b0_std': 'Token级',
    }
    
    for rank, (feat, r) in enumerate(sorted_features, 1):
        direction = '↑好' if r > 0 else '↓好'
        cat = categories.get(feat, '其他')
        marker = ' ★' if abs(r) > 0.15 else ''
        print(f"  {rank:<5} {feat:<30} {r:>+10.4f} {direction:>6} {cat:<15}{marker}")
    
    # ── 2. GRPO 组级评估 ──
    print(f"\n  ╔{'═'*78}╗")
    print(f"  ║  Part 2: GRPO 组级 Advantage Gap                                            ║")
    print(f"  ╚{'═'*78}╝")
    
    grpo_results = {}
    for k in feature_keys:
        gaps = []
        for qid, group in by_qid.items():
            if len(group) < 4:
                continue
            vals = np.array([s[k] for s in group])
            corr = np.array([s['is_correct'] for s in group], dtype=bool)
            std_v = np.std(vals)
            nc = int(np.sum(corr))
            if std_v > 1e-8 and 0 < nc < len(corr):
                advs = (vals - np.mean(vals)) / std_v
                gaps.append(float(np.mean(advs[corr]) - np.mean(advs[~corr])))
        grpo_results[k] = np.mean(gaps) if gaps else 0
    
    sorted_grpo = sorted(grpo_results.items(), key=lambda x: -x[1])
    
    print(f"\n  {'Rank':<5} {'特征':<30} {'adv_gap':>10} {'类别':<15}")
    print(f"  {'-'*62}")
    for rank, (feat, gap) in enumerate(sorted_grpo, 1):
        cat = categories.get(feat, '其他')
        marker = ' ★' if gap > 0.3 else ''
        print(f"  {rank:<5} {feat:<30} {gap:>+10.4f} {cat:<15}{marker}")
    
    # ── 3. 按类别汇总 ──
    print(f"\n  ╔{'═'*78}╗")
    print(f"  ║  Part 3: 按类别汇总——哪类特征最有区分力？                                    ║")
    print(f"  ╚{'═'*78}╝")
    
    cat_features = defaultdict(list)
    for feat, r in correlations.items():
        cat = categories.get(feat, '其他')
        cat_features[cat].append((feat, r, grpo_results.get(feat, 0)))
    
    print(f"\n  {'类别':<15} {'最佳特征':<30} {'|r|_max':>8} {'gap_max':>9} {'avg |r|':>8}")
    print(f"  {'-'*72}")
    
    cat_summary = {}
    for cat, feats in sorted(cat_features.items(), key=lambda x: -max(abs(f[1]) for f in x[1])):
        best_r = max(feats, key=lambda x: abs(x[1]))
        best_gap = max(feats, key=lambda x: x[2])
        avg_r = np.mean([abs(f[1]) for f in feats])
        cat_summary[cat] = {
            'best_r': best_r[1],
            'best_r_feat': best_r[0],
            'best_gap': best_gap[2],
            'best_gap_feat': best_gap[0],
            'avg_r': avg_r,
        }
        print(f"  {cat:<15} {best_r[0]:<30} {abs(best_r[1]):>8.4f} {best_gap[2]:>+9.4f} {avg_r:>8.4f}")
    
    # ── 4. 核心问题：熵增/熵减阶段的信号在哪里？ ──
    print(f"\n  ╔{'═'*78}╗")
    print(f"  ║  Part 4: 熵增 vs 熵减阶段——信号拆解                                        ║")
    print(f"  ╚{'═'*78}╝")
    
    up_features = [f for f in feature_keys if f.startswith('up_') or f == 'eff_up_only']
    down_features = [f for f in feature_keys if f.startswith('down_') or f == 'eff_down_only']
    
    print(f"\n  熵增阶段特征（探索）：")
    print(f"  {'特征':<25} {'Correct':>10} {'Incorrect':>10} {'Δ':>10} {'Pearson r':>10} {'GRPO gap':>10}")
    print(f"  {'-'*78}")
    for f in up_features:
        vc = [s[f] for s in samples if s['is_correct']]
        vi = [s[f] for s in samples if not s['is_correct']]
        mc, mi = np.mean(vc), np.mean(vi)
        print(f"  {f:<25} {mc:>10.4f} {mi:>10.4f} {mc-mi:>+10.4f} {correlations[f]:>+10.4f} {grpo_results[f]:>+10.4f}")
    
    print(f"\n  熵减阶段特征（压缩）：")
    print(f"  {'特征':<25} {'Correct':>10} {'Incorrect':>10} {'Δ':>10} {'Pearson r':>10} {'GRPO gap':>10}")
    print(f"  {'-'*78}")
    for f in down_features:
        vc = [s[f] for s in samples if s['is_correct']]
        vi = [s[f] for s in samples if not s['is_correct']]
        mc, mi = np.mean(vc), np.mean(vi)
        print(f"  {f:<25} {mc:>10.4f} {mi:>10.4f} {mc-mi:>+10.4f} {correlations[f]:>+10.4f} {grpo_results[f]:>+10.4f}")
    
    # ── 5. "有用信息"是什么？ ──
    print(f"\n  ╔{'═'*78}╗")
    print(f"  ║  Part 5: 结论——有用信息的数据定义                                         ║")
    print(f"  ╚{'═'*78}╝")
    
    # 找到 |r| > 0.1 的特征
    significant = [(f, r) for f, r in sorted_features if abs(r) > 0.08]
    
    print(f"\n  与正确性显著相关的特征 (|r| > 0.08):")
    print(f"  {'-'*60}")
    for f, r in significant:
        cat = categories.get(f, '其他')
        direction = "高→更可能正确" if r > 0 else "低→更可能正确"
        print(f"    {f:<28} r={r:+.4f}  [{cat}] {direction}")
    
    print(f"\n  与正确性不相关的特征类别:")
    for cat, info in cat_summary.items():
        if abs(info['best_r']) < 0.08:
            print(f"    {cat}: 最佳 |r| = {abs(info['best_r']):.4f} (无信号)")
    
    return correlations, grpo_results, cat_summary


def plot_analysis(samples, correlations, grpo_results, output_path):
    """综合可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle('What is "Useful Information"? — Feature Analysis for Reward Design',
                 fontsize=14, fontweight='bold')
    
    categories = {
        '共识': '#f39c12',
        '结构': '#9b59b6',
        '熵-全局': '#3498db',
        '熵-趋势': '#2980b9',
        'KL全局': '#1abc9c',
        '熵增阶段': '#e74c3c',
        '熵减阶段': '#e67e22',
        '效率指标': '#27ae60',
        'IB': '#8e44ad',
        'Token级': '#34495e',
    }
    
    cat_map = {
        'consensus': '共识',
        'response_length': '结构', 'n_steps': '结构', 'n_tokens': '结构',
        'mean_entropy': '熵-全局', 'std_entropy': '熵-全局', 'min_entropy': '熵-全局',
        'max_entropy': '熵-全局', 'entropy_range': '熵-全局',
        'entropy_trend': '熵-趋势', 'entropy_start': '熵-趋势', 'entropy_end': '熵-趋势',
        'entropy_mid': '熵-趋势', 'entropy_end_minus_start': '熵-趋势', 'final_step_entropy': '熵-趋势',
        'mean_kl_fwd': 'KL全局', 'mean_kl_rev': 'KL全局', 'mean_js': 'KL全局',
        'mean_cos_sim': 'KL全局', 'kl_asymmetry': 'KL全局',
        'up_frac': '熵增阶段', 'up_kl_fwd': '熵增阶段', 'up_kl_rev': '熵增阶段',
        'up_js': '熵增阶段', 'up_delta': '熵增阶段', 'up_ratio': '熵增阶段', 'up_asym': '熵增阶段',
        'down_frac': '熵减阶段', 'down_kl_fwd': '熵减阶段', 'down_kl_rev': '熵减阶段',
        'down_js': '熵减阶段', 'down_delta': '熵减阶段', 'down_ratio': '熵减阶段', 'down_asym': '熵减阶段',
        'eff_v1_mean': '效率指标', 'eff_v1_std': '效率指标', 'eff_v1_frac_pos': '效率指标',
        'eff_v2_js_weighted': '效率指标', 'eff_v3_delta_weighted': '效率指标',
        'eff_up_only': '效率指标', 'eff_down_only': '效率指标',
        'ib_total_info': 'IB', 'ib_total_comp': 'IB', 'ib_ratio': 'IB',
        'token_b0_mean': 'Token级', 'token_b0_std': 'Token级',
    }
    
    # Plot 1: Top 20 features by |Pearson r|
    ax = axes[0, 0]
    sorted_r = sorted(correlations.items(), key=lambda x: -abs(x[1]))[:20]
    names = [f[0] for f in sorted_r]
    vals = [f[1] for f in sorted_r]
    colors_r = [categories.get(cat_map.get(n, '其他'), '#95a5a6') for n in names]
    ax.barh(range(len(names)), vals, color=colors_r, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Pearson r with correctness')
    ax.set_title('Top 20 Features: Pearson r')
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Plot 2: Top 20 features by GRPO gap
    ax = axes[0, 1]
    sorted_g = sorted(grpo_results.items(), key=lambda x: -x[1])[:20]
    names_g = [f[0] for f in sorted_g]
    vals_g = [f[1] for f in sorted_g]
    colors_g = [categories.get(cat_map.get(n, '其他'), '#95a5a6') for n in names_g]
    ax.barh(range(len(names_g)), vals_g, color=colors_g, alpha=0.8)
    ax.set_yticks(range(len(names_g)))
    ax.set_yticklabels(names_g, fontsize=7)
    ax.set_xlabel('GRPO Advantage Gap')
    ax.set_title('Top 20 Features: GRPO Gap')
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Plot 3: Pearson r vs GRPO gap scatter
    ax = axes[0, 2]
    all_feats = list(set(correlations.keys()) & set(grpo_results.keys()))
    for f in all_feats:
        cat = cat_map.get(f, '其他')
        color = categories.get(cat, '#95a5a6')
        ax.scatter(correlations[f], grpo_results[f], color=color, s=30, alpha=0.6)
    
    # Annotate top features
    top_by_gap = sorted(all_feats, key=lambda f: -grpo_results[f])[:5]
    top_by_r = sorted(all_feats, key=lambda f: -abs(correlations[f]))[:5]
    for f in set(top_by_gap + top_by_r):
        ax.annotate(f.replace('entropy_', 'ent_').replace('token_b0_', 'tb0_'), 
                    (correlations[f], grpo_results[f]), fontsize=6, alpha=0.8)
    
    ax.set_xlabel('Pearson r')
    ax.set_ylabel('GRPO Gap')
    ax.set_title('Pearson r vs GRPO Gap')
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.axvline(0, color='gray', ls='--', alpha=0.3)
    ax.grid(alpha=0.3)
    
    # Legend for categories
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=cat) for cat, c in categories.items()]
    ax.legend(handles=legend_patches, fontsize=6, loc='lower right', ncol=2)
    
    # Plot 4: Category-level summary
    ax = axes[1, 0]
    cat_stats = defaultdict(lambda: {'rs': [], 'gaps': []})
    for f in all_feats:
        cat = cat_map.get(f, '其他')
        cat_stats[cat]['rs'].append(abs(correlations[f]))
        cat_stats[cat]['gaps'].append(grpo_results[f])
    
    cats = sorted(cat_stats.keys(), key=lambda c: -max(cat_stats[c]['gaps']))
    max_gaps = [max(cat_stats[c]['gaps']) for c in cats]
    max_rs = [max(cat_stats[c]['rs']) for c in cats]
    colors_cat = [categories.get(c, '#95a5a6') for c in cats]
    
    x = np.arange(len(cats))
    width = 0.35
    ax.bar(x - width/2, max_gaps, width, color=colors_cat, alpha=0.8, label='Max GRPO Gap')
    ax.bar(x + width/2, max_rs, width, color=colors_cat, alpha=0.4, label='Max |Pearson r|')
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Category Comparison: Gap vs |r|')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 5: Entropy profile correct vs incorrect
    ax = axes[1, 1]
    correct_profiles = []
    incorrect_profiles = []
    for s in samples:
        profile = [s.get('entropy_start', 0), s.get('entropy_mid', 0), s.get('entropy_end', 0)]
        if s['is_correct']:
            correct_profiles.append(profile)
        else:
            incorrect_profiles.append(profile)
    
    cp = np.array(correct_profiles)
    ip = np.array(incorrect_profiles)
    x_pos = [0, 1, 2]
    ax.errorbar(x_pos, cp.mean(axis=0), yerr=cp.std(axis=0)/np.sqrt(len(cp)),
                fmt='o-', color='#2ecc71', lw=2, capsize=5, label=f'Correct (n={len(cp)})')
    ax.errorbar(x_pos, ip.mean(axis=0), yerr=ip.std(axis=0)/np.sqrt(len(ip)),
                fmt='s-', color='#e74c3c', lw=2, capsize=5, label=f'Incorrect (n={len(ip)})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Start (step 1-3)', 'Mid (step 5-10)', 'End (step 13-15)'])
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Entropy Profile: Correct vs Incorrect')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Plot 6: 熵增 vs 熵减效率对比
    ax = axes[1, 2]
    eff_up_c = [s['eff_up_only'] for s in samples if s['is_correct']]
    eff_up_i = [s['eff_up_only'] for s in samples if not s['is_correct']]
    eff_down_c = [s['eff_down_only'] for s in samples if s['is_correct']]
    eff_down_i = [s['eff_down_only'] for s in samples if not s['is_correct']]
    
    labels_box = ['Entropy↑\nCorrect', 'Entropy↑\nIncorrect', 'Entropy↓\nCorrect', 'Entropy↓\nIncorrect']
    data_box = [eff_up_c, eff_up_i, eff_down_c, eff_down_i]
    colors_box = ['#2ecc71', '#e74c3c', '#27ae60', '#c0392b']
    
    bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    for i, (d, label) in enumerate(zip(data_box, labels_box)):
        ax.text(i + 1, np.median(d), f'{np.median(d):.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Efficiency (q)')
    ax.set_title('Entropy↑ vs Entropy↓ Efficiency\nby Correctness')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {output_path}")
    plt.close()


def main():
    samples, by_qid = load_all_data()
    correlations, grpo_results, cat_summary = analyze_features(samples, by_qid)
    
    output_path = os.path.join(EVAL_DIR, "useful_information_analysis.png")
    plot_analysis(samples, correlations, grpo_results, output_path)


if __name__ == '__main__':
    main()
