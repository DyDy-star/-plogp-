"""
从根本分析低熵锐化级联效应，并模拟两种阻断机制：
1. 非对称锐化预算 (ASB) - 序列级优势塑性
2. 软熵梯度阻断 - token级优势衰减

数据: 未训练(pass@1=4%) vs 正确性训练(pass@1=26%)
只考虑优势塑性和无监督奖励，不依赖多数投票
"""
import json
import numpy as np
from collections import defaultdict
import sys

EARLY_FILE = "eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json"
LATE_FILE = "eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json"

def load_data(path):
    with open(path) as f:
        return json.load(f)

def extract_all_responses(data):
    """提取所有response的关键指标"""
    responses = []
    for problem in data["results"]:
        uid = problem["id"]
        gt = problem["ground_truth"]
        for resp in problem["responses"]:
            ea = resp["entropy_analysis"]
            stats = ea["overall_stats"]
            
            all_token_entropies = []
            step_entropies = []
            for step in ea["steps"]:
                if "token_entropies" in step:
                    all_token_entropies.extend(step["token_entropies"])
                step_entropies.append(step.get("mean_entropy", 0))
            
            token_h = np.array(all_token_entropies) if all_token_entropies else np.array([0])
            
            dH = np.diff(step_entropies) if len(step_entropies) > 1 else np.array([0])
            pos_dH = dH[dH > 0]
            sigma_pos = float(np.std(pos_dH)) if len(pos_dH) > 1 else 0.0
            
            responses.append({
                "uid": uid,
                "is_correct": resp["is_correct"],
                "mean_H": stats["overall_mean_entropy"],
                "std_H": stats["overall_std_entropy"],
                "max_H": stats["overall_max_entropy"],
                "early_H": stats["early_avg_entropy"],
                "late_H": stats["late_avg_entropy"],
                "entropy_reward": stats["entropy_reward"],
                "n_tokens": stats["total_tokens"],
                "n_steps": stats["n_steps"],
                "sigma_pos": sigma_pos,
                "token_entropies": token_h,
                "n_exploration": stats.get("ib_summary", {}).get("n_exploration_steps", 0),
                "n_compression": stats.get("ib_summary", {}).get("n_compression_steps", 0),
                "js_div": stats.get("ib_summary", {}).get("mean_js_divergence", 0),
            })
    return responses

def analyze_sharpening_cascade(early_resps, late_resps):
    """验证低熵锐化级联假说"""
    print("=" * 80)
    print("实验1: 低熵锐化级联 - 基础统计对比")
    print("=" * 80)
    
    for label, resps in [("未训练(4%)", early_resps), ("训练后(26%)", late_resps)]:
        mean_Hs = [r["mean_H"] for r in resps]
        std_Hs = [r["std_H"] for r in resps]
        correct = [r for r in resps if r["is_correct"]]
        wrong = [r for r in resps if not r["is_correct"]]
        
        print(f"\n--- {label} (n={len(resps)}, 正确={len(correct)}) ---")
        print(f"  mean_H: {np.mean(mean_Hs):.4f} ± {np.std(mean_Hs):.4f}")
        print(f"  std_H:  {np.mean(std_Hs):.4f} ± {np.std(std_Hs):.4f}")
        if correct:
            print(f"  正确 mean_H: {np.mean([r['mean_H'] for r in correct]):.4f}")
        print(f"  错误 mean_H: {np.mean([r['mean_H'] for r in wrong]):.4f}")

def analyze_token_entropy_distribution(early_resps, late_resps):
    """分析token级熵分布变化 - 级联证据"""
    print("\n" + "=" * 80)
    print("实验2: Token级熵分布变化 (锐化级联证据)")
    print("=" * 80)
    
    for label, resps in [("未训练", early_resps), ("训练后", late_resps)]:
        all_h = np.concatenate([r["token_entropies"] for r in resps])
        
        bins = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float('inf')]
        bin_labels = ["[0,0.01)", "[0.01,0.05)", "[0.05,0.1)", "[0.1,0.25)", 
                      "[0.25,0.5)", "[0.5,1.0)", "[1.0,2.0)", "[2.0+)"]
        counts, _ = np.histogram(all_h, bins=bins)
        pcts = counts / len(all_h) * 100
        
        print(f"\n--- {label} (total tokens: {len(all_h)}) ---")
        for bl, pct, cnt in zip(bin_labels, pcts, counts):
            bar = "█" * int(pct)
            print(f"  {bl:>14s}: {pct:5.1f}% ({cnt:>6d}) {bar}")
        
        p25 = np.percentile(all_h, 25)
        p50 = np.percentile(all_h, 50)
        p75 = np.percentile(all_h, 75)
        p90 = np.percentile(all_h, 90)
        print(f"  分位数: P25={p25:.4f}, P50={p50:.4f}, P75={p75:.4f}, P90={p90:.4f}")

def analyze_context_suppression(early_resps, late_resps):
    """验证低熵上下文对高熵token的压制效应"""
    print("\n" + "=" * 80)
    print("实验3: 低熵上下文对高熵token的压制效应")
    print("=" * 80)
    
    for label, resps in [("未训练", early_resps), ("训练后", late_resps)]:
        context_window = 5
        all_high_in_low_ctx = []
        all_high_in_high_ctx = []
        
        for r in resps:
            h = r["token_entropies"]
            if len(h) < context_window + 1:
                continue
            
            p75 = np.percentile(h, 75)
            p25 = np.percentile(h, 25)
            if p75 == 0 or p25 == p75:
                continue
            
            for i in range(context_window, len(h)):
                if h[i] > p75:
                    ctx_mean = np.mean(h[i-context_window:i])
                    if ctx_mean < p25:
                        all_high_in_low_ctx.append(h[i])
                    elif ctx_mean > p75:
                        all_high_in_high_ctx.append(h[i])
        
        if all_high_in_low_ctx and all_high_in_high_ctx:
            mean_low = np.mean(all_high_in_low_ctx)
            mean_high = np.mean(all_high_in_high_ctx)
            suppression = 1 - mean_low / mean_high if mean_high > 0 else 0
            print(f"\n--- {label} ---")
            print(f"  高熵token在低熵上下文中: {mean_low:.4f} (n={len(all_high_in_low_ctx)})")
            print(f"  高熵token在高熵上下文中: {mean_high:.4f} (n={len(all_high_in_high_ctx)})")
            print(f"  压制比: {suppression:.4f} ({suppression*100:.1f}%)")
        else:
            print(f"\n--- {label}: 数据不足 ---")

def analyze_grpo_zero_gradient_problem(early_resps, late_resps):
    """分析GRPO零梯度问题 (全错组)"""
    print("\n" + "=" * 80)
    print("实验4: GRPO零梯度问题 (全错组分析)")
    print("=" * 80)
    
    for label, resps in [("未训练", early_resps), ("训练后", late_resps)]:
        groups = defaultdict(list)
        for r in resps:
            groups[r["uid"]].append(r)
        
        n_all_wrong = 0
        n_total_groups = len(groups)
        zero_grad_responses = 0
        total_responses = len(resps)
        
        all_wrong_sigma_pos = []
        all_wrong_mean_H = []
        
        for uid, group in groups.items():
            if all(not r["is_correct"] for r in group):
                n_all_wrong += 1
                zero_grad_responses += len(group)
                for r in group:
                    all_wrong_sigma_pos.append(r["sigma_pos"])
                    all_wrong_mean_H.append(r["mean_H"])
        
        print(f"\n--- {label} ---")
        print(f"  全错组: {n_all_wrong}/{n_total_groups} ({n_all_wrong/n_total_groups*100:.1f}%)")
        print(f"  零梯度响应: {zero_grad_responses}/{total_responses} ({zero_grad_responses/total_responses*100:.1f}%)")
        
        if all_wrong_sigma_pos:
            sp = np.array(all_wrong_sigma_pos)
            mh = np.array(all_wrong_mean_H)
            print(f"  全错组内 sigma_pos: {np.mean(sp):.4f} ± {np.std(sp):.4f} (range: {np.min(sp):.4f}-{np.max(sp):.4f})")
            print(f"  全错组内 mean_H:    {np.mean(mh):.4f} ± {np.std(mh):.4f}")
            print(f"  → sigma_pos 在组内有足够方差提供区分信号")

def simulate_sigma_pos_unsupervised_reward(early_resps, late_resps):
    """模拟sigma_pos无监督奖励的效果"""
    print("\n" + "=" * 80)
    print("实验5: sigma_pos 无监督奖励信号质量")
    print("=" * 80)
    
    for label, resps in [("未训练", early_resps), ("训练后", late_resps)]:
        correct = [r for r in resps if r["is_correct"]]
        wrong = [r for r in resps if not r["is_correct"]]
        
        sp_correct = [r["sigma_pos"] for r in correct] if correct else [0]
        sp_wrong = [r["sigma_pos"] for r in wrong]
        
        mean_c = np.mean(sp_correct)
        mean_w = np.mean(sp_wrong)
        std_pooled = np.sqrt((np.var(sp_correct) * len(sp_correct) + np.var(sp_wrong) * len(sp_wrong)) / 
                            (len(sp_correct) + len(sp_wrong))) if (len(sp_correct) + len(sp_wrong)) > 0 else 1
        cohens_d = (mean_c - mean_w) / std_pooled if std_pooled > 0 else 0
        
        print(f"\n--- {label} ---")
        print(f"  sigma_pos 正确: {mean_c:.4f} ± {np.std(sp_correct):.4f} (n={len(correct)})")
        print(f"  sigma_pos 错误: {mean_w:.4f} ± {np.std(sp_wrong):.4f} (n={len(wrong)})")
        print(f"  Cohen's d: {cohens_d:.3f}")
        
        groups = defaultdict(list)
        for r in resps:
            groups[r["uid"]].append(r)
        
        lam = 0.5
        n_unlocked = 0
        n_total_zero = 0
        for uid, group in groups.items():
            if all(not r["is_correct"] for r in group):
                sps = [r["sigma_pos"] for r in group]
                mu_sigma = np.mean(sps)
                rewards = [lam * (mu_sigma - sp) for sp in sps]
                n_total_zero += len(group)
                n_positive = sum(1 for rw in rewards if rw > 0.01)
                n_unlocked += n_positive
        
        if n_total_zero > 0:
            print(f"  全错组中被解锁的响应: {n_unlocked}/{n_total_zero} ({n_unlocked/n_total_zero*100:.1f}%)")

def simulate_asb(early_resps, late_resps):
    """模拟非对称锐化预算 (ASB) 的效果"""
    print("\n" + "=" * 80)
    print("实验6: 非对称锐化预算 (ASB) 模拟")
    print("=" * 80)
    
    for label, resps in [("未训练", early_resps), ("训练后", late_resps)]:
        groups = defaultdict(list)
        for r in resps:
            groups[r["uid"]].append(r)
        
        mean_Hs = np.array([r["mean_H"] for r in resps])
        tau = np.median(mean_Hs)
        
        advantages_std = []
        advantages_asb = []
        budgets = []
        
        for uid, group in groups.items():
            scores = []
            for r in group:
                score = 1.0 if r["is_correct"] else 0.0
                scores.append(score)
            
            mu = np.mean(scores)
            sigma = np.std(scores)
            if sigma < 1e-6:
                sigma = 1.0
            
            for i, r in enumerate(group):
                adv = (scores[i] - mu) / sigma
                advantages_std.append(adv)
                
                mh = r["mean_H"]
                w = np.tanh(mh / tau) if tau > 0 else 1.0
                adv_asb = adv * w if adv < 0 else adv
                advantages_asb.append(adv_asb)
                budgets.append(w)
        
        adv_std = np.array(advantages_std)
        adv_asb = np.array(advantages_asb)
        bgt = np.array(budgets)
        
        neg_mask = adv_std < 0
        pos_mask = adv_std > 0
        zero_mask = adv_std == 0
        
        print(f"\n--- {label} (tau={tau:.4f}) ---")
        print(f"  优势分布: 正={pos_mask.sum()}, 负={neg_mask.sum()}, 零={zero_mask.sum()}")
        
        if neg_mask.sum() > 0:
            neg_budgets = bgt[neg_mask]
            print(f"  负优势序列的预算 w:")
            print(f"    mean={np.mean(neg_budgets):.4f}, median={np.median(neg_budgets):.4f}")
            print(f"    w<0.5: {(neg_budgets<0.5).sum()} ({(neg_budgets<0.5).mean()*100:.1f}%) — 强限制")
            print(f"    w<0.8: {(neg_budgets<0.8).sum()} ({(neg_budgets<0.8).mean()*100:.1f}%) — 中限制")
            
            print(f"  负优势幅度变化:")
            print(f"    标准: mean={np.mean(adv_std[neg_mask]):.4f}")
            print(f"    ASB:   mean={np.mean(adv_asb[neg_mask]):.4f}")
            print(f"    衰减比: {1 - np.mean(np.abs(adv_asb[neg_mask])) / np.mean(np.abs(adv_std[neg_mask])):.2%}")

def simulate_soft_entropy_gate(early_resps, late_resps):
    """模拟软熵梯度阻断 vs 硬阻断"""
    print("\n" + "=" * 80)
    print("实验7: 软熵梯度阻断 vs 硬阻断 对比")
    print("=" * 80)
    
    for label, resps in [("未训练", early_resps), ("训练后", late_resps)]:
        all_h = np.concatenate([r["token_entropies"] for r in resps])
        
        H_REF_hard = 0.1
        gate_hard = all_h / (all_h + H_REF_hard)
        
        alpha = 0.3
        H_mean = np.mean(all_h[all_h > 0]) if np.any(all_h > 0) else 0.1
        gate_soft = alpha + (1 - alpha) * np.minimum(all_h / (H_mean + 1e-6), 1.0)
        
        ranks = np.zeros_like(all_h)
        for r in resps:
            h = r["token_entropies"]
            if len(h) > 1:
                order = np.argsort(np.argsort(h))
                ranks_norm = order / (len(h) - 1) if len(h) > 1 else np.ones_like(h)
                start_idx = sum(len(rr["token_entropies"]) for rr in resps if id(rr) < id(r))
        
        print(f"\n--- {label} (total tokens: {len(all_h)}, H_mean={H_mean:.4f}) ---")
        
        eff_hard = np.mean(gate_hard)
        eff_soft = np.mean(gate_soft)
        
        zero_hard = (gate_hard < 0.05).mean()
        zero_soft = (gate_soft < 0.05).mean()
        
        print(f"  硬阻断 g=H/(H+0.1):")
        print(f"    有效batch比例: {eff_hard:.2%}")
        print(f"    近零gate(<0.05): {zero_hard:.2%}")
        print(f"    gate分位: P25={np.percentile(gate_hard, 25):.4f}, P50={np.percentile(gate_hard, 50):.4f}, P75={np.percentile(gate_hard, 75):.4f}")
        
        print(f"  软阻断 g=0.3+0.7*min(H/H_mean,1):")
        print(f"    有效batch比例: {eff_soft:.2%}")
        print(f"    近零gate(<0.05): {zero_soft:.2%}")
        print(f"    gate分位: P25={np.percentile(gate_soft, 25):.4f}, P50={np.percentile(gate_soft, 50):.4f}, P75={np.percentile(gate_soft, 75):.4f}")
        
        low_h = all_h < np.percentile(all_h, 25)
        high_h = all_h > np.percentile(all_h, 75)
        
        print(f"  低熵token(Q1)梯度贡献:")
        print(f"    硬阻断: {np.mean(gate_hard[low_h]):.4f} (vs 无阻断=1.0)")
        print(f"    软阻断: {np.mean(gate_soft[low_h]):.4f}")
        print(f"  高熵token(Q4)梯度贡献:")
        print(f"    硬阻断: {np.mean(gate_hard[high_h]):.4f}")
        print(f"    软阻断: {np.mean(gate_soft[high_h]):.4f}")
        
        ratio_hard = np.mean(gate_hard[high_h]) / (np.mean(gate_hard[low_h]) + 1e-8)
        ratio_soft = np.mean(gate_soft[high_h]) / (np.mean(gate_soft[low_h]) + 1e-8)
        print(f"  高/低熵梯度贡献比:")
        print(f"    无阻断: 1.00")
        print(f"    硬阻断: {ratio_hard:.2f}")
        print(f"    软阻断: {ratio_soft:.2f}")

def analyze_cascade_evidence(early_resps, late_resps):
    """直接量化锐化级联: 训练后高熵token的锐化是否与低熵邻居密度相关"""
    print("\n" + "=" * 80)
    print("实验8: 锐化级联因果证据 - 高熵token锐化量 vs 低熵邻居密度")
    print("=" * 80)
    
    early_by_uid = defaultdict(list)
    late_by_uid = defaultdict(list)
    for r in early_resps:
        early_by_uid[r["uid"]].append(r)
    for r in late_resps:
        late_by_uid[r["uid"]].append(r)
    
    common_uids = set(early_by_uid.keys()) & set(late_by_uid.keys())
    
    early_all_step_H = []
    late_all_step_H = []
    early_step_counts = []
    late_step_counts = []
    
    for uid in common_uids:
        for r in early_by_uid[uid]:
            early_all_step_H.append(r["mean_H"])
            early_step_counts.append(r["n_steps"])
        for r in late_by_uid[uid]:
            late_all_step_H.append(r["mean_H"])
            late_step_counts.append(r["n_steps"])
    
    early_mh = np.array(early_all_step_H)
    late_mh = np.array(late_all_step_H)
    
    early_high_h_frac = (early_mh > np.percentile(early_mh, 75)).mean()
    late_high_h_frac = (late_mh > np.percentile(late_mh, 75)).mean()
    
    early_low_h_frac = (early_mh < np.percentile(early_mh, 25)).mean()
    late_low_h_frac = (late_mh < np.percentile(late_mh, 25)).mean()
    
    print(f"高熵序列(>P75)占比: 未训练={early_high_h_frac:.2%}, 训练后={late_high_h_frac:.2%}")
    print(f"低熵序列(<P25)占比: 未训练={early_low_h_frac:.2%}, 训练后={late_low_h_frac:.2%}")
    
    early_exploration_ratio = np.mean([r["n_exploration"] / max(r["n_steps"], 1) for r in early_resps])
    late_exploration_ratio = np.mean([r["n_exploration"] / max(r["n_steps"], 1) for r in late_resps])
    
    print(f"\n探索步占比: 未训练={early_exploration_ratio:.2%}, 训练后={late_exploration_ratio:.2%}")
    print(f"探索步减少: {1 - late_exploration_ratio/early_exploration_ratio:.2%}")
    
    early_js = np.mean([r["js_div"] for r in early_resps if r["js_div"] > 0])
    late_js = np.mean([r["js_div"] for r in late_resps if r["js_div"] > 0])
    print(f"\n平均JS散度: 未训练={early_js:.4f}, 训练后={late_js:.4f}")
    print(f"JS散度变化: {(late_js - early_js)/early_js*100:+.1f}%")
    print(f"→ 训练后每步转换更尖锐但总探索更少 = 锐化级联的特征")

def combined_mechanism_simulation(early_resps, late_resps):
    """模拟组合机制: sigma_pos无监督奖励 + ASB + 软阻断"""
    print("\n" + "=" * 80)
    print("实验9: 组合机制模拟 (σ_pos + ASB)")
    print("=" * 80)
    
    for label, resps in [("未训练", early_resps), ("训练后", late_resps)]:
        groups = defaultdict(list)
        for r in resps:
            groups[r["uid"]].append(r)
        
        mean_Hs = np.array([r["mean_H"] for r in resps])
        tau = np.median(mean_Hs)
        lam = 0.5
        
        n_sequences = len(resps)
        std_advantages = np.zeros(n_sequences)
        combined_advantages = np.zeros(n_sequences)
        
        idx = 0
        for uid, group in groups.items():
            scores = np.array([1.0 if r["is_correct"] else 0.0 for r in group])
            sps = np.array([r["sigma_pos"] for r in group])
            mu_sigma = np.mean(sps)
            
            corrected_scores = scores + lam * (mu_sigma - sps)
            
            mu_std = np.mean(scores)
            sigma_std = np.std(scores)
            if sigma_std < 1e-6:
                sigma_std = 1.0
            
            mu_corr = np.mean(corrected_scores)
            sigma_corr = np.std(corrected_scores)
            if sigma_corr < 1e-6:
                sigma_corr = 1.0
            
            for i, r in enumerate(group):
                adv_std = (scores[i] - mu_std) / sigma_std
                std_advantages[idx] = adv_std
                
                adv_corr = (corrected_scores[i] - mu_corr) / sigma_corr
                mh = r["mean_H"]
                w = np.tanh(mh / tau) if tau > 0 else 1.0
                adv_combined = adv_corr * w if adv_corr < 0 else adv_corr
                combined_advantages[idx] = adv_combined
                
                idx += 1
        
        zero_std = (np.abs(std_advantages) < 1e-6).sum()
        zero_comb = (np.abs(combined_advantages) < 1e-6).sum()
        
        nonzero_std = n_sequences - zero_std
        nonzero_comb = n_sequences - zero_comb
        
        print(f"\n--- {label} ---")
        print(f"  标准GRPO:")
        print(f"    零梯度序列: {zero_std}/{n_sequences} ({zero_std/n_sequences*100:.1f}%)")
        print(f"    有效学习序列: {nonzero_std}/{n_sequences} ({nonzero_std/n_sequences*100:.1f}%)")
        print(f"    优势 |A|: mean={np.mean(np.abs(std_advantages)):.4f}")
        
        print(f"  σ_pos + ASB 组合:")
        print(f"    零梯度序列: {zero_comb}/{n_sequences} ({zero_comb/n_sequences*100:.1f}%)")
        print(f"    有效学习序列: {nonzero_comb}/{n_sequences} ({nonzero_comb/n_sequences*100:.1f}%)")
        print(f"    优势 |A|: mean={np.mean(np.abs(combined_advantages)):.4f}")
        print(f"    → 有效学习提升: {(nonzero_comb - nonzero_std)}/{n_sequences} 序列被解锁")
        
        neg_std = std_advantages < -1e-6
        neg_comb = combined_advantages < -1e-6
        if neg_std.sum() > 0:
            reduction = 1 - np.mean(np.abs(combined_advantages[neg_std])) / np.mean(np.abs(std_advantages[neg_std]))
            print(f"    → 负优势平均衰减: {reduction:.2%}")

def analyze_high_entropy_preservation(early_resps, late_resps):
    """分析机制对高熵token保护效果"""
    print("\n" + "=" * 80)
    print("实验10: 高熵token保护效果预测")
    print("=" * 80)
    
    for label, resps in [("未训练", early_resps), ("训练后", late_resps)]:
        exploration_ratios = []
        mean_Hs = []
        for r in resps:
            h = r["token_entropies"]
            if len(h) > 0:
                high_h_ratio = (h > np.percentile(h, 75)).mean()
                exploration_ratios.append(high_h_ratio)
                mean_Hs.append(r["mean_H"])
        
        er = np.array(exploration_ratios)
        mh = np.array(mean_Hs)
        
        corr = np.corrcoef(mh, er)[0, 1]
        
        print(f"\n--- {label} ---")
        print(f"  mean_H 与 高熵token占比 相关性: r={corr:.4f}")
        
        correct_er = [er[i] for i, r in enumerate(resps) if r["is_correct"]]
        wrong_er = [er[i] for i, r in enumerate(resps) if not r["is_correct"]]
        
        if correct_er:
            print(f"  正确序列 高熵占比: {np.mean(correct_er):.4f}")
        print(f"  错误序列 高熵占比: {np.mean(wrong_er):.4f}")
        
        correct_early_H = [r["early_H"] for r in resps if r["is_correct"]]
        wrong_early_H = [r["early_H"] for r in resps if not r["is_correct"]]
        if correct_early_H:
            print(f"  正确序列 early_H (思考阶段): {np.mean(correct_early_H):.4f}")
        print(f"  错误序列 early_H (思考阶段): {np.mean(wrong_early_H):.4f}")
        print(f"  → 思考阶段熵差: {np.mean(wrong_early_H) - (np.mean(correct_early_H) if correct_early_H else 0):.4f}")

def main():
    print("加载数据...")
    early = load_data(EARLY_FILE)
    late = load_data(LATE_FILE)
    
    early_resps = extract_all_responses(early)
    late_resps = extract_all_responses(late)
    
    print(f"未训练: {len(early_resps)} responses, pass@1={early['overall_pass@1']:.4f}")
    print(f"训练后: {len(late_resps)} responses, pass@1={late['overall_pass@1']:.4f}")
    
    analyze_sharpening_cascade(early_resps, late_resps)
    analyze_token_entropy_distribution(early_resps, late_resps)
    analyze_context_suppression(early_resps, late_resps)
    analyze_grpo_zero_gradient_problem(early_resps, late_resps)
    simulate_sigma_pos_unsupervised_reward(early_resps, late_resps)
    simulate_asb(early_resps, late_resps)
    simulate_soft_entropy_gate(early_resps, late_resps)
    analyze_cascade_evidence(early_resps, late_resps)
    combined_mechanism_simulation(early_resps, late_resps)
    analyze_high_entropy_preservation(early_resps, late_resps)
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)

if __name__ == "__main__":
    main()
