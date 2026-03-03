#!/usr/bin/env python3
"""
对抗性分解分析:
  1. 探索质量(R_inc) 和压缩质量(R_dec) 在组内是正相关还是负相关?
     - 正相关 → 统一奖励 S3 足够
     - 负相关 → 多目标分解有价值 (两个目标存在张力)
  
  2. 模拟对抗性选择 vs 统一选择:
     - 统一32: GRPO on 32 samples with S3
     - 双批16+16: 按 R_inc top16 + R_dec top16
     - 联合top32: 从64中选 S3 top32
     
  3. 多目标归一化 GRPO:
     - R = norm(R_inc) + norm(R_dec) + norm(R_cons)
"""

import json
import numpy as np
from collections import defaultdict
from scipy import stats as sp_stats


def load_samples(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    samples = []
    for result in data['results']:
        pid = result.get('id', result.get('problem_id', ''))
        for resp in result['responses']:
            ea = resp['entropy_analysis']
            steps = ea.get('steps', [])
            
            all_ent = []
            for s in steps:
                if 'token_entropies' in s and s['token_entropies']:
                    all_ent.extend(s['token_entropies'])
            
            if len(all_ent) < 10:
                continue
            
            ent = np.array(all_ent, dtype=np.float64)
            L = len(ent)
            eps = 1e-10
            
            v = ent[1:] - ent[:-1]
            abs_v = np.abs(v)
            
            inc_mask = v > 0
            dec_mask = v < 0
            v_inc = v[inc_mask]
            v_dec = v[dec_mask]
            ent_at_v = ent[1:]
            ent_dec = ent_at_v[dec_mask]
            ent_inc = ent_at_v[inc_mask]
            
            # ============ 分项指标 ============
            
            # 统一奖励: S3 路径效率
            total_path = np.sum(abs_v) + eps
            S3 = (ent[0] - ent[-1]) / total_path
            
            # F4 对数守恒
            I_inc = np.sum(v_inc) / L if len(v_inc) > 0 else eps
            I_dec = np.sum(np.abs(v_dec)) / L if len(v_dec) > 0 else eps
            F4 = np.log(max(I_dec, eps) / max(I_inc, eps))
            
            # ---- ① 探索质量 R_inc ----
            # 探索质量 = 熵增时不乱增噪声
            # 方案A: -mean(v | v>0) → 小幅探索更好
            R_inc_A = -np.mean(v_inc) if len(v_inc) > 0 else 0
            # 方案B: -mean(H | v>0) → 探索时熵不要太高
            R_inc_B = -np.mean(ent_inc) if len(ent_inc) > 0 else 0
            # 方案C: 探索集中度 Gini(v_inc) → 少数大探索好于多小噪声
            if len(v_inc) > 2:
                sorted_vi = np.sort(v_inc)
                n = len(sorted_vi)
                idx = np.arange(1, n+1)
                R_inc_C = (2*np.sum(idx*sorted_vi)/(n*np.sum(sorted_vi)+eps)) - (n+1)/n
            else:
                R_inc_C = 0
            # 方案D: -I_inc/T → 归一化探索量小 (与S3一致)
            R_inc_D = -I_inc
            
            # ---- ② 压缩质量 R_dec ----
            # 压缩质量 = 熵减时不丢有用信息
            # 方案A: -mean(H | v<0) → 压缩到低熵
            R_dec_A = -np.mean(ent_dec) if len(ent_dec) > 0 else 0
            # 方案B: mean(|v|/H | v<0) → 每单位熵的压缩效率
            if len(v_dec) > 0 and len(ent_dec) > 0:
                R_dec_B = np.mean(np.abs(v_dec) / (ent_dec + eps))
            else:
                R_dec_B = 0
            # 方案C: I_dec/T → 归一化压缩量大 (与S3一致)
            R_dec_C = I_dec
            # 方案D: -mean(|v| | v<0) → 小幅压缩=精细=不丢信息  
            R_dec_D = -np.mean(np.abs(v_dec)) if len(v_dec) > 0 else 0
            
            # ---- ③ 守恒约束 R_cons ----
            R_cons_A = (ent[0] - ent[-1]) / L  # net/T
            R_cons_B = F4   # log(I_dec/I_inc)
            R_cons_C = 1 - I_inc / max(I_dec, eps)  # ratio
            
            samples.append({
                'problem_id': pid,
                'is_correct': resp.get('is_correct', False),
                'S3': S3,
                'F4': F4,
                'R_inc_A': R_inc_A, 'R_inc_B': R_inc_B,
                'R_inc_C': R_inc_C, 'R_inc_D': R_inc_D,
                'R_dec_A': R_dec_A, 'R_dec_B': R_dec_B,
                'R_dec_C': R_dec_C, 'R_dec_D': R_dec_D,
                'R_cons_A': R_cons_A, 'R_cons_B': R_cons_B,
                'R_cons_C': R_cons_C,
            })
    
    return samples


def grpo_gap(samples, metric_fn):
    """计算 GRPO Gap"""
    groups = defaultdict(list)
    for s in samples:
        groups[s['problem_id']].append(s)
    
    c_adv, i_adv = [], []
    for pid, group in groups.items():
        if len(group) < 4:
            continue
        vals = [metric_fn(s) for s in group]
        m, sd = np.mean(vals), np.std(vals)
        if sd < 1e-10:
            continue
        for s, v in zip(group, vals):
            a = (v - m) / sd
            (c_adv if s['is_correct'] else i_adv).append(a)
    
    if not c_adv or not i_adv:
        return 0.0
    return np.mean(c_adv) - np.mean(i_adv)


def main():
    files = {
        'Checkpoint': 'eval_results_aime_full_entropy/aime_eval_full_entropy_20260206_131034.json',
        'Base':       'eval_results_aime_full_entropy/aime_eval_full_entropy_20260206_131750.json',
    }
    
    for model_name, fpath in files.items():
        print(f"\n{'='*80}")
        print(f"  {model_name}")
        print(f"{'='*80}")
        
        samples = load_samples(fpath)
        n_c = sum(1 for s in samples if s['is_correct'])
        print(f"  样本: {len(samples)}, 正确: {n_c}")
        
        # ==============================================================
        # PART 1: 组内相关性分析
        # R_inc 和 R_dec 在同一 prompt 的 32 个样本中是正相关还是负相关?
        # ==============================================================
        print(f"\n  ─── Part 1: 组内 R_inc 与 R_dec 的相关性 ───")
        
        groups = defaultdict(list)
        for s in samples:
            groups[s['problem_id']].append(s)
        
        inc_dec_pairs = [
            ('R_inc_A(-v_inc)', 'R_dec_A(-H_dec)', 'R_inc_A', 'R_dec_A'),
            ('R_inc_B(-H_inc)', 'R_dec_A(-H_dec)', 'R_inc_B', 'R_dec_A'),
            ('R_inc_D(-I_inc)', 'R_dec_C(I_dec)',   'R_inc_D', 'R_dec_C'),
            ('R_inc_A(-v_inc)', 'R_dec_B(|v|/H)',   'R_inc_A', 'R_dec_B'),
        ]
        
        print(f"\n  {'R_inc 指标':<20} {'R_dec 指标':<20} {'组内Pearson':>12} {'方向':>6}")
        print(f"  {'-'*60}")
        
        for inc_name, dec_name, inc_key, dec_key in inc_dec_pairs:
            corrs = []
            for pid, group in groups.items():
                if len(group) < 8:
                    continue
                vi = [s[inc_key] for s in group]
                vd = [s[dec_key] for s in group]
                if np.std(vi) > 1e-10 and np.std(vd) > 1e-10:
                    r, _ = sp_stats.pearsonr(vi, vd)
                    corrs.append(r)
            
            if corrs:
                mean_r = np.mean(corrs)
                direction = "正相关" if mean_r > 0.1 else ("负相关" if mean_r < -0.1 else "弱/无")
                print(f"  {inc_name:<20} {dec_name:<20} {mean_r:>+.4f}      {direction}")
        
        # 关键: S3 与各分项的相关性
        print(f"\n  S3(路径效率) 与各分项的组内相关性:")
        for key in ['R_inc_A', 'R_inc_B', 'R_inc_D', 'R_dec_A', 'R_dec_B', 'R_dec_C']:
            corrs = []
            for pid, group in groups.items():
                if len(group) < 8:
                    continue
                vs = [s['S3'] for s in group]
                vk = [s[key] for s in group]
                if np.std(vs) > 1e-10 and np.std(vk) > 1e-10:
                    r, _ = sp_stats.pearsonr(vs, vk)
                    corrs.append(r)
            if corrs:
                print(f"    S3 ↔ {key}: r = {np.mean(corrs):+.4f}")
        
        # ==============================================================
        # PART 2: 模拟对抗性选择 vs 统一选择
        # 在组内模拟: 假设每组有32个样本
        # ==============================================================
        print(f"\n  ─── Part 2: 对抗性选择 vs 统一选择 (模拟) ───")
        
        # 定义几种选择+奖励策略
        strategies = {}
        
        # 策略0: 标准 GRPO 32样本 + S3
        strategies['std_32_S3'] = ('标准GRPO 32+S3', 
            lambda g: g, lambda s: s['S3'])
        
        # 策略1: 标准 GRPO 32样本 + F4
        strategies['std_32_F4'] = ('标准GRPO 32+F4',
            lambda g: g, lambda s: s['F4'])
        
        # 策略2: 对抗性 — 按R_inc选top16 + 按R_dec选top16, 用S3
        def adversarial_select(group):
            # Sort by R_inc_D (small exploration), take top 16
            by_inc = sorted(group, key=lambda s: s['R_inc_D'], reverse=True)
            inc_set = set(id(s) for s in by_inc[:16])
            # Sort by R_dec_C (large compression), take top 16
            by_dec = sorted(group, key=lambda s: s['R_dec_C'], reverse=True)
            dec_set = set(id(s) for s in by_dec[:16])
            # Union (may overlap)
            selected_ids = inc_set | dec_set
            selected = [s for s in group if id(s) in selected_ids]
            return selected
        
        strategies['adv_inc_dec_S3'] = ('对抗16+16 (R_inc+R_dec), 用S3',
            adversarial_select, lambda s: s['S3'])
        
        # 策略3: 守恒门控 — 只保留满足 I_dec > I_inc 的样本, 用S3
        def cons_gate(group):
            return [s for s in group if s['R_cons_C'] > 0] or group[:16]
        
        strategies['cons_gate_S3'] = ('守恒门控 + S3',
            cons_gate, lambda s: s['S3'])
        
        # 策略4: 多目标归一化 GRPO
        # R = norm(R_inc) + norm(R_dec) + norm(R_cons) (组内归一化)
        def multi_obj_reward(group):
            """对组内样本计算多目标归一化奖励"""
            inc_vals = np.array([s['R_inc_D'] for s in group])
            dec_vals = np.array([s['R_dec_C'] for s in group])
            cons_vals = np.array([s['R_cons_B'] for s in group])
            
            def norm(x):
                m, sd = np.mean(x), np.std(x)
                if sd < 1e-10:
                    return np.zeros_like(x)
                return (x - m) / sd
            
            combined = norm(inc_vals) + norm(dec_vals) + norm(cons_vals)
            
            for s, c in zip(group, combined):
                s['_multi_obj'] = c
            
            return group
        
        strategies['multi_obj'] = ('多目标归一化 (norm_inc+norm_dec+norm_cons)',
            multi_obj_reward, lambda s: s.get('_multi_obj', 0))
        
        # 策略5: 对抗性选择 + 多目标归一化
        def adv_multi(group):
            selected = adversarial_select(group)
            return multi_obj_reward(selected)
        
        strategies['adv_multi'] = ('对抗16+16 + 多目标归一化',
            adv_multi, lambda s: s.get('_multi_obj', 0))
        
        # 策略6: 双批对抗 — 模拟"生成两次"
        # 将32样本随机分成两组16, 第一组用R_inc选top8, 第二组用R_dec选top8
        # 合并16个, 用S3训练
        def dual_batch_sim(group):
            """模拟双批生成: 随机分成两半, 各按不同标准选"""
            np.random.seed(42)  # 固定种子
            half = len(group) // 2
            indices = np.random.permutation(len(group))
            batch1 = [group[i] for i in indices[:half]]
            batch2 = [group[i] for i in indices[half:]]
            # Batch1: top half by R_inc
            batch1_sel = sorted(batch1, key=lambda s: s['R_inc_D'], reverse=True)[:half//2]
            # Batch2: top half by R_dec
            batch2_sel = sorted(batch2, key=lambda s: s['R_dec_C'], reverse=True)[:half//2]
            return batch1_sel + batch2_sel
        
        strategies['dual_batch'] = ('双批模拟: 各16→8, 合16, 用S3',
            dual_batch_sim, lambda s: s['S3'])
        
        # 策略7: Pareto选择 — 只保留非支配样本
        def pareto_select(group):
            """选择在(R_inc, R_dec)空间中的Pareto非支配样本"""
            inc = np.array([s['R_inc_D'] for s in group])
            dec = np.array([s['R_dec_C'] for s in group])
            n = len(group)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and inc[j] >= inc[i] and dec[j] >= dec[i] and \
                       (inc[j] > inc[i] or dec[j] > dec[i]):
                        dominated[i] = True
                        break
            pareto = [group[i] for i in range(n) if not dominated[i]]
            if len(pareto) < 4:
                pareto = sorted(group, key=lambda s: s['S3'], reverse=True)[:16]
            return pareto
        
        strategies['pareto_S3'] = ('Pareto非支配选择 + S3',
            pareto_select, lambda s: s['S3'])
        
        # 评估每个策略
        print(f"\n  {'策略':<45} {'GRPO Gap':>10} {'有效样本':>8}")
        print(f"  {'-'*65}")
        
        for key in ['std_32_S3', 'std_32_F4', 'adv_inc_dec_S3', 'cons_gate_S3',
                     'multi_obj', 'adv_multi', 'dual_batch', 'pareto_S3']:
            desc, select_fn, reward_fn = strategies[key]
            
            # 应用选择和奖励
            c_adv, i_adv = [], []
            total_selected = 0
            
            for pid, group in groups.items():
                if len(group) < 8:
                    continue
                selected = select_fn(list(group))
                total_selected += len(selected)
                
                vals = [reward_fn(s) for s in selected]
                m, sd = np.mean(vals), np.std(vals)
                if sd < 1e-10:
                    continue
                for s, v in zip(selected, vals):
                    a = (v - m) / sd
                    (c_adv if s['is_correct'] else i_adv).append(a)
            
            gap = (np.mean(c_adv) - np.mean(i_adv)) if c_adv and i_adv else 0
            n_sel = total_selected // max(len(groups), 1)
            
            marker = '★' if gap > 0.15 else ('↑' if gap > 0.1 else ' ')
            print(f"  {marker} {desc:<43} {gap:>+.4f}   ~{n_sel}/组")
        
        # ==============================================================
        # PART 3: 对抗性分解的理论分析
        # ==============================================================
        print(f"\n  ─── Part 3: 正确/错误样本在 (R_inc, R_dec) 空间的分布 ───")
        
        # 统计正确vs错误样本在(R_inc, R_dec)各象限的分布
        all_inc = np.array([s['R_inc_D'] for s in samples])
        all_dec = np.array([s['R_dec_C'] for s in samples])
        
        # 组内中位数
        for pid, group in groups.items():
            inc_med = np.median([s['R_inc_D'] for s in group])
            dec_med = np.median([s['R_dec_C'] for s in group])
            for s in group:
                s['_inc_above'] = s['R_inc_D'] > inc_med
                s['_dec_above'] = s['R_dec_C'] > dec_med
        
        # 四象限统计
        quadrants = {
            '高探索+高压缩': (True, True),
            '高探索+低压缩': (True, False),
            '低探索+高压缩': (False, True),
            '低探索+低压缩': (False, False),
        }
        
        print(f"\n  {'象限':<20} {'正确率':>8} {'正确数':>6} {'总数':>6}")
        print(f"  {'-'*45}")
        
        for qname, (inc_high, dec_high) in quadrants.items():
            q_samples = [s for s in samples if 
                         s.get('_inc_above', False) == inc_high and 
                         s.get('_dec_above', False) == dec_high]
            n_q = len(q_samples)
            n_c_q = sum(1 for s in q_samples if s['is_correct'])
            rate = n_c_q / max(n_q, 1)
            print(f"  {qname:<20} {rate:>7.1%}  {n_c_q:>5}  {n_q:>5}")
        
        # ==============================================================
        # PART 4: 对抗互补性分析
        # 对抗有价值的前提: R_inc top16 和 R_dec top16 重叠度低
        # ==============================================================
        print(f"\n  ─── Part 4: 选择重叠度 (对抗性价值) ───")
        
        overlaps = []
        for pid, group in groups.items():
            if len(group) < 16:
                continue
            by_inc = sorted(range(len(group)), key=lambda i: group[i]['R_inc_D'], reverse=True)[:len(group)//2]
            by_dec = sorted(range(len(group)), key=lambda i: group[i]['R_dec_C'], reverse=True)[:len(group)//2]
            overlap = len(set(by_inc) & set(by_dec))
            overlaps.append(overlap / (len(group) // 2))
        
        mean_overlap = np.mean(overlaps)
        print(f"  R_inc top50% 与 R_dec top50% 的平均重叠率: {mean_overlap:.1%}")
        if mean_overlap > 0.7:
            print(f"  → 高重叠 ({mean_overlap:.0%}): R_inc 和 R_dec 高度一致, 对抗分解价值不大")
        elif mean_overlap < 0.3:
            print(f"  → 低重叠 ({mean_overlap:.0%}): R_inc 和 R_dec 存在张力, 对抗分解有价值!")
        else:
            print(f"  → 中等重叠 ({mean_overlap:.0%}): 存在一定互补性")


if __name__ == '__main__':
    main()
