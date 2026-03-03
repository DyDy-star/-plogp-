import json
import numpy as np

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_sigma_pos(H_arr):
    n = len(H_arr)
    if n < 10:
        return None
    rho = np.corrcoef(H_arr[:-1], H_arr[1:])[0, 1]
    if np.isnan(rho): rho = 0.0
    rho = max(0.05, min(0.95, rho))
    alpha = 1.0 - rho
    ema = np.zeros(n)
    epsilon = np.zeros(n)
    ema[0] = H_arr[0]
    for t in range(1, n):
        epsilon[t] = H_arr[t] - ema[t - 1]
        ema[t] = alpha * H_arr[t] + (1 - alpha) * ema[t - 1]
    pos_eps = epsilon[epsilon > 0]
    if len(pos_eps) < 3:
        return None
    sigma_pos = np.std(pos_eps)
    if sigma_pos < 1e-6:
        return None
    return sigma_pos

def analyze(data, label):
    print(f"\n{'='*80}")
    print(f"  {label}  (pass@1={data['overall_pass@1']:.4f})")
    print(f"{'='*80}")

    correct_sigmas = []
    wrong_sigmas = []
    all_sigmas = []

    for problem in data['results']:
        for resp in problem['responses']:
            ea = resp.get('entropy_analysis')
            if not ea or 'steps' not in ea:
                continue
            all_ents = []
            for step in ea['steps']:
                all_ents.extend(step.get('token_entropies', []))
            if len(all_ents) < 20:
                continue
            sp = compute_sigma_pos(np.array(all_ents))
            if sp is None:
                continue
            all_sigmas.append(sp)
            if resp['is_correct']:
                correct_sigmas.append(sp)
            else:
                wrong_sigmas.append(sp)

    all_sigmas = np.array(all_sigmas)
    correct_sigmas = np.array(correct_sigmas)
    wrong_sigmas = np.array(wrong_sigmas)

    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    print(f"\n  全部 (N={len(all_sigmas)}):")
    print(f"    min={all_sigmas.min():.6f}  max={all_sigmas.max():.6f}")
    print(f"    mean={all_sigmas.mean():.6f}  std={all_sigmas.std():.6f}")
    print(f"    分位数:")
    for p in pcts:
        print(f"      {p:>3}%: {np.percentile(all_sigmas, p):.6f}")

    print(f"\n  正确 (N={len(correct_sigmas)}):")
    print(f"    min={correct_sigmas.min():.6f}  max={correct_sigmas.max():.6f}")
    print(f"    mean={correct_sigmas.mean():.6f}  std={correct_sigmas.std():.6f}")
    print(f"    分位数:")
    for p in pcts:
        print(f"      {p:>3}%: {np.percentile(correct_sigmas, p):.6f}")

    print(f"\n  错误 (N={len(wrong_sigmas)}):")
    print(f"    min={wrong_sigmas.min():.6f}  max={wrong_sigmas.max():.6f}")
    print(f"    mean={wrong_sigmas.mean():.6f}  std={wrong_sigmas.std():.6f}")
    print(f"    分位数:")
    for p in pcts:
        print(f"      {p:>3}%: {np.percentile(wrong_sigmas, p):.6f}")

    print(f"\n  直方图 (全部):")
    edges = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 5.0, 100]
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        n_all = np.sum((all_sigmas >= lo) & (all_sigmas < hi))
        n_c = np.sum((correct_sigmas >= lo) & (correct_sigmas < hi))
        n_w = np.sum((wrong_sigmas >= lo) & (wrong_sigmas < hi))
        bar = '#' * int(n_all / max(len(all_sigmas),1) * 100)
        print(f"    [{lo:>5.2f}, {hi:>5.2f})  全部:{n_all:>4} ({n_all/len(all_sigmas):>5.1%})  "
              f"正确:{n_c:>3} ({n_c/max(len(correct_sigmas),1):>5.1%})  "
              f"错误:{n_w:>3} ({n_w/max(len(wrong_sigmas),1):>5.1%})  {bar}")

data_later = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_092147.json")
data_early = load_data("/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/eval_results_aime_full_entropy/aime_eval_full_entropy_20260207_090427.json")

analyze(data_later, "后期数据")
analyze(data_early, "早期数据")
