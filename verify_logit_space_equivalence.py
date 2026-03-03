"""
Verify that the -p*log(p) (entropy contribution) redistribution produces
correct gradients with L1 conservation, target preservation, numerical
stability, and proper runner-up protection.
"""

import torch
import numpy as np

torch.manual_seed(42)

V = 151936  # Qwen2.5 vocab size
ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
BATCH_SIZES = [1, 4, 16]


def plogp_redist(logits, tgt, grad_in, alpha):
    """Entropy-contribution redistribution: weight = -p*log(p)."""
    grad_out = grad_in.clone()
    with torch.no_grad():
        z = logits.float()
        lse = torch.logsumexp(z, dim=-1, keepdim=True)
        surp = lse - z
        p_local = torch.exp(z - lse)
        w = p_local * surp  # -p*log(p), bounded by 1/e

        w_at_tgt = w.gather(-1, tgt.unsqueeze(-1))
        w_total = w.sum(-1, keepdim=True) - w_at_tgt + 1e-10

        g = grad_out.float()
        g_tgt = g.gather(-1, tgt.unsqueeze(-1))
        g.scatter_(-1, tgt.unsqueeze(-1), 0.0)
        G_total = g.sum(-1, keepdim=True)

        g_desired = G_total * w / w_total
        g_desired.scatter_(-1, tgt.unsqueeze(-1), 0.0)

        g_new = (1.0 - alpha) * g + alpha * g_desired
        g_new.scatter_(-1, tgt.unsqueeze(-1), g_tgt)

        grad_out = g_new.to(grad_in.dtype)
    return grad_out


def make_test_case(batch_size, vocab_size, advantage_sign="positive", dtype=torch.float32):
    logits = torch.randn(batch_size, vocab_size, dtype=dtype)
    logits[:, 0] += 5.0
    targets = torch.zeros(batch_size, dtype=torch.long)
    p = torch.softmax(logits.float(), dim=-1)

    if advantage_sign == "positive":
        A = torch.randn(batch_size, 1).abs()
    elif advantage_sign == "negative":
        A = -torch.randn(batch_size, 1).abs()
    else:
        A = torch.randn(batch_size, 1)

    grad = A * p
    for i in range(batch_size):
        grad[i, targets[i]] = -A[i, 0] * (1.0 - p[i, targets[i]])
    return logits, targets, grad.to(dtype), A


all_pass = True

# ============================================================
# Test 1: L1 Conservation for positive, negative, and mixed advantages
# ============================================================
print("=" * 70)
print("Test 1: L1 Conservation (positive, negative, mixed)")
print("=" * 70)

for sign in ["positive", "negative", "mixed"]:
    for B in [4, 8]:
        for alpha in ALPHAS:
            logits, tgt, grad_in, A = make_test_case(B, V, sign)
            g_out = plogp_redist(logits, tgt, grad_in.clone(), alpha)

            max_l1_err = 0
            for i in range(B):
                mask = torch.ones(V, dtype=torch.bool)
                mask[tgt[i]] = False
                l1_in = grad_in[i, mask].sum().item()
                l1_out = g_out[i, mask].sum().item()
                err = abs(l1_in - l1_out) / (abs(l1_in) + 1e-20)
                max_l1_err = max(max_l1_err, err)

            passed = max_l1_err < 1e-5
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  {sign:>8}, B={B:>2}, alpha={alpha:.1f}: "
                  f"max L1 rel err = {max_l1_err:.2e}  [{status}]")

print()

# ============================================================
# Test 2: Target Gradient Preservation
# ============================================================
print("=" * 70)
print("Test 2: Target Gradient Preservation")
print("=" * 70)

for sign in ["positive", "negative", "mixed"]:
    for alpha in ALPHAS:
        logits, tgt, grad_in, A = make_test_case(8, V, sign)
        g_out = plogp_redist(logits, tgt, grad_in.clone(), alpha)

        max_tgt_err = 0
        for i in range(8):
            orig = grad_in[i, tgt[i]].item()
            after = g_out[i, tgt[i]].item()
            err = abs(orig - after) / (abs(orig) + 1e-20)
            max_tgt_err = max(max_tgt_err, err)

        passed = max_tgt_err < 1e-6
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {sign:>8}, alpha={alpha:.1f}: "
              f"max target grad err = {max_tgt_err:.2e}  [{status}]")

print()

# ============================================================
# Test 3: Weight boundedness — w = p * surp <= 1/e
# ============================================================
print("=" * 70)
print("Test 3: Weight boundedness (w = p*surp <= 1/e ≈ 0.3679)")
print("=" * 70)

for temp in [0.01, 0.1, 0.5, 1.0, 2.0]:
    logits = torch.randn(4, V) * (1.0 / temp)
    logits[:, 0] += 5.0 / temp
    z = logits.float()
    lse = torch.logsumexp(z, dim=-1, keepdim=True)
    surp = lse - z
    p_local = torch.exp(z - lse)
    w = p_local * surp

    max_w = w.max().item()
    passed = max_w <= 1.0 / np.e + 1e-6
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  temp={temp:.2f}: max(w) = {max_w:.6f}  [{status}]")

print()

# ============================================================
# Test 4: Runner-up protection — top tokens get less gradient share
# ============================================================
print("=" * 70)
print("Test 4: Runner-up protection")
print("=" * 70)

alpha = 0.5
for sign in ["positive", "negative"]:
    logits, tgt, grad_in, A = make_test_case(8, V, sign)
    g_std = grad_in.clone()
    g_mod = plogp_redist(logits, tgt, grad_in.clone(), alpha)

    p = torch.softmax(logits.float(), dim=-1)

    top_k = 100
    tail_k = 1000
    runner_ratios = []
    tail_ratios = []

    for i in range(8):
        pi = p[i].clone()
        pi[tgt[i]] = -1.0
        sorted_idx = pi.argsort(descending=True)
        runners = sorted_idx[:top_k]
        tails = sorted_idx[-tail_k:]

        r_std = g_std[i, runners].abs().mean().item()
        r_mod = g_mod[i, runners].abs().mean().item()
        t_std = g_std[i, tails].abs().mean().item()
        t_mod = g_mod[i, tails].abs().mean().item()

        runner_ratios.append(r_mod / (r_std + 1e-20))
        tail_ratios.append(t_mod / (t_std + 1e-20))

    avg_runner = np.mean(runner_ratios)
    avg_tail = np.mean(tail_ratios)

    runner_protected = avg_runner < 1.0
    tail_bounded = avg_tail < 5.0  # much stricter than -log(p) which was >>10

    passed = runner_protected and tail_bounded
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False

    print(f"  {sign:>8}: runner_ratio={avg_runner:.4f} (<1 = protected), "
          f"tail_ratio={avg_tail:.4f} (<5 = bounded)  [{status}]")

print()

# ============================================================
# Test 5: Stress test — peaked distributions, both signs
# ============================================================
print("=" * 70)
print("Test 5: Stress Test — peaked distributions, both signs")
print("=" * 70)

for temp in [0.01, 0.05, 0.1, 0.5]:
    for sign in ["positive", "negative"]:
        logits = torch.randn(4, V) * (1.0 / temp)
        logits[:, 0] += 10.0 / temp
        tgt = torch.zeros(4, dtype=torch.long)

        p = torch.softmax(logits.float(), dim=-1)
        A = torch.ones(4, 1) if sign == "positive" else -torch.ones(4, 1)
        grad_in = A * p
        for i in range(4):
            grad_in[i, tgt[i]] = -A[i, 0] * (1.0 - p[i, tgt[i]])

        g_out = plogp_redist(logits, tgt, grad_in.clone(), alpha=0.3)

        has_nan = torch.isnan(g_out).any().item()
        has_inf = torch.isinf(g_out).any().item()

        passed = not has_nan and not has_inf
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"  temp={temp:.2f}, {sign:>8}: nan={has_nan}, inf={has_inf}  [{status}]")

print()

# ============================================================
# Test 6: BFloat16 stability
# ============================================================
print("=" * 70)
print("Test 6: BFloat16 Stability")
print("=" * 70)

for sign in ["positive", "negative"]:
    for alpha in [0.3, 0.5, 1.0]:
        logits_f32 = torch.randn(8, V)
        logits_f32[:, 0] += 8.0
        logits_bf16 = logits_f32.bfloat16()
        tgt = torch.zeros(8, dtype=torch.long)

        p = torch.softmax(logits_f32, dim=-1)
        A = torch.ones(8, 1) if sign == "positive" else -torch.ones(8, 1)
        grad_f32 = A * p
        for i in range(8):
            grad_f32[i, tgt[i]] = -A[i, 0] * (1.0 - p[i, tgt[i]])
        grad_bf16 = grad_f32.bfloat16()

        g_out = plogp_redist(logits_bf16, tgt, grad_bf16.clone(), alpha)

        n_nan = torch.isnan(g_out).sum().item()
        n_inf = torch.isinf(g_out).sum().item()

        passed = n_nan == 0 and n_inf == 0
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"  {sign:>8}, alpha={alpha:.1f}: nan={n_nan}, inf={n_inf}  [{status}]")

print()
print("=" * 70)
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 70)
