"""
Logit Distribution Comparison: Standard GRPO vs SurprisalRedistribution-GRPO

Usage:
    python compare_logits.py \
        --model_std  /path/to/standard_grpo_hf_merged \
        --model_sr   /path/to/sr_grpo_hf_merged \
        --data_path  /path/to/AIME-TTT/test.parquet \
        --mode both \
        --max_samples 5 \
        --output_dir experiments/results

Optional:
    --model_base /path/to/base_model   (for KL-from-base analysis)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_distributions import Visualizer

# ---- Default paths ----
DEFAULT_STD_MODEL = (
    "/data/user5/TTRL begin/verl/checkpoints/TTRL-verl/"
    "AIME-TTT-Qwen2.5-Math-1.5B/0303/GT-baseline-Len@3k-grpo-151743/"
    "global_step_240/actor/huggingface_merged"
)
DEFAULT_SR_MODEL = (
    "/data/user5/TTRL begin/verl/checkpoints/TTRL-verl/"
    "AIME-TTT-Qwen2.5-Math-1.5B/0303/GT-plogpPos-Len@3k-grpo-064004/"
    "global_step_240/actor/huggingface_merged"
)
DEFAULT_BASE_MODEL = "/data/user5/models/Qwen2.5-Math-1.5B"
DEFAULT_DATA_PATH = "/data/user5/TTRL/verl/data/AIME-TTT/test.parquet"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PositionMetrics:
    """Metrics collected at a single token position."""
    position: int
    entropy: float
    effective_vocab: float
    top1_prob: float
    top5_mass: float
    top10_mass: float
    runner_up_ratio: float          # pi_2 / pi_1
    entropy_concentration: float    # max(h_i) / mean(h_i)
    kl_from_base: Optional[float] = None
    top_tokens: List[Tuple[str, float]] = field(default_factory=list)
    top_entropy_contribs: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class SequenceResult:
    """All metrics for one prompt–response pair, for one model."""
    prompt_text: str
    response_text: str
    response_tokens: List[str]
    metrics: List[PositionMetrics]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_position_metrics(
    logits: torch.Tensor,
    tokenizer,
    base_logits: Optional[torch.Tensor] = None,
    top_k: int = 20,
) -> PositionMetrics:
    """Compute distribution metrics from a single position's logits (vocab_size,)."""
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        entropy = -(probs * log_probs).sum().item()
        effective_vocab = np.exp(entropy)

        sorted_probs, sorted_indices = probs.sort(descending=True)
        top1_prob = sorted_probs[0].item()
        top5_mass = sorted_probs[:5].sum().item()
        top10_mass = sorted_probs[:10].sum().item()

        runner_up_ratio = (sorted_probs[1] / (sorted_probs[0] + 1e-10)).item()

        h_i = -(probs * log_probs)  # per-token entropy contribution
        h_nonzero = h_i[h_i > 1e-12]
        if len(h_nonzero) > 0:
            entropy_concentration = (h_nonzero.max() / h_nonzero.mean()).item()
        else:
            entropy_concentration = 0.0

        kl_from_base = None
        if base_logits is not None:
            base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
            kl = (probs * (log_probs - base_log_probs)).sum().item()
            kl_from_base = max(0.0, kl)

        top_tokens = []
        top_entropy_contribs = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i].item()
            tok_str = tokenizer.decode([idx])
            top_tokens.append((tok_str, sorted_probs[i].item()))

        _, h_sorted_indices = h_i.sort(descending=True)
        for i in range(min(top_k, len(h_sorted_indices))):
            idx = h_sorted_indices[i].item()
            tok_str = tokenizer.decode([idx])
            top_entropy_contribs.append((tok_str, h_i[idx].item()))

    return PositionMetrics(
        position=0,
        entropy=entropy,
        effective_vocab=effective_vocab,
        top1_prob=top1_prob,
        top5_mass=top5_mass,
        top10_mass=top10_mass,
        runner_up_ratio=runner_up_ratio,
        entropy_concentration=entropy_concentration,
        kl_from_base=kl_from_base,
        top_tokens=top_tokens,
        top_entropy_contribs=top_entropy_contribs,
    )


def compute_kl_between_models(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> float:
    """KL(A || B) at a single position."""
    with torch.no_grad():
        log_a = F.log_softmax(logits_a.float(), dim=-1)
        probs_a = F.softmax(logits_a.float(), dim=-1)
        log_b = F.log_softmax(logits_b.float(), dim=-1)
        kl = (probs_a * (log_a - log_b)).sum().item()
    return max(0.0, kl)


def collect_rank_probs(logits: torch.Tensor, max_rank: int = 200) -> np.ndarray:
    """Return sorted probabilities for Zipf plot (top max_rank)."""
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=-1)
        sorted_probs, _ = probs.sort(descending=True)
    return sorted_probs[:max_rank].cpu().numpy()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(path: str, device: str = "cuda:0"):
    print(f"Loading model from {path} ...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aime_prompts(data_path: str, max_samples: int = 5) -> List[Dict]:
    """Load prompts from parquet. Returns list of dicts with 'prompt' and 'ground_truth'."""
    df = pd.read_parquet(data_path)
    samples = []
    for i, row in df.iterrows():
        if len(samples) >= max_samples:
            break
        prompt_field = row.get("prompt", None)
        if prompt_field is not None:
            if isinstance(prompt_field, str):
                prompt_text = prompt_field
            elif isinstance(prompt_field, list):
                prompt_text = prompt_field[-1]["content"] if isinstance(prompt_field[-1], dict) else str(prompt_field[-1])
            else:
                prompt_text = str(prompt_field)
        else:
            prompt_text = str(row.iloc[0])

        gt = ""
        rm_field = row.get("reward_model", None)
        if rm_field is not None:
            if isinstance(rm_field, dict):
                gt = rm_field.get("ground_truth", "")
            elif isinstance(rm_field, str):
                try:
                    rm_dict = json.loads(rm_field)
                    gt = rm_dict.get("ground_truth", "")
                except (json.JSONDecodeError, TypeError):
                    gt = rm_field

        samples.append({"prompt": prompt_text, "ground_truth": str(gt), "index": i})
    return samples


# ---------------------------------------------------------------------------
# Experiment 1: Teacher-Forcing
# ---------------------------------------------------------------------------

def generate_reference_response(
    model, tokenizer, prompt: str, max_new_tokens: int = 1024, device: str = "cuda:0"
) -> str:
    """Generate a reference response using greedy decoding."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def teacher_forcing_analysis(
    model,
    tokenizer,
    prompt: str,
    response: str,
    base_model=None,
    device: str = "cuda:0",
) -> SequenceResult:
    """Run teacher-forcing forward pass, collecting logits at each response position."""
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prefix + response

    encoding = tokenizer(full_text, return_tensors="pt").to(device)
    prefix_encoding = tokenizer(prefix, return_tensors="pt")
    prompt_len = prefix_encoding["input_ids"].shape[1]
    total_len = encoding["input_ids"].shape[1]
    response_len = total_len - prompt_len

    if response_len <= 0:
        return SequenceResult(prompt, response, [], [])

    with torch.no_grad():
        output = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )
        all_logits = output.logits[0]  # (seq_len, vocab_size)

        base_all_logits = None
        if base_model is not None:
            base_output = base_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            base_all_logits = base_output.logits[0]

    response_token_ids = encoding["input_ids"][0, prompt_len:].tolist()
    response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

    metrics_list = []
    for pos in range(response_len):
        logit_pos = prompt_len - 1 + pos  # logits at position p predict token p+1
        if logit_pos >= all_logits.shape[0]:
            break

        base_logit = base_all_logits[logit_pos] if base_all_logits is not None else None
        m = compute_position_metrics(
            all_logits[logit_pos],
            tokenizer,
            base_logits=base_logit,
        )
        m.position = pos
        metrics_list.append(m)

    return SequenceResult(prompt, response, response_tokens, metrics_list)


# ---------------------------------------------------------------------------
# Experiment 2: Free Generation
# ---------------------------------------------------------------------------

def free_generation_analysis(
    model,
    tokenizer,
    prompt: str,
    base_model=None,
    max_new_tokens: int = 1024,
    device: str = "cuda:0",
) -> SequenceResult:
    """Generate freely and collect logits at each step using model.generate with output_scores."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][prompt_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    response_tokens = [tokenizer.decode([tid]) for tid in generated_ids.tolist()]
    scores = outputs.scores  # tuple of (vocab_size,) tensors

    base_logits_list = None
    if base_model is not None:
        full_ids = outputs.sequences
        with torch.no_grad():
            base_output = base_model(
                input_ids=full_ids.to(device),
                attention_mask=torch.ones_like(full_ids).to(device),
            )
        base_logits_list = base_output.logits[0]

    metrics_list = []
    for pos, score in enumerate(scores):
        logits_pos = score[0] if score.dim() > 1 else score
        base_logit = None
        if base_logits_list is not None:
            base_idx = prompt_len - 1 + pos
            if base_idx < base_logits_list.shape[0]:
                base_logit = base_logits_list[base_idx]

        m = compute_position_metrics(logits_pos, tokenizer, base_logits=base_logit)
        m.position = pos
        metrics_list.append(m)

    return SequenceResult(prompt, response_text, response_tokens, metrics_list)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_metrics(results: List[SequenceResult]) -> Dict[str, List[float]]:
    """Flatten all position metrics across all sequences."""
    agg = {
        "entropy": [], "effective_vocab": [], "top1_prob": [],
        "top5_mass": [], "top10_mass": [], "runner_up_ratio": [],
        "entropy_concentration": [], "kl_from_base": [],
    }
    for seq in results:
        for m in seq.metrics:
            agg["entropy"].append(m.entropy)
            agg["effective_vocab"].append(m.effective_vocab)
            agg["top1_prob"].append(m.top1_prob)
            agg["top5_mass"].append(m.top5_mass)
            agg["top10_mass"].append(m.top10_mass)
            agg["runner_up_ratio"].append(m.runner_up_ratio)
            agg["entropy_concentration"].append(m.entropy_concentration)
            if m.kl_from_base is not None:
                agg["kl_from_base"].append(m.kl_from_base)
    return agg


def compute_aggregate_stats(agg_std: dict, agg_sr: dict) -> pd.DataFrame:
    """Produce a summary table with mean, std, and Wilcoxon p-value."""
    from scipy.stats import wilcoxon

    rows = []
    for key in ["entropy", "effective_vocab", "top1_prob", "top5_mass",
                 "top10_mass", "runner_up_ratio", "entropy_concentration", "kl_from_base"]:
        vals_std = np.array(agg_std.get(key, []))
        vals_sr = np.array(agg_sr.get(key, []))
        if len(vals_std) == 0 or len(vals_sr) == 0:
            continue
        min_len = min(len(vals_std), len(vals_sr))
        vals_std = vals_std[:min_len]
        vals_sr = vals_sr[:min_len]
        try:
            _, p_val = wilcoxon(vals_std, vals_sr)
        except Exception:
            p_val = float("nan")

        rows.append({
            "Metric": key,
            "STD_mean": f"{vals_std.mean():.4f}",
            "STD_std": f"{vals_std.std():.4f}",
            "SR_mean": f"{vals_sr.mean():.4f}",
            "SR_std": f"{vals_sr.std():.4f}",
            "Diff%": f"{(vals_sr.mean() - vals_std.mean()) / (abs(vals_std.mean()) + 1e-10) * 100:+.2f}%",
            "p_value": f"{p_val:.4e}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KL between models (per position)
# ---------------------------------------------------------------------------

def compute_pairwise_kl_teacher_forcing(
    model_a, model_b, tokenizer, prompt: str, response: str, device: str = "cuda:0"
) -> List[float]:
    """Compute KL(A||B) at each response position under teacher forcing."""
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prefix + response
    encoding = tokenizer(full_text, return_tensors="pt").to(device)
    prefix_encoding = tokenizer(prefix, return_tensors="pt")
    prompt_len = prefix_encoding["input_ids"].shape[1]
    total_len = encoding["input_ids"].shape[1]

    with torch.no_grad():
        logits_a = model_a(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]).logits[0]
        logits_b = model_b(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]).logits[0]

    kl_list = []
    for pos in range(total_len - prompt_len):
        logit_idx = prompt_len - 1 + pos
        if logit_idx >= logits_a.shape[0]:
            break
        kl_list.append(compute_kl_between_models(logits_a[logit_idx], logits_b[logit_idx]))
    return kl_list


# ---------------------------------------------------------------------------
# Zipf data collection
# ---------------------------------------------------------------------------

def collect_zipf_data_tf(
    model, tokenizer, prompt: str, response: str,
    positions: List[int], device: str = "cuda:0", max_rank: int = 200,
) -> Dict[int, np.ndarray]:
    """Collect rank-probability arrays at specified positions under teacher forcing."""
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prefix + response
    encoding = tokenizer(full_text, return_tensors="pt").to(device)
    prefix_len = tokenizer(prefix, return_tensors="pt")["input_ids"].shape[1]

    with torch.no_grad():
        logits = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]).logits[0]

    result = {}
    for pos in positions:
        logit_idx = prefix_len - 1 + pos
        if logit_idx < logits.shape[0]:
            result[pos] = collect_rank_probs(logits[logit_idx], max_rank)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare logit distributions: Standard GRPO vs SR-GRPO")
    parser.add_argument("--model_std", type=str, default=DEFAULT_STD_MODEL,
                        help="Path to standard GRPO HF-merged checkpoint")
    parser.add_argument("--model_sr", type=str, default=DEFAULT_SR_MODEL,
                        help="Path to SR-GRPO HF-merged checkpoint")
    parser.add_argument("--model_base", type=str, default=DEFAULT_BASE_MODEL,
                        help="Path to base model (for KL-from-base)")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to AIME24 parquet")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["teacher_forcing", "free_gen", "both"])
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    viz = Visualizer(args.output_dir)

    # -- Load models --
    model_std, tok_std = load_model_and_tokenizer(args.model_std, args.device)
    model_sr, tok_sr = load_model_and_tokenizer(args.model_sr, args.device)
    tokenizer = tok_std

    base_model = None
    if args.model_base:
        base_model, _ = load_model_and_tokenizer(args.model_base, args.device)

    # -- Load data --
    samples = load_aime_prompts(args.data_path, args.max_samples)
    print(f"Loaded {len(samples)} AIME prompts")

    # ===================================================================
    # Experiment 1: Teacher-Forcing
    # ===================================================================
    if args.mode in ("teacher_forcing", "both"):
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: Teacher-Forcing Analysis")
        print("=" * 70)

        tf_results_std = []
        tf_results_sr = []
        all_kl_between = []
        all_zipf_std = {}
        all_zipf_sr = {}

        for si, sample in enumerate(samples):
            print(f"\n--- Sample {si + 1}/{len(samples)} ---")
            prompt = sample["prompt"]

            ref_model = base_model if base_model is not None else model_std
            ref_response = generate_reference_response(
                ref_model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens, device=args.device,
            )
            print(f"  Reference response length: {len(tokenizer.encode(ref_response))} tokens")

            res_std = teacher_forcing_analysis(
                model_std, tokenizer, prompt, ref_response,
                base_model=base_model, device=args.device,
            )
            res_sr = teacher_forcing_analysis(
                model_sr, tokenizer, prompt, ref_response,
                base_model=base_model, device=args.device,
            )
            tf_results_std.append(res_std)
            tf_results_sr.append(res_sr)

            # V6: KL between the two models
            kl_list = compute_pairwise_kl_teacher_forcing(
                model_std, model_sr, tokenizer, prompt, ref_response, device=args.device,
            )
            all_kl_between.append(kl_list)

            # V2: Entropy trajectory
            if res_std.metrics and res_sr.metrics:
                viz.plot_entropy_trajectory(
                    [m.entropy for m in res_std.metrics],
                    [m.entropy for m in res_sr.metrics],
                    title=f"TF Sample {si + 1}: Entropy Trajectory",
                    filename=f"tf_entropy_trajectory_s{si + 1}.png",
                )

            # V1: Top-20 bar chart at selected positions
            n_pos = len(res_std.metrics)
            key_positions = [0, n_pos // 4, n_pos // 2, 3 * n_pos // 4, n_pos - 1]
            key_positions = [p for p in key_positions if p < n_pos]
            for kp in key_positions[:3]:
                viz.plot_top_k_comparison(
                    res_std.metrics[kp].top_tokens,
                    res_sr.metrics[kp].top_tokens,
                    title=f"TF S{si + 1} Pos {kp}: Top-20 Token Probabilities",
                    filename=f"tf_top20_s{si + 1}_pos{kp}.png",
                )

            # V3: Zipf data
            zipf_positions = key_positions[:3]
            zipf_std = collect_zipf_data_tf(
                model_std, tokenizer, prompt, ref_response, zipf_positions, args.device,
            )
            zipf_sr = collect_zipf_data_tf(
                model_sr, tokenizer, prompt, ref_response, zipf_positions, args.device,
            )
            for zp in zipf_positions:
                if zp in zipf_std and zp in zipf_sr:
                    viz.plot_zipf(
                        zipf_std[zp], zipf_sr[zp],
                        title=f"TF S{si + 1} Pos {zp}: Rank-Probability (Zipf)",
                        filename=f"tf_zipf_s{si + 1}_pos{zp}.png",
                    )

        # V4: Runner-up scatter (aggregate across all samples)
        all_ru_std = [m.runner_up_ratio for r in tf_results_std for m in r.metrics]
        all_ru_sr = [m.runner_up_ratio for r in tf_results_sr for m in r.metrics]
        if all_ru_std and all_ru_sr:
            min_len = min(len(all_ru_std), len(all_ru_sr))
            viz.plot_runner_up_scatter(
                all_ru_std[:min_len], all_ru_sr[:min_len],
                title="TF: Runner-up Protection (pi_2/pi_1)",
                filename="tf_runner_up_scatter.png",
            )

        # V5: Effective vocab box plot
        all_ev_std = [m.effective_vocab for r in tf_results_std for m in r.metrics]
        all_ev_sr = [m.effective_vocab for r in tf_results_sr for m in r.metrics]
        if all_ev_std and all_ev_sr:
            viz.plot_effective_vocab_box(
                all_ev_std, all_ev_sr,
                title="TF: Effective Vocabulary Size",
                filename="tf_effective_vocab_box.png",
            )

        # V6: KL heatmap
        if all_kl_between:
            viz.plot_kl_heatmap(
                all_kl_between,
                title="TF: KL(STD || SR) per Position",
                filename="tf_kl_heatmap.png",
            )

        # Aggregate stats
        agg_std = aggregate_metrics(tf_results_std)
        agg_sr = aggregate_metrics(tf_results_sr)
        stats_df = compute_aggregate_stats(agg_std, agg_sr)
        print("\n" + "=" * 70)
        print("Teacher-Forcing Aggregate Statistics")
        print("=" * 70)
        print(stats_df.to_string(index=False))
        stats_df.to_csv(os.path.join(args.output_dir, "tf_aggregate_stats.csv"), index=False)

    # ===================================================================
    # Experiment 2: Free Generation
    # ===================================================================
    if args.mode in ("free_gen", "both"):
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Free Generation Analysis")
        print("=" * 70)

        fg_results_std = []
        fg_results_sr = []

        for si, sample in enumerate(samples):
            print(f"\n--- Sample {si + 1}/{len(samples)} ---")
            prompt = sample["prompt"]

            res_std = free_generation_analysis(
                model_std, tokenizer, prompt,
                base_model=base_model, max_new_tokens=args.max_new_tokens, device=args.device,
            )
            res_sr = free_generation_analysis(
                model_sr, tokenizer, prompt,
                base_model=base_model, max_new_tokens=args.max_new_tokens, device=args.device,
            )
            fg_results_std.append(res_std)
            fg_results_sr.append(res_sr)

            print(f"  STD response: {len(res_std.response_tokens)} tokens")
            print(f"  SR  response: {len(res_sr.response_tokens)} tokens")

            # Find divergence point
            div_pos = None
            min_len = min(len(res_std.response_tokens), len(res_sr.response_tokens))
            for p in range(min_len):
                if res_std.response_tokens[p] != res_sr.response_tokens[p]:
                    div_pos = p
                    break
            if div_pos is not None:
                print(f"  First divergence at position {div_pos}: "
                      f"STD='{res_std.response_tokens[div_pos]}' vs SR='{res_sr.response_tokens[div_pos]}'")

            # V2: Entropy trajectory
            if res_std.metrics and res_sr.metrics:
                viz.plot_entropy_trajectory(
                    [m.entropy for m in res_std.metrics],
                    [m.entropy for m in res_sr.metrics],
                    title=f"FG Sample {si + 1}: Entropy Trajectory",
                    filename=f"fg_entropy_trajectory_s{si + 1}.png",
                    divergence_pos=div_pos,
                )

        # V4: Runner-up scatter
        all_ru_std = [m.runner_up_ratio for r in fg_results_std for m in r.metrics]
        all_ru_sr = [m.runner_up_ratio for r in fg_results_sr for m in r.metrics]
        if all_ru_std and all_ru_sr:
            min_len = min(len(all_ru_std), len(all_ru_sr))
            viz.plot_runner_up_scatter(
                all_ru_std[:min_len], all_ru_sr[:min_len],
                title="FG: Runner-up Protection (pi_2/pi_1)",
                filename="fg_runner_up_scatter.png",
            )

        # V5: Effective vocab box
        all_ev_std = [m.effective_vocab for r in fg_results_std for m in r.metrics]
        all_ev_sr = [m.effective_vocab for r in fg_results_sr for m in r.metrics]
        if all_ev_std and all_ev_sr:
            viz.plot_effective_vocab_box(
                all_ev_std, all_ev_sr,
                title="FG: Effective Vocabulary Size",
                filename="fg_effective_vocab_box.png",
            )

        # Aggregate stats
        agg_std = aggregate_metrics(fg_results_std)
        agg_sr = aggregate_metrics(fg_results_sr)
        stats_df = compute_aggregate_stats(agg_std, agg_sr)
        print("\n" + "=" * 70)
        print("Free Generation Aggregate Statistics")
        print("=" * 70)
        print(stats_df.to_string(index=False))
        stats_df.to_csv(os.path.join(args.output_dir, "fg_aggregate_stats.csv"), index=False)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
