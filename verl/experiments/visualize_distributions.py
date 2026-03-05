"""
Visualization functions for logit distribution comparison.
Six core visualizations:
  V1: Side-by-side Top-20 Bar Chart
  V2: Entropy Trajectory Along Sequence
  V3: Zipf Plot (Rank-Probability)
  V4: Runner-up Protection Scatter
  V5: Effective Vocab Size Box Plot
  V6: KL Divergence Heatmap

Style: consistent with analyze_step_entropy.py
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

STD_COLOR = "#3498DB"   # blue — Standard GRPO (Baseline)
SR_COLOR = "#E74C3C"    # red  — SurprisalRedistribution-GRPO
AUX_GREEN = "#2ECC71"
AUX_PURPLE = "#9B59B6"


class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig, filename: str):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path}")

    # =================================================================
    # V1: Side-by-side Top-20 Bar Chart
    # =================================================================
    def plot_top_k_comparison(
        self,
        top_tokens_std: List[Tuple[str, float]],
        top_tokens_sr: List[Tuple[str, float]],
        title: str = "Top-20 Token Probabilities",
        filename: str = "top20_comparison.png",
        k: int = 20,
    ):
        top_std = top_tokens_std[:k]
        top_sr = top_tokens_sr[:k]

        all_tokens_ordered = []
        seen = set()
        for tok, _ in top_std:
            if tok not in seen:
                all_tokens_ordered.append(tok)
                seen.add(tok)
        for tok, _ in top_sr:
            if tok not in seen:
                all_tokens_ordered.append(tok)
                seen.add(tok)
        all_tokens_ordered = all_tokens_ordered[:k]

        std_map = {t: p for t, p in top_std}
        sr_map = {t: p for t, p in top_sr}

        labels = [repr(t).strip("'") if len(t.strip()) <= 12
                  else repr(t[:10]).strip("'") + ".."
                  for t in all_tokens_ordered]
        std_vals = [std_map.get(t, 0.0) for t in all_tokens_ordered]
        sr_vals = [sr_map.get(t, 0.0) for t in all_tokens_ordered]

        x = np.arange(len(labels))
        width = 0.38

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width / 2, std_vals, width,
               label="Standard GRPO", color=STD_COLOR, alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width / 2, sr_vals, width,
               label="SR-GRPO", color=SR_COLOR, alpha=0.8,
               edgecolor='black', linewidth=0.5)

        ax.set_xlabel("Token", fontsize=12, fontweight='bold')
        ax.set_ylabel("Probability", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        self._save(fig, filename)

    # =================================================================
    # V2: Entropy Trajectory Along Sequence
    # =================================================================
    def plot_entropy_trajectory(
        self,
        entropy_std: List[float],
        entropy_sr: List[float],
        title: str = "Entropy Trajectory",
        filename: str = "entropy_trajectory.png",
        divergence_pos: Optional[int] = None,
        window: int = 10,
    ):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        xs_std = np.arange(len(entropy_std))
        xs_sr = np.arange(len(entropy_sr))
        ax.plot(xs_std, entropy_std, color=STD_COLOR, alpha=0.3, linewidth=0.5)
        ax.plot(xs_sr, entropy_sr, color=SR_COLOR, alpha=0.3, linewidth=0.5)

        if len(entropy_std) >= window:
            smooth_std = np.convolve(entropy_std, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(len(smooth_std)) + window // 2, smooth_std,
                    color=STD_COLOR, linewidth=2.5, label="Standard GRPO (smoothed)",
                    marker='o', markersize=2)
        if len(entropy_sr) >= window:
            smooth_sr = np.convolve(entropy_sr, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(len(smooth_sr)) + window // 2, smooth_sr,
                    color=SR_COLOR, linewidth=2.5, label="SR-GRPO (smoothed)",
                    marker='o', markersize=2)

        if divergence_pos is not None:
            ax.axvline(divergence_pos, color="gray", linestyle="--", linewidth=2,
                       alpha=0.7, label=f"Divergence @ {divergence_pos}")

        ax.set_ylabel("Entropy (nats)", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        min_len = min(len(entropy_std), len(entropy_sr))
        diff = np.array(entropy_sr[:min_len]) - np.array(entropy_std[:min_len])
        colors = [SR_COLOR if d > 0 else STD_COLOR for d in diff]
        ax2.bar(range(min_len), diff, color=colors, alpha=0.6, width=1.0)
        ax2.axhline(0, color="black", linewidth=1)
        ax2.set_xlabel("Token Position", fontsize=12, fontweight='bold')
        ax2.set_ylabel("SR − STD", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, filename)

    # =================================================================
    # V3: Zipf Plot (Rank-Probability)
    # =================================================================
    def plot_zipf(
        self,
        rank_probs_std: np.ndarray,
        rank_probs_sr: np.ndarray,
        title: str = "Rank-Probability (Zipf) Plot",
        filename: str = "zipf_plot.png",
    ):
        fig, ax = plt.subplots(figsize=(10, 7))

        ranks = np.arange(1, len(rank_probs_std) + 1)
        ax.plot(ranks, rank_probs_std, color=STD_COLOR, linewidth=2.5,
                label="Standard GRPO", marker="o", markersize=3)
        ax.plot(ranks, rank_probs_sr, color=SR_COLOR, linewidth=2.5,
                label="SR-GRPO", marker="s", markersize=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank (log scale)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Probability (log scale)", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which="both")

        if len(rank_probs_std) >= 50 and len(rank_probs_sr) >= 50:
            for probs, color, name in [
                (rank_probs_std, STD_COLOR, "STD"),
                (rank_probs_sr, SR_COLOR, "SR"),
            ]:
                log_r = np.log(ranks[1:50])
                log_p = np.log(probs[1:50] + 1e-30)
                slope = np.polyfit(log_r, log_p, 1)[0]
                ax.annotate(f"{name} slope={slope:.2f}",
                            xy=(0.6, 0.95 if name == "STD" else 0.88),
                            xycoords="axes fraction", color=color,
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white', alpha=0.7))

        fig.tight_layout()
        self._save(fig, filename)

    # =================================================================
    # V4: Runner-up Protection Scatter
    # =================================================================
    def plot_runner_up_scatter(
        self,
        ru_std: List[float],
        ru_sr: List[float],
        title: str = "Runner-up Protection (π₂ / π₁)",
        filename: str = "runner_up_scatter.png",
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(ru_std, ru_sr, alpha=0.15, s=10,
                   color=AUX_PURPLE, edgecolors="none")

        lim = max(max(ru_std), max(ru_sr)) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, alpha=0.5, label="y = x")
        ax.set_xlim(0, min(lim, 1.0))
        ax.set_ylim(0, min(lim, 1.0))
        ax.set_xlabel("Standard GRPO: π₂ / π₁", fontsize=12, fontweight='bold')
        ax.set_ylabel("SR-GRPO: π₂ / π₁", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect("equal")

        above = sum(1 for s, r in zip(ru_std, ru_sr) if r > s)
        total = len(ru_std)
        ax.annotate(f"Above y=x: {above}/{total} ({above / total * 100:.1f}%)",
                    xy=(0.05, 0.92), xycoords="axes fraction", fontsize=11,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="wheat", alpha=0.7))

        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, filename)

    # =================================================================
    # V5: Effective Vocabulary Size Box Plot
    # =================================================================
    def plot_effective_vocab_box(
        self,
        ev_std: List[float],
        ev_sr: List[float],
        title: str = "Effective Vocabulary Size = exp(H)",
        filename: str = "effective_vocab_box.png",
    ):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax = axes[0]
        bp = ax.boxplot(
            [ev_std, ev_sr],
            labels=["Standard GRPO", "SR-GRPO"],
            patch_artist=True,
            widths=0.5,
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor(STD_COLOR)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(SR_COLOR)
        bp["boxes"][1].set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2.5)

        ax.set_ylabel("Effective Vocab Size", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis="y")

        mean_std, mean_sr = np.mean(ev_std), np.mean(ev_sr)
        ax.annotate(f"Mean: {mean_std:.1f}", xy=(1, mean_std),
                    fontsize=10, fontweight='bold', color=STD_COLOR,
                    ha="center", va="bottom")
        ax.annotate(f"Mean: {mean_sr:.1f}", xy=(2, mean_sr),
                    fontsize=10, fontweight='bold', color=SR_COLOR,
                    ha="center", va="bottom")

        ax2 = axes[1]
        clip_val = np.percentile(ev_std + ev_sr, 99)
        ev_std_c = [v for v in ev_std if v <= clip_val]
        ev_sr_c = [v for v in ev_sr if v <= clip_val]
        bins = np.linspace(0, clip_val, 80)
        ax2.hist(ev_std_c, bins=bins, color=STD_COLOR, alpha=0.5,
                 label="Standard GRPO", density=True, edgecolor='black', linewidth=0.3)
        ax2.hist(ev_sr_c, bins=bins, color=SR_COLOR, alpha=0.5,
                 label="SR-GRPO", density=True, edgecolor='black', linewidth=0.3)
        ax2.set_xlabel("Effective Vocab Size", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Density", fontsize=12, fontweight='bold')
        ax2.set_title("Distribution of Effective Vocab Size",
                       fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, filename)

    # =================================================================
    # V6: KL Divergence Heatmap
    # =================================================================
    def plot_kl_heatmap(
        self,
        kl_lists: List[List[float]],
        title: str = "KL(STD || SR) per Position",
        filename: str = "kl_heatmap.png",
    ):
        max_len = max(len(kl) for kl in kl_lists)
        matrix = np.full((len(kl_lists), max_len), np.nan)
        for i, kl in enumerate(kl_lists):
            matrix[i, :len(kl)] = kl

        fig, axes = plt.subplots(2, 1,
                                 figsize=(16, 4 + len(kl_lists) * 0.8),
                                 gridspec_kw={"height_ratios": [max(len(kl_lists), 1), 1]})

        ax = axes[0]
        vmax = np.nanpercentile(matrix, 98)
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xlabel("Token Position", fontsize=12, fontweight='bold')
        ax.set_ylabel("Sample", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(kl_lists)))
        ax.set_yticklabels([f"S{i + 1}" for i in range(len(kl_lists))])
        plt.colorbar(im, ax=ax, label="KL divergence", shrink=0.8)

        ax2 = axes[1]
        mean_kl = np.nanmean(matrix, axis=0)
        ax2.fill_between(range(len(mean_kl)), mean_kl, alpha=0.4, color="darkorange")
        ax2.plot(range(len(mean_kl)), mean_kl, color="darkorange", linewidth=2)
        ax2.set_xlabel("Token Position", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Mean KL", fontsize=12, fontweight='bold')
        ax2.set_xlim(0, max_len)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, filename)
